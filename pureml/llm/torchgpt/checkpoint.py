from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from pureml.llm.tokenization import (
    BytePairTokenizer,
    CharacterTokenizer,
    TextTokenizer,
)
from pureml.llm.torchgpt.model import TinyGPT

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LoadedModelCheckpoint:
    model: TinyGPT
    tokenizer: TextTokenizer
    step: int
    best_validation_loss: float


@dataclass(frozen=True)
class LoadedTrainingCheckpoint(LoadedModelCheckpoint):
    optimizer: Optimizer
    scheduler: CosineAnnealingLR | None


def save_training_checkpoint(
    path: Path,
    *,
    model: TinyGPT,
    optimizer: Optimizer,
    scheduler: CosineAnnealingLR,
    tokenizer: TextTokenizer,
    step: int,
    best_validation_loss: float,
) -> None:
    if step < 0:
        raise ValueError("step must be non-negative")

    device = next(model.parameters()).device

    checkpoint: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": {
            "dropout": model.dropout,
            "vocab_size": model.vocab_size,
            "ctx_length": model.ctx_length,
            "embedding_dim": model.embedding_dim,
            "num_heads": model.num_heads,
            "num_layers": model.num_layers,
            "hidden_dim": model.hidden_dim,
            "init_std": model.init_std,
        },
        "model_state_dict": model.state_dict(),
        "optimizer_type": type(optimizer).__name__,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_type": type(scheduler).__name__,
        "scheduler_state_dict": scheduler.state_dict(),
        **_serialize_tokenizer(tokenizer),
        "step": step,
        "best_validation_loss": best_validation_loss,
        "cpu_rng_state": torch.get_rng_state(),
        "device_type": device.type,
        "device_rng_state": _get_device_rng_state(device),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def _serialize_tokenizer(tokenizer: TextTokenizer) -> dict[str, Any]:
    if isinstance(tokenizer, CharacterTokenizer):
        if tokenizer.id_to_char is None:
            raise ValueError("tokenizer must be fitted before saving a checkpoint")
        characters = "".join(
            tokenizer.id_to_char[token_id] for token_id in range(tokenizer.vocab_size)
        )
        return {
            "tokenizer_type": "character",
            "tokenizer_characters": characters,
        }

    if isinstance(tokenizer, BytePairTokenizer):
        return {
            "tokenizer_type": "byte_pair",
            "tokenizer_config": tokenizer.to_dict(),
        }

    raise TypeError(f"unsupported tokenizer: {type(tokenizer).__name__}")


def load_training_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> LoadedTrainingCheckpoint:
    checkpoint = _load_checkpoint_payload(path, device)
    if checkpoint.get("optimizer_type") != "AdamW":
        raise ValueError(f"unsupported optimizer: {checkpoint.get('optimizer_type')!r}")

    model, tokenizer = _load_model_and_tokenizer(checkpoint, device)

    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler = _load_scheduler(checkpoint, optimizer)

    torch.set_rng_state(checkpoint["cpu_rng_state"].cpu())
    _set_device_rng_state(
        device=device,
        saved_device_type=checkpoint["device_type"],
        rng_state=checkpoint["device_rng_state"],
    )

    return LoadedTrainingCheckpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        step=checkpoint["step"],
        best_validation_loss=checkpoint["best_validation_loss"],
    )


def _load_scheduler(
    checkpoint: dict[str, Any],
    optimizer: Optimizer,
) -> CosineAnnealingLR | None:
    scheduler_type = checkpoint.get("scheduler_type")
    scheduler_state = checkpoint.get("scheduler_state_dict")

    if scheduler_type is None and scheduler_state is None:
        return None
    if scheduler_type != "CosineAnnealingLR":
        raise ValueError(f"unsupported scheduler: {scheduler_type!r}")
    if not isinstance(scheduler_state, dict):
        raise ValueError("invalid scheduler state in checkpoint")

    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=scheduler_state["T_max"],
        eta_min=scheduler_state["eta_min"],
    )
    scheduler.load_state_dict(scheduler_state)
    return scheduler


def load_model_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> LoadedModelCheckpoint:
    checkpoint = _load_checkpoint_payload(path, device)
    model, tokenizer = _load_model_and_tokenizer(checkpoint, device)

    return LoadedModelCheckpoint(
        model=model,
        tokenizer=tokenizer,
        step=checkpoint["step"],
        best_validation_loss=checkpoint["best_validation_loss"],
    )


def _load_checkpoint_payload(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid checkpoint payload: {path}")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "unsupported checkpoint format version: "
            f"{checkpoint.get('format_version')!r}"
        )
    if checkpoint.get("tokenizer_type") not in ("character", "byte_pair"):
        raise ValueError(f"unsupported tokenizer: {checkpoint.get('tokenizer_type')!r}")

    return checkpoint


def _load_model_and_tokenizer(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[TinyGPT, TextTokenizer]:
    model = TinyGPT(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = _load_tokenizer(checkpoint)
    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError(
            "checkpoint tokenizer vocabulary size does not match the model"
        )

    return model, tokenizer


def _load_tokenizer(checkpoint: dict[str, Any]) -> TextTokenizer:
    if checkpoint["tokenizer_type"] == "character":
        return CharacterTokenizer().fit(checkpoint["tokenizer_characters"])

    tokenizer_config = checkpoint.get("tokenizer_config")
    if not isinstance(tokenizer_config, dict):
        raise ValueError("invalid byte-pair tokenizer config in checkpoint")
    return BytePairTokenizer.from_dict(tokenizer_config)


def _get_device_rng_state(device: torch.device) -> torch.Tensor | None:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def _set_device_rng_state(
    *,
    device: torch.device,
    saved_device_type: str,
    rng_state: torch.Tensor | None,
) -> None:
    if rng_state is None or saved_device_type != device.type:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng_state.cpu(), device)
    elif device.type == "mps":
        torch.mps.set_rng_state(rng_state.cpu())
