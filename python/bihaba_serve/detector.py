"""
Architecture detector & local model directory scanner.
Reads config.json to route models cleanly to the right isolated engine binary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Qwen3.8 dibangun di atas arsitektur Qwen3.5 (DeltaNet + gated attention); engine memvalidasi dimensinya sendiri.
QWEN35_ARCHS = {"qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text",
                "qwen3_8", "qwen3_8_moe", "qwen3_8_text", "qwen3_8_moe_text"}
TRANSFORMER_ARCHS = {"qwen2", "qwen3", "llama", "mistral"}


@dataclass
class ModelInfo:
    name: str
    path: Path
    model_type: str
    engine: str          # "qwen35" or "transformer"
    template: str        # "qwen35", "chatml", or "llama3"
    is_moe: bool = False
    num_layers: int = 0
    hidden_size: int = 0


def detect_model(model_path: str | Path) -> ModelInfo:
    p = Path(model_path).expanduser().resolve()
    if p.is_file() and p.suffix in (".taalas2", ".taalas"):
        eng = "qwen35" if p.suffix == ".taalas2" else "transformer"
        return ModelInfo(name=p.stem, path=p, model_type=eng, engine=eng, template="qwen35" if eng == "qwen35" else "chatml")

    cfg_path = p / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json found in {p}")

    cfg = json.loads(cfg_path.read_text())
    tc = cfg.get("text_config", cfg)
    mtype = str(tc.get("model_type") or cfg.get("model_type") or "").lower()
    archs = [a.lower() for a in cfg.get("architectures", [])]

    is_q35 = (mtype in QWEN35_ARCHS) or any("qwen3_5" in a or "qwen35" in a or "qwen3_8" in a for a in archs)
    is_tr = (mtype in TRANSFORMER_ARCHS)

    if is_q35:
        engine = "qwen35"
        template = "qwen35"
    elif is_tr:
        engine = "transformer"
        template = "llama3" if mtype == "llama" else ("qwen35" if mtype == "qwen3" else "chatml")
    else:
        raise ValueError(
            f"Unsupported model_type={mtype!r} in {p.name}. "
            f"Supported: Qwen3.5/MoE ({sorted(QWEN35_ARCHS)}) and Transformer ({sorted(TRANSFORMER_ARCHS)})."
        )

    return ModelInfo(
        name=p.name,
        path=p,
        model_type=mtype,
        engine=engine,
        template=template,
        is_moe=int(tc.get("num_experts", 0) or 0) > 0,
        num_layers=int(tc.get("num_hidden_layers", 0) or 0),
        hidden_size=int(tc.get("hidden_size", 0) or 0),
    )


def scan_models_dir(models_dir: str | Path, engine_filter: str | None = None) -> dict[str, ModelInfo]:
    root = Path(models_dir).expanduser().resolve()
    found: dict[str, ModelInfo] = {}
    if not root.is_dir():
        return found
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (child / "config.json").exists():
            continue
        if not list(child.glob("*.safetensors")):
            continue
        try:
            info = detect_model(child)
        except Exception:
            continue
        if engine_filter and info.engine != engine_filter:
            continue
        found[info.name] = info
    return found
