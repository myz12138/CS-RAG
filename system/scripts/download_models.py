"""Download pinned default models into this release, or check files offline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "embedding": ("BAAI/bge-small-zh-v1.5", "7999e1d3359715c523056ef9478215996d62a620"),
    "reranker": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
}


def check_files(directory: Path, kind: str) -> list[str]:
    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    if kind == "embedding":
        required.extend(["modules.json", "1_Pooling/config.json"])
    errors = []
    for name in required:
        path = directory / name
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            errors.append(f"缺失或损坏：{name}")
    weights = list(directory.glob("*.safetensors")) + list(directory.glob("pytorch_model*.bin"))
    if not weights or not all(path.stat().st_size > 1024 * 1024 for path in weights):
        errors.append("缺少完整权重文件（不能仅下载 Git LFS 指针）")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="下载默认模型，缓存和权重保存在项目内。失败后可重跑。")
    parser.add_argument("--check", action="store_true", help="仅离线检查必要文件，不下载、不加载模型")
    parser.add_argument("--only", choices=tuple(MODELS), help="只处理一种模型")
    parser.add_argument("--models-root", type=Path, default=ROOT / "models", help="模型根目录，默认为本项目 models")
    args = parser.parse_args()
    model_root = args.models_root.resolve()
    if not args.check:
        # Set before importing the Hub library. No global environment changes.
        os.environ["HF_HOME"] = str(ROOT / "runtime" / "huggingface")
        os.environ["HF_HUB_CACHE"] = str(ROOT / "runtime" / "huggingface" / "hub")
        os.environ["HF_XET_CACHE"] = str(ROOT / "runtime" / "huggingface" / "xet")
        from huggingface_hub import snapshot_download

    failed = False
    for kind, (repo, revision) in MODELS.items():
        if args.only and args.only != kind:
            continue
        target = model_root / kind / repo.split("/")[-1]
        print(f"{kind}: {repo}\n目录: {target}", flush=True)
        if not args.check:
            try:
                snapshot_download(
                    repo_id=repo, revision=revision, local_dir=str(target),
                    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE*"],
                )
            except Exception as exc:
                failed = True
                print(f"下载失败：{exc}\n检查网络后重跑同一命令；不会用空权重继续运行。", flush=True)
                continue
        errors = check_files(target, kind)
        if errors:
            failed = True
            print("\n".join(errors), flush=True)
        else:
            print("必要文件检查通过（不是推理质量测试）", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
