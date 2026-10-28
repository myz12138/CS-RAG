"""Check paths without importing/executing the algorithm or invoking models."""

import argparse
from pathlib import Path
import sys

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from backend.app.algorithm_adapter.integration import STAGE_SCRIPTS, missing_entrypoints
from backend.app.config import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 CS-RAG 算法入口，不执行算法或模型")
    parser.add_argument("--env-file", type=Path, help="与 uvicorn 一致的配置文件")
    args = parser.parse_args()
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("指定的配置文件不存在")
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    root = Settings.from_env().reference_root
    print(f"系统目录：{SYSTEM_ROOT}")
    print(f"算法目录：{root}")
    missing = missing_entrypoints(root)
    for stage, path in STAGE_SCRIPTS.items():
        print(f"{'缺失' if path in missing else '存在'}：{stage} -> {path}")
    print("仅检查文件入口；不代表依赖、模型、输入输出协议或真实问答已通过验收。")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
