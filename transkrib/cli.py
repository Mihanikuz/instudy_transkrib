"""CLI: python -m transkrib.cli <команда>.

T-1 — version; T-2 — models-check, models-import, models-download. Дальше добавят T-5 (run-file)
и T-6 (status, pause, resume, check, health).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from transkrib import __version__, models_store
from transkrib.config import ConfigError, load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m transkrib.cli", description="Управление сервером транскрибации")
    sub = parser.add_subparsers(dest="command", required=True, metavar="КОМАНДА")
    sub.add_parser("version", help="версия сервера")

    check = sub.add_parser("models-check", help="проверить модели на диске (без сети)")
    check.add_argument("--deep", action="store_true", help="пересчитать хэши файлов (долго)")

    imp = sub.add_parser("models-import", help="перенести модели из каталога models/ gigascribe")
    imp.add_argument("source", type=Path, help="каталог models/ gigascribe (в нём gigaam/ и pyannote/)")

    sub.add_parser("models-download", help="скачать модели из сети (нужен HF_TOKEN для pyannote)")
    return parser


def _models_dir() -> Path:
    return load_settings().models_dir


def cmd_models_check(deep: bool) -> int:
    models_dir = _models_dir()
    problems = models_store.check(models_dir, deep=deep)
    if problems:
        print(f"Модели в {models_dir} не готовы:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"Модели в {models_dir} на месте: GigaAM {models_store.GIGAAM_MODEL}, {models_store.PYANNOTE_REPO}")
    return 0


def cmd_models_import(source: Path) -> int:
    models_store.import_models(source, _models_dir())
    return cmd_models_check(deep=False)


def cmd_models_download() -> int:
    settings = load_settings()
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    models_store.download_models(settings.models_dir, token)
    return cmd_models_check(deep=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "version":
            print(f"instudy_transkrib {__version__}")
            return 0
        if args.command == "models-check":
            return cmd_models_check(args.deep)
        if args.command == "models-import":
            return cmd_models_import(args.source)
        if args.command == "models-download":
            return cmd_models_download()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    except (OSError, RuntimeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 2  # argparse не пропустит неизвестную команду; на случай новых подкоманд без обработчика


if __name__ == "__main__":
    sys.exit(main())
