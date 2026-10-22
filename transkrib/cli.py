"""CLI: python -m transkrib.cli <команда>.

T-1 — только version. Остальные команды добавляют T-2 (models-*), T-5 (run-file),
T-6 (status, pause, resume, check, health).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from transkrib import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m transkrib.cli", description="Управление сервером транскрибации")
    sub = parser.add_subparsers(dest="command", required=True, metavar="КОМАНДА")
    sub.add_parser("version", help="версия сервера")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        print(f"instudy_transkrib {__version__}")
        return 0
    return 2  # argparse не пропустит неизвестную команду; на случай новых подкоманд без обработчика


if __name__ == "__main__":
    sys.exit(main())
