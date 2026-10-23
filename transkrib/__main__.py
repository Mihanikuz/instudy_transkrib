"""Точка входа демона: python -m transkrib (цикл — transkrib/daemon.py). --version — только версия."""

from __future__ import annotations

import sys

from transkrib import __version__


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args in (["--version"], ["version"]):
        print(f"instudy_transkrib {__version__}")
        return 0
    if args:
        print("Использование: python -m transkrib [--version]", file=sys.stderr)
        return 2
    from transkrib.daemon import main as daemon_main

    return daemon_main()


if __name__ == "__main__":
    sys.exit(main())
