"""Точка входа демона: python -m transkrib.

Пока заглушка: печатает версию и выходит. Главный цикл появится в T-6.
"""

from __future__ import annotations

import sys

from transkrib import __version__


def main() -> int:
    print(f"instudy_transkrib {__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
