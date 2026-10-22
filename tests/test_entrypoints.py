from __future__ import annotations

import subprocess
import sys

from transkrib import __version__, cli


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=60, check=False)


def test_daemon_entrypoint_prints_version() -> None:
    res = run("-m", "transkrib")
    assert res.returncode == 0, res.stderr
    assert __version__ in res.stdout


def test_cli_version() -> None:
    res = run("-m", "transkrib.cli", "version")
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == f"instudy_transkrib {__version__}"


def test_cli_without_command_fails() -> None:
    assert run("-m", "transkrib.cli").returncode == 2


def test_cli_main_in_process(capsys) -> None:
    assert cli.main(["version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_package_import_is_light() -> None:
    # Каркас не должен тянуть GPU-стек: torch, gigaam и pyannote импортируются только в load() бэкендов (T-2).
    code = (
        "import sys, transkrib, transkrib.config, transkrib.log, transkrib.cli;"
        "heavy = [m for m in ('torch', 'gigaam', 'pyannote') if m in sys.modules];"
        "print(heavy); sys.exit(1 if heavy else 0)"
    )
    res = run("-c", code)
    assert res.returncode == 0, res.stdout + res.stderr
