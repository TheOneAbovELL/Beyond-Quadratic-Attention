"""Single entry point for the whole Beyond Quadratic Attention lab.

Double-click ``start_lab.bat`` (Windows) or run ``python start.py``. It checks that the
required packages are importable, then shows a menu. Pressing Enter opens the research
interface in your browser.

Non-interactive use: ``python start.py <action>`` where action is one of
``ui | check | test | correctness | book | results | shell | env``.

Everything stays local: the interface binds to localhost, usage statistics are disabled,
and nothing is downloaded unless you explicitly agree to install a missing package.
"""
from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
REQUIRED = {"torch": "torch", "numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib",
            "psutil": "psutil", "yaml": "pyyaml"}
OPTIONAL = {"streamlit": "streamlit", "pytest": "pytest"}
BOOK = ROOT / "docs" / "book" / "Beyond_Quadratic_Attention_Book.pdf"
RESULTS = ROOT / "docs" / "results.md"


def say(msg: str = "") -> None:
    print(msg, flush=True)


def missing(modules) -> list:
    return [m for m in modules if importlib.util.find_spec(m) is None]


def ask(question: str, default: bool = False) -> bool:
    if not sys.stdin or not sys.stdin.isatty():
        return default
    answer = input(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not answer else answer.startswith("y")


def lab(*args) -> int:
    """Run `python lab.py <args>` (lab.py puts src/ on the path; no install needed)."""
    return subprocess.call([PY, str(ROOT / "lab.py"), *map(str, args)], cwd=ROOT)


def ensure_environment() -> bool:
    if sys.version_info < (3, 10):
        say(f"Python 3.10 or newer is required (found {sys.version.split()[0]}).")
        return False
    need = missing(REQUIRED)
    if need:
        pkgs = [REQUIRED[m] for m in need]
        say(f"Missing required packages: {', '.join(pkgs)}")
        if not ask("Download and install them from PyPI now?"):
            say("Cannot continue without them. Install them, then run start again.")
            return False
        if subprocess.call([PY, "-m", "pip", "install", *pkgs]) != 0:
            return False
    opt = missing(OPTIONAL)
    if opt:
        say(f"Note: optional packages not installed: {', '.join(OPTIONAL[m] for m in opt)} "
            "(the interface needs streamlit; the tests need pytest).")
        if ask("Install them from PyPI now?"):
            subprocess.call([PY, "-m", "pip", "install", *[OPTIONAL[m] for m in opt]])
    return True


def free_port(start: int = 8501) -> int:
    """First port from `start` that nothing is listening on (never stops other programs)."""
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def open_file(path: Path) -> int:
    if not path.exists():
        say(f"Not found: {path}")
        return 1
    say(f"Opening {path.relative_to(ROOT)} ...")
    if os.name == "nt":
        os.startfile(path)  # noqa: S606 - local file in this project
    else:
        subprocess.call(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
    return 0


# ------------------------------------------------------------------------------- actions
def action_ui() -> int:
    if missing(["streamlit"]):
        say("The research interface needs streamlit, which is not installed.")
        return 1
    port = free_port()
    say(f"\nStarting the lab at http://localhost:{port}  (it opens in your browser).")
    say("Keep this window open while you use it; press Ctrl+C here to stop.\n")
    try:
        return lab("ui", "--port", port)
    except KeyboardInterrupt:
        say("\nLab stopped.")
        return 0


def action_check() -> int:
    say("\nEnvironment:")
    rc = lab("env")
    say("\nSmoke benchmark (every attention method on tiny shapes, about a minute):")
    return rc or lab("bench", "--smoke")


def action_test() -> int:
    if missing(["pytest"]):
        say("pytest is not installed.")
        return 1
    say("\nRunning the essential test suite (about 3-5 minutes, depending on machine load) ...")
    return lab("test")


def action_correctness() -> int:
    say("\nNumerical correctness laboratory (every method against the float64 reference) ...")
    return lab("correctness")


def action_book() -> int:
    return open_file(BOOK)


def action_results() -> int:
    return open_file(RESULTS)


def action_shell() -> int:
    if os.name == "nt":
        os.system(f'start "Beyond Quadratic Attention" cmd /k "cd /d "{ROOT}" && '
                  f'echo Lab commands: python lab.py --help"')
    else:
        say(f"Run:  cd {ROOT}  &&  python lab.py --help")
    return 0


def action_env() -> int:
    return lab("env")


ACTIONS = {
    "ui": ("Open the lab (research interface in your browser)", action_ui),
    "check": ("Quick health check (environment + smoke benchmark, about a minute)", action_check),
    "test": ("Run the essential test suite (~3-5 minutes)", action_test),
    "correctness": ("Run the numerical correctness laboratory", action_correctness),
    "book": ("Open the project book (PDF)", action_book),
    "results": ("Open the results summary (docs/results.md)", action_results),
    "shell": ("Open a terminal in the project folder", action_shell),
    "env": ("Show environment information", action_env),
}
MENU = ["ui", "check", "test", "correctness", "book", "results", "shell"]


def menu() -> int:
    while True:
        say("\n=============  Beyond Quadratic Attention  =============")
        for i, key in enumerate(MENU, 1):
            say(f"  {i}  {ACTIONS[key][0]}" + ("   [Enter]" if i == 1 else ""))
        say("  0  Quit")
        choice = input("\nChoose: ").strip() or "1"
        if choice == "0":
            return 0
        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            ACTIONS[MENU[int(choice) - 1]][1]()
        else:
            say("Please type a number from the menu.")


def main(argv) -> int:
    os.chdir(ROOT)
    if argv and argv[0] in ("-h", "--help"):
        say(__doc__)
        return 0
    if argv and argv[0] not in ACTIONS:
        say(f"Unknown action {argv[0]!r}. Choose one of: {', '.join(ACTIONS)}")
        return 2
    if not ensure_environment():
        return 1
    if argv:
        return ACTIONS[argv[0]][1]()
    if not sys.stdin or not sys.stdin.isatty():
        return action_ui()
    return menu()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        say("\nBye.")
        sys.exit(0)
