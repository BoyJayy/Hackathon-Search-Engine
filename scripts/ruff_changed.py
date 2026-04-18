from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_BRANCH = "main"
GIT_BIN = shutil.which("git") or "git"


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [GIT_BIN, *args],
        check=check,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def resolve_merge_base(base_branch: str) -> str:
    branch_check = run_git("rev-parse", "--verify", base_branch, check=False)
    if branch_check.returncode != 0:
        return "HEAD"

    merge_base = run_git("merge-base", "HEAD", base_branch)
    return merge_base.stdout.strip() or "HEAD"


def collect_changed_python_files(base_branch: str) -> list[str]:
    merge_base = resolve_merge_base(base_branch)
    tracked = run_git(
        "diff",
        "--name-only",
        "--diff-filter=ACMR",
        f"{merge_base}...HEAD",
    )
    untracked = run_git("ls-files", "--others", "--exclude-standard")

    candidates = {
        line.strip()
        for output in (tracked.stdout, untracked.stdout)
        for line in output.splitlines()
        if line.strip().endswith(".py")
    }

    return sorted(path for path in candidates if (REPO_ROOT / path).is_file())


def main() -> int:
    base_branch = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_BRANCH
    files = collect_changed_python_files(base_branch)

    if not files:
        sys.stdout.write("No changed Python files to lint.\n")
        return 0

    command = [str(REPO_ROOT / ".venv" / "bin" / "ruff"), "check", *files]
    sys.stdout.write(f"Running: {shlex.join(command)}\n")
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)  # noqa: S603
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
