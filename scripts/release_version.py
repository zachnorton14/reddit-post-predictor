"""Version tags for deploys: `<prefix>-<n>`, numbered from the git tags that already exist.

    python scripts/release_version.py next rc <commit>    prints the version for a deploy of <commit>, e.g. rc-4
    python scripts/release_version.py latest rc           prints the highest existing rc-<n> (nothing if none)

deploy.yml uses `next rc` (staging). release-check.yml uses `latest rc` to find the staged model to gate against.
A production deploy will use prefix `model`. Numbers count up from the highest existing `<prefix>-<n>` tag. Tags
are made only when a deploy finishes (its GitHub release), so a failed deploy leaves no gap. If a `<prefix>-<n>` tag
already points at <commit>, `next` prints that version again, so a re-run of the same deploy reuses its number.
Needs the repo's tags (checkout fetch-depth: 0, or `git fetch --tags`).
"""

import re
import subprocess
import sys


def git_tags(*args) -> list[str]:
    out = subprocess.run(["git", "tag", "--list", *args], check=True, capture_output=True, text=True).stdout
    return out.split()


def numbers(prefix: str, tags: list[str]) -> list[int]:
    pattern = re.compile(rf"{re.escape(prefix)}-(\d+)")
    return [int(m.group(1)) for t in tags if (m := pattern.fullmatch(t))]


def latest(prefix: str, tags: list[str]) -> str | None:
    found = numbers(prefix, tags)
    return f"{prefix}-{max(found)}" if found else None


def pick(prefix: str, all_tags: list[str], tags_at_commit: list[str]) -> str:
    if here := latest(prefix, tags_at_commit):
        return here
    return f"{prefix}-{max(numbers(prefix, all_tags), default=0) + 1}"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 3 and argv[0] == "next":
        _, prefix, commit = argv
        print(pick(prefix, git_tags(f"{prefix}-*"), git_tags(f"{prefix}-*", "--points-at", commit)))
    elif len(argv) == 2 and argv[0] == "latest":
        found = latest(argv[1], git_tags(f"{argv[1]}-*"))
        if found:
            print(found)
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
