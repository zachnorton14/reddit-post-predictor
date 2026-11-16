"""Store candidates and promote champions in the Hugging Face model repo.

    python scripts/hf_models.py path <model.json>            prints models/<name> (checks the name)
    python scripts/hf_models.py upload <dir> <path>          uploads a candidate folder to <path>, then writes the
                                                             upload's commit into <dir>/run.json as hf_revision
    python scripts/hf_models.py promote <run.json> <version> tags that commit <version> and points champion.json at it
    python scripts/hf_models.py fetch-champion <dir>         downloads the champion to <dir>, if there is one

Model names are set by hand in pipeline/train.py (see .claude/skills/model-release/SKILL.md):
<source .py name>-<param count>[-<detail>...], for example dadjokes_models-22M-minilm_l6-hgb_blend.

Needs a Hugging Face token with access to the repo in $HF_TOKEN. The layout is in the repo's README.
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = os.environ.get("MODEL_REPO", "zachnorton03/reddit-post-predictor-models")
NAME = re.compile(r"^[a-z0-9_]+-\d+(\.\d+)?[KMB](-[a-z0-9_.]+)*$")


def model_path(model_json: dict) -> str:
    name = model_json.get("name", "")
    if not NAME.match(name):
        raise ValueError(f"model.json name {name!r} must look like <source>-<params>[-<detail>...], "
                         "for example dadjokes_models-22M-minilm_l6-hgb_blend")
    return f"models/{name}"


def champion_record(run: dict, model_json: dict, version: str) -> dict:
    # hf_revision is the HF commit that uploaded this run. Later runs of the same model replace the folder, so the
    # tag must point at this commit, not at the repo's latest one.
    if not run.get("hf_revision"):
        raise ValueError("run.json has no hf_revision (the run's Hugging Face upload commit). "
                         "Runs from before this field existed can't be promoted; retrain on dev first.")
    return {"version": version, "name": model_json["name"], "path": run["hf_path"], "commit": run["commit"],
            "revision": run["hf_revision"]}


def record_revision(folder: Path, revision: str) -> None:
    """Add the upload commit to folder/run.json (the local copy that becomes the GitHub artifact)."""
    run_json = folder / "run.json"
    if run_json.exists():
        run = json.loads(run_json.read_text())
        run["hf_revision"] = revision
        run_json.write_text(json.dumps(run, indent=2) + "\n")


def fetch_champion(dest: Path) -> dict | None:
    """Download the champion's folder (model/, metrics.json, run.json) into dest. None if there is none."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    if not HfApi().file_exists(REPO, "champion.json"):
        return None
    with open(hf_hub_download(REPO, "champion.json")) as f:
        champ = json.load(f)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(REPO, revision=champ["version"], allow_patterns=f"{champ['path']}/*", local_dir=tmp)
        shutil.copytree(Path(tmp) / champ["path"], dest, dirs_exist_ok=True)
    return champ


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("path").add_argument("model_json")
    u = sub.add_parser("upload")
    u.add_argument("folder")
    u.add_argument("path")
    r = sub.add_parser("promote")
    r.add_argument("run_json")
    r.add_argument("version")
    sub.add_parser("fetch-champion").add_argument("dest")
    args = p.parse_args(argv)

    if args.cmd == "path":
        try:
            print(model_path(json.loads(Path(args.model_json).read_text())))
        except ValueError as e:
            print(f"::error::{e}", file=sys.stderr)
            return 1
        return 0

    if args.cmd == "promote":
        run_json = Path(args.run_json)
        model_json = json.loads((run_json.parent / "model" / "model.json").read_text())
        try:
            champ = champion_record(json.loads(run_json.read_text()), model_json, args.version)
        except ValueError as e:
            print(f"::error::{e}", file=sys.stderr)
            return 1

    from huggingface_hub import HfApi

    api = HfApi()
    if args.cmd == "upload":
        # Replace the folder: files from an earlier run of the same model must not mix with this one.
        commit = api.upload_folder(repo_id=REPO, folder_path=args.folder, path_in_repo=args.path, delete_patterns="*",
                                   commit_message=f"Add candidate {args.path}")
        record_revision(Path(args.folder), commit.oid)
        print(f"uploaded {args.folder} to {REPO}/{args.path} at {commit.oid}")
    elif args.cmd == "promote":
        # Tag first, so champion.json never names a tag that doesn't exist.
        api.create_tag(REPO, tag=args.version, revision=champ["revision"])
        api.upload_file(path_or_fileobj=json.dumps(champ, indent=2).encode(), path_in_repo="champion.json",
                        repo_id=REPO, commit_message=f"Promote {champ['path']} as {args.version}")
        print(f"champion is now {args.version} ({champ['name']}, {champ['revision'][:7]})")
    else:
        champ = fetch_champion(Path(args.dest))
        print(f"champion {champ['version']} ({champ['name']}) in {args.dest}" if champ else "no champion yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
