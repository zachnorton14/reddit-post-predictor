"""Store candidates, fetch them by pinned revision, and promote champions in the Hugging Face model repo.

    python scripts/hf_models.py path <model.json>                 prints models/<name> (checks the name)
    python scripts/hf_models.py upload <dir> <path>               uploads a train-eval folder to <path>
    python scripts/hf_models.py candidate <candidate.yaml> [--resolve]
                                                                  checks the file and prints it as JSON; --resolve
                                                                  adds `revision`, the repo commit to score and ship
    python scripts/hf_models.py fetch <path> <revision> <dir> [--name N --type T]
                                                                  downloads <path> at <revision> into <dir> and makes
                                                                  sure <dir>/model.json exists (written from N and T
                                                                  for experiment runs, which have none)
    python scripts/hf_models.py promote <release.json> <version>  records the release next to the weights, tags that
                                                                  commit <version> and points champion.json at it
    python scripts/hf_models.py fetch-champion <dir>              downloads the champion to <dir>, if there is one

candidate.yaml (at the repo root) names the model that a `dev` -> `release` pull request ships: a folder in this
repo that already holds trained weights, from an experiment run or a train-eval run. Nothing is retrained to ship
it. Model names: <source .py name>-<param count>[-<detail>...], for example run_pooling-435M-deberta_large_attn4
(see .claude/skills/model-release/SKILL.md).

Needs a Hugging Face token with access to the repo in $HF_TOKEN (reads work without one: the repo is public).
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
FOLDER = re.compile(r"^(models|experiments)(/[A-Za-z0-9_.-]+)+$")
TYPE = re.compile(r"^[a-z0-9_]+$")
# Files promote adds next to the weights. Everything else in the folder must be the same as what the gate scored.
RECORD_FILES = {"model.json", "release.json"}


def model_path(model_json: dict) -> str:
    name = model_json.get("name", "")
    if not NAME.match(name):
        raise ValueError(f"model.json name {name!r} must look like <source>-<params>[-<detail>...], "
                         "for example dadjokes_models-22M-minilm_l6-hgb_blend")
    return f"models/{name}"


def load_candidate(path) -> dict:
    """Read and check candidate.yaml: exactly `path`, `name` and `type`. Raises ValueError naming the problem."""
    import yaml

    cand = yaml.safe_load(Path(path).read_text())
    if not isinstance(cand, dict) or set(cand) != {"path", "name", "type"}:
        raise ValueError(f"{path} must have exactly the keys path, name and type")
    folder = str(cand["path"]).strip("/")
    if not FOLDER.match(folder) or ".." in folder.split("/"):
        raise ValueError(f"{path}: path {cand['path']!r} must be a folder under models/ or experiments/")
    model_path({"name": cand["name"]})
    if not TYPE.match(str(cand["type"])):
        raise ValueError(f"{path}: type {cand['type']!r} must name a loader in pipeline/evaluate.py LOADERS")
    return {"path": folder, "name": cand["name"], "type": cand["type"]}


def resolve(cand: dict, api) -> str:
    """The repo commit the candidate is scored and shipped at (the latest). Fails if the folder is empty there."""
    revision = api.model_info(REPO).sha
    files = [f for f in api.list_repo_tree(REPO, path_in_repo=cand["path"], revision=revision, recursive=True)
             if hasattr(f, "blob_id")]
    if not files:
        raise ValueError(f"{REPO}/{cand['path']} has no files at {revision[:7]}")
    return revision


def fetch(path: str, revision: str, dest: Path, name=None, type_=None) -> dict:
    """Download the folder at `revision` into dest and return its model.json, writing one if it has none."""
    from huggingface_hub import snapshot_download

    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(REPO, revision=revision, allow_patterns=f"{path}/**", local_dir=tmp)
        if not (Path(tmp) / path).is_dir():
            raise ValueError(f"{REPO}/{path} not found at {revision[:7]}")
        shutil.copytree(Path(tmp) / path, dest, dirs_exist_ok=True)
    meta_file = Path(dest) / "model.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        wrong = {k: (meta.get(k), v) for k, v in (("name", name), ("type", type_)) if v and meta.get(k) != v}
        if wrong:
            raise ValueError(f"{path}/model.json does not match candidate.yaml: {wrong}")
        return meta
    if not (name and type_):
        raise ValueError(f"{path} has no model.json; give --name and --type")
    meta = {"type": type_, "name": name, "hf_path": path}
    meta_file.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def weights(api, path: str, revision: str) -> dict:
    """{file: git blob id} for the folder's files at `revision`, leaving out the records promote adds."""
    return {f.path: f.blob_id for f in api.list_repo_tree(REPO, path_in_repo=path, revision=revision, recursive=True)
            if hasattr(f, "blob_id") and Path(f.path).name not in RECORD_FILES}


def champion_record(release: dict, version: str, revision: str) -> dict:
    missing = [k for k in ("path", "name", "type", "commit") if not release.get(k)]
    if missing:
        raise ValueError(f"release.json lacks {', '.join(missing)}")
    return {"version": version, "name": release["name"], "type": release["type"], "path": release["path"],
            "commit": release["commit"], "revision": revision}


def promote(api, release: dict, version: str) -> dict:
    """Add release.json (and model.json if missing) next to the gated weights, tag that commit, update champion.json.

    The tag goes on the commit that adds the records, after checking that the weights there are the ones the gate
    scored at release["revision"]. Later uploads can't move the champion: it's read at its tag.
    """
    from huggingface_hub import CommitOperationAdd

    path, scored = release["path"], release["revision"]
    champion_record(release, version, scored)  # check the fields before writing anything
    ops = [CommitOperationAdd(f"{path}/release.json", json.dumps(release, indent=2).encode())]
    if not api.file_exists(REPO, f"{path}/model.json", revision=scored):
        meta = {"type": release["type"], "name": release["name"], "hf_path": path}
        ops.append(CommitOperationAdd(f"{path}/model.json", json.dumps(meta, indent=2).encode()))
    commit = api.create_commit(REPO, operations=ops, commit_message=f"Release {path} as {version}")
    if weights(api, path, commit.oid) != weights(api, path, scored):
        raise ValueError(f"{path} changed after the gate scored it at {scored[:7]}; not promoting. "
                         "Rerun the release check.")
    champ = champion_record(release, version, commit.oid)
    # Tag first, so champion.json never names a tag that doesn't exist.
    api.create_tag(REPO, tag=version, revision=commit.oid)
    api.upload_file(path_or_fileobj=json.dumps(champ, indent=2).encode(), path_in_repo="champion.json",
                    repo_id=REPO, commit_message=f"Promote {path} as {version}")
    return champ


def record_revision(folder: Path, revision: str) -> None:
    """Add the upload commit to folder/run.json (the local copy that becomes the GitHub artifact)."""
    run_json = folder / "run.json"
    if run_json.exists():
        run = json.loads(run_json.read_text())
        run["hf_revision"] = revision
        run_json.write_text(json.dumps(run, indent=2) + "\n")


def fetch_champion(dest: Path) -> dict | None:
    """Download the champion's folder (model.json and weights) into dest. None if there is none."""
    from huggingface_hub import HfApi, hf_hub_download

    if not HfApi().file_exists(REPO, "champion.json"):
        return None
    with open(hf_hub_download(REPO, "champion.json")) as f:
        champ = json.load(f)
    fetch(champ["path"], champ["revision"], Path(dest), champ["name"], champ["type"])
    return champ


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("path").add_argument("model_json")
    u = sub.add_parser("upload")
    u.add_argument("folder")
    u.add_argument("path")
    c = sub.add_parser("candidate")
    c.add_argument("file")
    c.add_argument("--resolve", action="store_true")
    f = sub.add_parser("fetch")
    f.add_argument("path")
    f.add_argument("revision")
    f.add_argument("dest")
    f.add_argument("--name")
    f.add_argument("--type")
    r = sub.add_parser("promote")
    r.add_argument("release_json")
    r.add_argument("version")
    sub.add_parser("fetch-champion").add_argument("dest")
    args = p.parse_args(argv)

    try:
        if args.cmd == "path":
            print(model_path(json.loads(Path(args.model_json).read_text())))
            return 0
        if args.cmd == "candidate":
            cand = load_candidate(args.file)
            if args.resolve:
                from huggingface_hub import HfApi

                cand["revision"] = resolve(cand, HfApi())
            print(json.dumps(cand))
            return 0
        if args.cmd == "promote":
            release = json.loads(Path(args.release_json).read_text())
            champion_record(release, args.version, release.get("revision") or "")  # fail before touching HF
            if not release.get("revision"):
                raise ValueError("release.json lacks revision (the repo commit the gate scored)")

        from huggingface_hub import HfApi

        api = HfApi()
        if args.cmd == "upload":
            # Replace the folder: files from an earlier run of the same model must not mix with this one.
            commit = api.upload_folder(repo_id=REPO, folder_path=args.folder, path_in_repo=args.path,
                                       delete_patterns="*", commit_message=f"Add candidate {args.path}")
            record_revision(Path(args.folder), commit.oid)
            print(f"uploaded {args.folder} to {REPO}/{args.path} at {commit.oid}")
        elif args.cmd == "fetch":
            meta = fetch(args.path, args.revision, Path(args.dest), args.name, args.type)
            print(f"{args.path} at {args.revision[:7]} ({meta['type']}) in {args.dest}")
        elif args.cmd == "promote":
            champ = promote(api, release, args.version)
            print(f"champion is now {args.version} ({champ['name']}, {champ['revision'][:7]})")
        else:
            champ = fetch_champion(Path(args.dest))
            print(f"champion {champ['version']} ({champ['name']}) in {args.dest}" if champ else "no champion yet")
    except ValueError as e:
        print(f"::error::{e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
