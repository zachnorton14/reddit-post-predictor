"""Command-line interface: ``python -m vcl <command>``.

Exit codes: 0 ok, 1 VCL/API error, 2 config/usage error, 3 timed out waiting for ready.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .client import DEFAULT_ENDPOINT, VCLClient, VCLError
from .config import ConfigError, get_environment, load_environments, resolve_token
from .provision import ReadyTimeout, connect_info, provision, teardown, wait_ready, write_inventory

DEFAULT_STATE_FILE = ".vcl_state.json"
log = logging.getLogger("vcl")


# -- output helpers -----------------------------------------------------------


def _emit(args, data) -> None:
    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return
    if isinstance(data, list):
        for row in data:
            print("  ".join(f"{k}={v}" for k, v in row.items()))
    elif isinstance(data, dict):
        for k, v in data.items():
            print(f"{k}: {v}")
    else:
        print(data)


def _github_output(values: dict) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for k, v in values.items():
            fh.write(f"{k}={v}\n")


def _read_state(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"no request id given and state file {path} not found") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"state file {path} is not valid JSON") from exc


def _request_id(args) -> int:
    if getattr(args, "request_id", None) is not None:
        return args.request_id
    return int(_read_state(args.state_file)["request_id"])


def _client(args) -> VCLClient:
    return VCLClient(resolve_token(args.token_file), endpoint=args.endpoint, timeout=args.timeout)


def _public(info: dict, show_password: bool) -> dict:
    return info if show_password else {k: v for k, v in info.items() if k != "password"}


# -- commands -------------------------------------------------------------------


def cmd_test(args) -> int:
    client = _client(args)
    result = client.test("vclctl")
    _emit(args, {"status": "ok", "endpoint": client.endpoint, "response": result.get("message", result)})
    return 0


def cmd_envs(args) -> int:
    envs = load_environments(args.env_file)
    _emit(args, [e.to_dict() for e in envs.values()])
    return 0


def cmd_images(args) -> int:
    images = _client(args).get_images()
    if args.search:
        needle = args.search.lower()
        images = [i for i in images if needle in str(i.get("name", "")).lower()]
    _emit(args, [{"id": i.get("id"), "name": i.get("name")} for i in images])
    return 0


def cmd_list(args) -> int:
    _emit(args, _client(args).get_request_ids())
    return 0


def cmd_up(args) -> int:
    env = get_environment(args.env, args.env_file)
    client = _client(args)
    created = provision(client, env, args.length)
    request_id = created["request_id"]
    state = {"env": env.name, **created}
    Path(args.state_file).write_text(json.dumps(state, indent=2), encoding="utf-8")

    if args.no_wait:
        _github_output({"request_id": request_id})
        _emit(args, state)
        return 0

    try:
        timeout = (args.wait_timeout or env.wait_timeout_minutes) * 60
        wait_ready(client, request_id, timeout, poll_seconds=args.poll)
        info = connect_info(client, request_id, args.remote_ip)
    except VCLError:
        if args.keep_on_failure:
            log.error("leaving reservation %d in place (--keep-on-failure)", request_id)
        else:
            log.error("provisioning failed; ending reservation %d", request_id)
            try:
                teardown(client, request_id)
                Path(args.state_file).unlink(missing_ok=True)
            except VCLError as cleanup_exc:
                log.error("cleanup failed, end it manually: %s", cleanup_exc)
        raise

    state.update(_public(info, show_password=False))
    if env.ssh_user:
        state["user"] = env.ssh_user
    Path(args.state_file).write_text(json.dumps(state, indent=2), encoding="utf-8")
    if args.inventory:
        write_inventory(info, args.inventory, args.group, env.ssh_user, args.include_password)
        state["inventory"] = str(args.inventory)
    _github_output({k: state[k] for k in ("request_id", "ip", "user", "port") if state.get(k) is not None})
    _emit(args, state)
    return 0


def cmd_status(args) -> int:
    request_id = _request_id(args)
    status = _client(args).get_request_status(request_id)
    _emit(args, {"request_id": request_id, **status})
    return 0


def cmd_connect(args) -> int:
    request_id = _request_id(args)
    info = connect_info(_client(args), request_id, args.remote_ip)
    if args.inventory:
        write_inventory(info, args.inventory, args.group, include_password=args.include_password)
    _emit(args, _public(info, args.show_password))
    return 0


def cmd_extend(args) -> int:
    request_id = _request_id(args)
    _client(args).extend_request(request_id, args.minutes)
    _emit(args, {"request_id": request_id, "extended_minutes": args.minutes})
    return 0


def cmd_down(args) -> int:
    request_id = _request_id(args)
    ended = teardown(_client(args), request_id)
    if args.request_id is None:
        Path(args.state_file).unlink(missing_ok=True)
    _emit(args, {"request_id": request_id, "status": "ended" if ended else "already-ended"})
    return 0


# -- parser ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m vcl", description="Provision NCSU VCL reservations headlessly.")
    p.add_argument("--endpoint", default=os.environ.get("VCL_ENDPOINT", DEFAULT_ENDPOINT))
    p.add_argument("--token-file", help="file containing the VCL API token (else $VCL_TOKEN)")
    p.add_argument("--env-file", help="environment profile YAML (else $VCL_ENV_FILE / defaults)")
    p.add_argument("--timeout", type=float, default=30, help="per-call HTTP timeout, seconds")
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def with_id(sp):
        sp.add_argument("request_id", nargs="?", type=int, help="reservation id (default: from state file)")
        sp.add_argument("--state-file", default=DEFAULT_STATE_FILE)
        return sp

    def with_inventory(sp):
        sp.add_argument("--inventory", help="write an Ansible INI inventory to this path")
        sp.add_argument("--group", default="vcl", help="inventory group name")
        sp.add_argument("--include-password", action="store_true", help="put ansible_password in the inventory")
        sp.add_argument("--remote-ip", help="IP allowed to connect (default: this machine's IP per VCL)")
        return sp

    sub.add_parser("test", help="verify token and connectivity").set_defaults(func=cmd_test)
    sub.add_parser("envs", help="list environment profiles").set_defaults(func=cmd_envs)
    sp = sub.add_parser("images", help="list reservable images")
    sp.add_argument("--search", help="case-insensitive name filter")
    sp.set_defaults(func=cmd_images)
    sub.add_parser("list", help="list your current reservations").set_defaults(func=cmd_list)

    sp = with_inventory(sub.add_parser("up", help="reserve an environment and wait until ready"))
    sp.add_argument("env", help="environment profile name")
    sp.add_argument("--length", type=int, help="reservation length in minutes (overrides profile)")
    sp.add_argument("--no-wait", action="store_true", help="return right after the request is created")
    sp.add_argument("--wait-timeout", type=int, help="minutes to wait for ready (overrides profile)")
    sp.add_argument("--poll", type=float, default=30, help="status poll interval, seconds")
    sp.add_argument("--keep-on-failure", action="store_true", help="don't end the reservation if it fails")
    sp.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    sp.set_defaults(func=cmd_up)

    with_id(sub.add_parser("status", help="show reservation status")).set_defaults(func=cmd_status)
    sp = with_inventory(with_id(sub.add_parser("connect", help="show connection info")))
    sp.add_argument("--show-password", action="store_true")
    sp.set_defaults(func=cmd_connect)
    sp = sub.add_parser("extend", help="extend a reservation")
    sp.add_argument("minutes", type=int)
    with_id(sp).set_defaults(func=cmd_extend)
    with_id(sub.add_parser("down", help="end a reservation")).set_defaults(func=cmd_down)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="vcl: %(message)s",
        stream=sys.stderr,
    )
    try:
        return args.func(args)
    except ReadyTimeout as exc:
        log.error("%s", exc)
        return 3
    except VCLError as exc:
        log.error("%s", exc)
        return 1
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
