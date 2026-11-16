"""High-level provisioning flows built on VCLClient."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from .client import VCLClient, VCLError
from .config import Environment

log = logging.getLogger("vcl")


class ReadyTimeout(VCLError):
    """The reservation did not become ready in time (exit code 3)."""


def provision(client: VCLClient, env: Environment, length_override: int | None = None) -> dict:
    """Create a reservation for ``env``. Returns {request_id, image_id, image_name, length_minutes}."""
    length = length_override if length_override is not None else env.length_minutes
    if length <= 0:
        raise VCLError(f"reservation length must be positive, got {length}")
    image = client.find_image(env.image)
    log.info("requesting %s (image %s) for %d min", image["name"], image["id"], length)
    request_id = client.add_request(int(image["id"]), length)
    log.info("created reservation %d", request_id)
    return {
        "request_id": request_id,
        "image_id": int(image["id"]),
        "image_name": image["name"],
        "length_minutes": length,
    }


def wait_ready(
    client: VCLClient,
    request_id: int,
    timeout_seconds: float,
    poll_seconds: float = 30,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict:
    """Poll until the reservation is ready. Raises on failure or timeout."""
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    deadline = clock() + timeout_seconds
    while True:
        status = client.get_request_status(request_id)
        state = status.get("status")
        if state == "ready":
            log.info("reservation %d is ready", request_id)
            return status
        if state in ("failed", "timedout"):
            raise VCLError(f"reservation {request_id} {state}")
        if state == "loading" and "time" in status:
            log.info("reservation %d loading, ~%s min remaining", request_id, status["time"])
        else:
            log.info("reservation %d status: %s", request_id, state)
        if clock() >= deadline:
            raise ReadyTimeout(f"reservation {request_id} not ready after {timeout_seconds:.0f}s")
        sleep(poll_seconds)


def connect_info(client: VCLClient, request_id: int, remote_ip: str | None = None) -> dict:
    """Fetch connection details. ``remote_ip`` is the address VCL will allow to connect."""
    remote_ip = remote_ip or client.get_ip()
    data = client.get_request_connect_data(request_id, remote_ip)
    ip = data.get("serverIP") or data.get("serverip")
    if not ip:
        raise VCLError(f"reservation {request_id}: connect data has no server IP")
    port = data.get("connectport")
    if not port:
        for method in data.get("connectMethods") or []:
            ports = method.get("connectports") or []
            if ports:
                port = ports[0].get("port") if isinstance(ports[0], dict) else ports[0]
                break
    return {
        "request_id": int(request_id),
        "ip": ip,
        "user": data.get("user"),
        "port": int(port) if port else 22,
        "password": data.get("password"),
        "remote_ip": remote_ip,
    }


def write_inventory(
    info: dict,
    path: str | Path,
    group: str = "vcl",
    ssh_user: str | None = None,
    include_password: bool = False,
) -> Path:
    """Write a one-host Ansible INI inventory."""
    user = ssh_user or info.get("user")
    parts = [info["ip"], f"ansible_port={info['port']}"]
    if user:
        parts.append(f"ansible_user={user}")
    if include_password and info.get("password"):
        parts.append(f"ansible_password={info['password']}")
    path = Path(path)
    path.write_text(f"[{group}]\n{' '.join(parts)}\n", encoding="utf-8")
    return path


def teardown(client: VCLClient, request_id: int) -> bool:
    """End a reservation. Returns True if ended now, False if it was already gone."""
    try:
        client.end_request(request_id)
        log.info("ended reservation %d", request_id)
        return True
    except VCLError:
        active = {int(r["requestid"]) for r in client.get_request_ids()}
        if int(request_id) not in active:
            log.info("reservation %d already ended", request_id)
            return False
        raise
