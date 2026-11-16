"""Thin XML-RPC client for the NCSU VCL scheduling API.

Mirrors the calls used by wufctl's vcl_lib: token auth via an
``X-Authorization: Bearer`` header plus ``X-APIVERSION: 2``.
"""

from __future__ import annotations

import xmlrpc.client
from typing import Any
from urllib.parse import urlparse

DEFAULT_ENDPOINT = "https://vcl.ncsu.edu/scheduling/index.php?mode=xmlrpccall"
API_VERSION = "2"


class VCLError(Exception):
    """An error returned by the VCL API or raised while talking to it."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        base = super().__str__()
        return f"[{self.code}] {base}" if self.code is not None else base


def _timeout_transport(base: type[xmlrpc.client.Transport]):
    class _Transport(base):
        def __init__(self, timeout: float, headers):
            super().__init__(headers=headers)
            self._timeout = timeout

        def make_connection(self, host):
            conn = super().make_connection(host)
            conn.timeout = self._timeout
            return conn

    return _Transport


_HTTPTransport = _timeout_transport(xmlrpc.client.Transport)
_HTTPSTransport = _timeout_transport(xmlrpc.client.SafeTransport)


class VCLClient:
    """One method per VCL XML-RPC call. All methods raise ``VCLError`` on failure."""

    def __init__(self, token: str, endpoint: str = DEFAULT_ENDPOINT, timeout: float = 30):
        if not token:
            raise VCLError("missing VCL API token")
        headers = [
            ("X-Authorization", f"Bearer {token}"),
            ("X-APIVERSION", API_VERSION),
        ]
        transport_cls = _HTTPSTransport if urlparse(endpoint).scheme == "https" else _HTTPTransport
        self.endpoint = endpoint
        self._proxy = xmlrpc.client.ServerProxy(
            endpoint,
            transport=transport_cls(timeout, headers),
            allow_none=True,
        )

    # -- plumbing ---------------------------------------------------------

    def _call(self, method: str, *args: Any) -> Any:
        try:
            result = getattr(self._proxy, method)(*args)
        except xmlrpc.client.Fault as exc:
            raise VCLError(f"{method}: {exc.faultString}", exc.faultCode) from exc
        except xmlrpc.client.ProtocolError as exc:
            raise VCLError(f"{method}: HTTP {exc.errcode} {exc.errmsg}") from exc
        except OSError as exc:
            raise VCLError(f"{method}: cannot reach {self.endpoint}: {exc}") from exc
        if isinstance(result, dict) and result.get("status") == "error":
            raise VCLError(
                f"{method}: {result.get('errormsg', 'unknown error')}",
                result.get("errorcode"),
            )
        return result

    # -- API calls ----------------------------------------------------------

    def test(self, message: str = "ping") -> dict:
        return self._call("XMLRPCtest", message)

    def get_ip(self) -> str:
        return self._call("XMLRPCgetIP")["ip"]

    def get_images(self) -> list[dict]:
        return list(self._call("XMLRPCgetImages"))

    def add_request(self, image_id: int, length_minutes: int, start: str | int = "now") -> int:
        result = self._call("XMLRPCaddRequest", int(image_id), start, int(length_minutes), "", 0)
        return int(result["requestid"])

    def get_request_ids(self) -> list[dict]:
        return list(self._call("XMLRPCgetRequestIds").get("requests", []))

    def get_request_status(self, request_id: int) -> dict:
        return self._call("XMLRPCgetRequestStatus", int(request_id))

    def get_request_connect_data(self, request_id: int, remote_ip: str) -> dict:
        return self._call("XMLRPCgetRequestConnectData", int(request_id), remote_ip)

    def extend_request(self, request_id: int, minutes: int) -> dict:
        return self._call("XMLRPCextendRequest", int(request_id), int(minutes))

    def end_request(self, request_id: int) -> dict:
        return self._call("XMLRPCendRequest", int(request_id))

    # -- helpers ------------------------------------------------------------

    def find_image(self, image: str | int) -> dict:
        """Resolve an image by id, exact name, or unique case-insensitive substring."""
        images = self.get_images()
        key = str(image).strip()
        for img in images:
            if str(img.get("id")) == key:
                return img
        for img in images:
            if str(img.get("name", "")).lower() == key.lower():
                return img
        matches = [img for img in images if key.lower() in str(img.get("name", "")).lower()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise VCLError(f"no VCL image matches {image!r} (try `python -m vcl images`)")
        names = ", ".join(f"{m['name']} (id {m['id']})" for m in matches[:10])
        raise VCLError(f"image {image!r} is ambiguous; candidates: {names}")
