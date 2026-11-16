"""VCL client/CLI tests against an in-process fake XML-RPC server; no network."""

import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

from vcl import cli
from vcl.client import VCLClient, VCLError
from vcl.config import ConfigError, Environment, load_environments
from vcl.provision import ReadyTimeout, connect_info, provision, teardown, wait_ready, write_inventory

IMAGES = [
    {"id": 10, "name": "Ubuntu 22.04 LTS"},
    {"id": 20, "name": "Ubuntu 22.04 RTX 2080 Ti GPU"},
    {"id": 21, "name": "Ubuntu 20.04 RTX 2080 Ti GPU"},
]


class FakeVCL:
    """State behind the fake server. ``statuses`` is the sequence getRequestStatus returns."""

    def __init__(self):
        self.next_id = 100
        self.requests = {}
        self.statuses = ["loading", "ready"]
        self.calls = []

    def _status(self, rid):
        if rid not in self.requests:
            return {"status": "error", "errorcode": 1, "errormsg": "unknown requestid"}
        return None

    def register(self, server):
        def add_request(imageid, start, length, foruser, nousercheck):
            self.calls.append(("add", imageid, start, length))
            rid = self.next_id
            self.next_id += 1
            self.requests[rid] = {"imageid": imageid, "length": length}
            return {"status": "success", "requestid": rid}

        def get_status(rid):
            err = self._status(rid)
            if err:
                return err
            state = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return {"status": state, "time": 3} if state == "loading" else {"status": state}

        def connect_data(rid, remote_ip):
            err = self._status(rid)
            if err:
                return err
            self.calls.append(("connect", rid, remote_ip))
            return {"status": "ready", "serverIP": "152.7.1.2", "user": "cgsizem3",
                    "password": "s3cret", "connectport": 22}

        def end(rid):
            err = self._status(rid)
            if err:
                return err
            del self.requests[rid]
            self.calls.append(("end", rid))
            return {"status": "success"}

        def extend(rid, minutes):
            err = self._status(rid)
            if err:
                return err
            self.requests[rid]["length"] += minutes
            return {"status": "success"}

        fns = {
            "XMLRPCtest": lambda msg: {"status": "success", "message": msg},
            "XMLRPCgetIP": lambda: {"status": "success", "ip": "10.0.0.5"},
            "XMLRPCgetImages": lambda: IMAGES,
            "XMLRPCaddRequest": add_request,
            "XMLRPCgetRequestIds": lambda: {
                "status": "success",
                "requests": [{"requestid": r, "imageid": v["imageid"]} for r, v in self.requests.items()],
            },
            "XMLRPCgetRequestStatus": get_status,
            "XMLRPCgetRequestConnectData": connect_data,
            "XMLRPCextendRequest": extend,
            "XMLRPCendRequest": end,
        }
        for name, fn in fns.items():
            server.register_function(fn, name)


class _Handler(SimpleXMLRPCRequestHandler):
    rpc_paths = ()  # accept any path, incl. ?mode=xmlrpccall

    def do_POST(self):
        self.server.seen_headers.append(dict(self.headers))
        super().do_POST()

    def log_message(self, *args):
        pass


class VCLTestCase(unittest.TestCase):
    def setUp(self):
        self.server = SimpleXMLRPCServer(("127.0.0.1", 0), requestHandler=_Handler,
                                         allow_none=True, logRequests=False)
        self.server.seen_headers = []
        self.fake = FakeVCL()
        self.fake.register(self.server)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        port = self.server.server_address[1]
        self.endpoint = f"http://127.0.0.1:{port}/scheduling/index.php?mode=xmlrpccall"
        self.client = VCLClient("tok123", endpoint=self.endpoint, timeout=5)
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def run_cli(self, *argv, env=None):
        out = io.StringIO()
        environ = {"VCL_TOKEN": "tok123", **(env or {})}
        with mock.patch.dict(os.environ, environ), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["--endpoint", self.endpoint, "--json", *argv])
        return code, out.getvalue()


class ClientTest(VCLTestCase):
    def test_sends_auth_headers(self):
        self.client.test("hi")
        headers = self.server.seen_headers[-1]
        self.assertEqual(headers["X-Authorization"], "Bearer tok123")
        self.assertEqual(headers["X-APIVERSION"], "2")

    def test_error_struct_raises(self):
        with self.assertRaises(VCLError) as ctx:
            self.client.get_request_status(999)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("unknown requestid", str(ctx.exception))

    def test_missing_token(self):
        with self.assertRaises(VCLError):
            VCLClient("", endpoint=self.endpoint)

    def test_find_image(self):
        self.assertEqual(self.client.find_image(10)["id"], 10)
        self.assertEqual(self.client.find_image("ubuntu 22.04 lts")["id"], 10)
        self.assertEqual(self.client.find_image("22.04 RTX")["id"], 20)
        with self.assertRaisesRegex(VCLError, "ambiguous"):
            self.client.find_image("2080")
        with self.assertRaisesRegex(VCLError, "no VCL image"):
            self.client.find_image("windows")


class ProvisionTest(VCLTestCase):
    env = Environment(name="gpu", image="22.04 RTX", length_minutes=240)

    def test_provision_uses_profile_length_and_override(self):
        provision(self.client, self.env)
        provision(self.client, self.env, length_override=45)
        adds = [c for c in self.fake.calls if c[0] == "add"]
        self.assertEqual(adds, [("add", 20, "now", 240), ("add", 20, "now", 45)])

    def test_wait_ready_loading_then_ready(self):
        rid = provision(self.client, self.env)["request_id"]
        sleeps = []
        status = wait_ready(self.client, rid, timeout_seconds=60, poll_seconds=5, sleep=sleeps.append)
        self.assertEqual(status["status"], "ready")
        self.assertEqual(sleeps, [5])

    def test_wait_ready_failed(self):
        rid = provision(self.client, self.env)["request_id"]
        self.fake.statuses = ["failed"]
        with self.assertRaisesRegex(VCLError, "failed"):
            wait_ready(self.client, rid, 60, sleep=lambda s: None)

    def test_wait_ready_timeout(self):
        rid = provision(self.client, self.env)["request_id"]
        self.fake.statuses = ["loading"]
        clock = iter([0, 10, 100]).__next__
        with self.assertRaises(ReadyTimeout):
            wait_ready(self.client, rid, 60, sleep=lambda s: None, clock=clock)

    def test_connect_info_and_inventory(self):
        rid = provision(self.client, self.env)["request_id"]
        info = connect_info(self.client, rid)
        self.assertEqual((info["ip"], info["user"], info["port"], info["remote_ip"]),
                         ("152.7.1.2", "cgsizem3", 22, "10.0.0.5"))
        inv = write_inventory(info, self.dir / "inv.ini")
        self.assertEqual(inv.read_text(), "[vcl]\n152.7.1.2 ansible_port=22 ansible_user=cgsizem3\n")
        write_inventory(info, inv, group="train", include_password=True)
        self.assertIn("ansible_password=s3cret", inv.read_text())
        self.assertTrue(inv.read_text().startswith("[train]"))

    def test_teardown_idempotent(self):
        rid = provision(self.client, self.env)["request_id"]
        self.assertTrue(teardown(self.client, rid))
        self.assertFalse(teardown(self.client, rid))


class ConfigTest(unittest.TestCase):
    def write(self, text):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        tmp.write(text)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_packaged_defaults_load(self):
        with mock.patch.dict(os.environ, {"VCL_ENV_FILE": ""}):
            envs = load_environments(Path(cli.__file__).with_name("environments.yaml"))
        self.assertIn("gpu-train", envs)

    def test_valid(self):
        envs = load_environments(self.write("environments:\n  a: {image: 5, length_minutes: 30}\n"))
        self.assertEqual(envs["a"], Environment(name="a", image="5", length_minutes=30))

    def test_invalid(self):
        bad = [
            "environments: {}\n",
            "environments:\n  a: {length_minutes: 30}\n",
            "environments:\n  a: {image: x}\n",
            "environments:\n  a: {image: x, length_minutes: -5}\n",
            "environments:\n  a: {image: x, length_minutes: '60'}\n",
            "environments:\n  a: {image: x, length_minutes: 60, lenght: 1}\n",
        ]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ConfigError):
                load_environments(self.write(text))


class CLITest(VCLTestCase):
    def setUp(self):
        super().setUp()
        self.env_file = self.dir / "envs.yaml"
        self.env_file.write_text(
            "environments:\n"
            "  gpu-train: {image: '22.04 RTX', length_minutes: 240, wait_timeout_minutes: 1}\n"
            "  bad-image: {image: '2080', length_minutes: 60}\n",
            encoding="utf-8",
        )
        self.state = self.dir / "state.json"
        self.inv = self.dir / "inv.ini"
        self.gh_out = self.dir / "gh_out"
        self.common = ["--env-file", str(self.env_file)]

    def up(self, *extra):
        return self.run_cli(*self.common, "up", "gpu-train", "--poll", "0", "--state-file", str(self.state),
                            "--inventory", str(self.inv), *extra, env={"GITHUB_OUTPUT": str(self.gh_out)})

    def test_up_status_extend_down(self):
        code, out = self.up("--length", "90")
        self.assertEqual(code, 0)
        data = json.loads(out)
        rid = data["request_id"]
        self.assertEqual((data["length_minutes"], data["ip"]), (90, "152.7.1.2"))
        self.assertNotIn("password", data)
        self.assertNotIn("s3cret", self.state.read_text())
        self.assertIn("152.7.1.2", self.inv.read_text())
        self.assertIn(f"request_id={rid}\n", self.gh_out.read_text())
        self.assertIn("ip=152.7.1.2\n", self.gh_out.read_text())

        code, out = self.run_cli("status", "--state-file", str(self.state))
        self.assertEqual((code, json.loads(out)["status"]), (0, "ready"))
        code, _ = self.run_cli("extend", "30", "--state-file", str(self.state))
        self.assertEqual(code, 0)
        self.assertEqual(self.fake.requests[rid]["length"], 120)

        code, out = self.run_cli("down", "--state-file", str(self.state))
        self.assertEqual((code, json.loads(out)["status"]), (0, "ended"))
        self.assertFalse(self.state.exists())
        code, out = self.run_cli("down", str(rid))
        self.assertEqual((code, json.loads(out)["status"]), (0, "already-ended"))

    def test_up_failure_ends_reservation(self):
        self.fake.statuses = ["failed"]
        code, _ = self.up()
        self.assertEqual(code, 1)
        self.assertEqual(self.fake.requests, {})
        self.assertFalse(self.state.exists())

    def test_up_failure_keep(self):
        self.fake.statuses = ["failed"]
        code, _ = self.up("--keep-on-failure")
        self.assertEqual(code, 1)
        self.assertEqual(len(self.fake.requests), 1)

    def test_up_timeout_exit_code(self):
        self.fake.statuses = ["loading"]
        with mock.patch("vcl.provision.time.monotonic", side_effect=[0, 0, 1000]):
            code, _ = self.up()
        self.assertEqual(code, 3)
        self.assertEqual(self.fake.requests, {})

    def test_errors_exit_codes(self):
        self.assertEqual(self.run_cli(*self.common, "up", "nope")[0], 2)
        self.assertEqual(self.run_cli(*self.common, "up", "bad-image", "--state-file", str(self.state))[0], 1)
        with mock.patch.dict(os.environ, {"VCL_TOKEN": ""}), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["--endpoint", self.endpoint, "test"]), 2)

    def test_connect_hides_password(self):
        rid = self.client.add_request(20, 60)
        _, out = self.run_cli("connect", str(rid))
        self.assertNotIn("password", json.loads(out))
        _, out = self.run_cli("connect", str(rid), "--show-password")
        self.assertEqual(json.loads(out)["password"], "s3cret")

    def test_images_search(self):
        _, out = self.run_cli("images", "--search", "2080")
        self.assertEqual([i["id"] for i in json.loads(out)], [20, 21])


if __name__ == "__main__":
    unittest.main()
