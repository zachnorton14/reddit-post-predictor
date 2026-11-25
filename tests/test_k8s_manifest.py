"""The production manifest keeps the settings that make a rollout zero-downtime, and the drill only removes the model."""

from pathlib import Path

import jinja2
import pytest
import yaml

ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "infra" / "k8s" / "serve.yaml.j2"


def render(**values) -> dict:
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    env.filters["bool"] = lambda v: v if isinstance(v, bool) else str(v).lower() in ("true", "yes", "1")  # Ansible's
    docs = list(yaml.safe_load_all(env.from_string(TEMPLATE.read_text()).render(**values)))
    return {d["kind"]: d for d in docs if d}


@pytest.fixture
def manifest():
    return render(serve_image="ghcr.io/o/r/serve:sha-abc", serve_version="model-3")


def container(m):
    return m["Deployment"]["spec"]["template"]["spec"]["containers"][0]


def test_rollout_never_drops_capacity(manifest):
    spec = manifest["Deployment"]["spec"]
    assert spec["strategy"]["rollingUpdate"]["maxUnavailable"] == 0
    assert spec["strategy"]["rollingUpdate"]["maxSurge"] >= 1
    assert spec["replicas"] >= 2
    assert spec["progressDeadlineSeconds"] > 0


def test_probes_and_ports_match_the_server(manifest):
    c = container(manifest)
    assert c["ports"][0]["containerPort"] == 8000  # docker/serve.Dockerfile, pipeline/serve.py
    for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
        assert c[probe]["httpGet"]["path"] == "/health"
    assert c["lifecycle"]["preStop"]["exec"]["command"][0] == "sleep"
    grace = manifest["Deployment"]["spec"]["template"]["spec"]["terminationGracePeriodSeconds"]
    assert grace > int(c["lifecycle"]["preStop"]["exec"]["command"][1])
    svc = manifest["Service"]["spec"]
    assert svc["type"] == "NodePort" and svc["ports"][0]["targetPort"] == "http"
    assert svc["selector"] == manifest["Deployment"]["spec"]["selector"]["matchLabels"]


def test_image_and_version(manifest):
    c = container(manifest)
    assert c["image"] == "ghcr.io/o/r/serve:sha-abc"
    assert c["imagePullPolicy"] == "IfNotPresent"  # the cluster holds no registry credential
    assert {"name": "MODEL_VERSION", "value": "model-3"} in c["env"]
    assert manifest["Deployment"]["metadata"]["annotations"]["kubernetes.io/change-cause"] == "model-3"


def test_drill_only_removes_the_model(manifest):
    broken = render(serve_image="ghcr.io/o/r/serve:sha-abc", serve_version="model-3", serve_break=True)
    extra = [e for e in container(broken)["env"] if e not in container(manifest)["env"]]
    assert extra == [{"name": "MODEL_DIR", "value": "/model-missing"}]
    assert "rollback drill" in broken["Deployment"]["metadata"]["annotations"]["kubernetes.io/change-cause"]
    assert render(serve_image="i", serve_version="v", serve_break="false") == render(serve_image="i", serve_version="v")


def test_the_drill_marker_matches_probe_undo_target():
    import importlib.util

    spec = importlib.util.spec_from_file_location("probe", ROOT / "scripts" / "probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    broken = render(serve_image="i", serve_version="v", serve_break=True)
    assert probe.BROKEN in container(broken)["env"]


def test_overrides():
    m = render(serve_image="i", serve_version="v", replicas=2, node_port=30123, progress_deadline=90)
    assert m["Deployment"]["spec"]["replicas"] == 2
    assert m["Service"]["spec"]["ports"][0]["nodePort"] == 30123
    assert m["Deployment"]["spec"]["progressDeadlineSeconds"] == 90


@pytest.mark.parametrize("path", sorted((ROOT / "infra" / "ansible").glob("*.yml"))
                         + sorted((ROOT / ".github" / "workflows").glob("*.yml")), ids=lambda p: p.name)
def test_playbooks_and_workflows_parse(path):
    assert yaml.safe_load(path.read_text(encoding="utf-8"))
