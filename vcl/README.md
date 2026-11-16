# vcl: headless NCSU VCL provisioning

This package is a Python client and CLI for the NCSU VCL XML-RPC API. It reserves VCL instances from scripts or CI, waits until they're ready, writes an Ansible inventory, and ends the reservations afterwards. It speaks the same API as [wufctl](https://github.com/ShounakDeshmukh/wufctl), but it's built for automation rather than interactive use.

It needs only the Python standard library plus `pyyaml`:

```
pip install -r vcl/requirements.txt
```

## Setup

1. Create an API token at **vcl.ncsu.edu → Manage → User Preferences → Manage Tokens**.
2. Export the token. In GitHub Actions, store it as the `VCL_TOKEN` secret.
   ```
   export VCL_TOKEN=...            # bash
   $env:VCL_TOKEN = "..."          # PowerShell
   ```
   You can also pass `--token-file path`. The token is never printed.
3. Check that the token works with `python -m vcl test`.
4. Find the real image names, then edit [environments.yaml](environments.yaml). The defaults are placeholders:
   ```
   python -m vcl images --search 2080
   ```

## Environments

Each environment is a named profile with an image and a reservation length:

```yaml
environments:
  gpu-train:
    image: "Ubuntu 22.04 RTX 2080 Ti"   # id, exact name, or unique substring
    length_minutes: 240                  # override per run with --length
    wait_timeout_minutes: 30
    ssh_user: null                       # optional inventory user override
```

The CLI looks for a profile file in this order:

1. `--env-file`
2. `$VCL_ENV_FILE`
3. `./vcl_environments.yaml`
4. the packaged `vcl/environments.yaml`

VCL caps the reservation length at your user group's maximum.

## Commands

| Command | What it does |
|---|---|
| `python -m vcl envs` | List the profiles. |
| `python -m vcl images [--search TEXT]` | List the images you can reserve. |
| `python -m vcl up ENV [--length MIN] [--inventory inv.ini] [--no-wait]` | Reserve an instance, wait until it's ready, and fetch the connection info. |
| `python -m vcl status [ID]` | Show a reservation's status. |
| `python -m vcl connect [ID] [--inventory inv.ini] [--show-password]` | Show the IP, user, and port. |
| `python -m vcl extend MINUTES [ID]` | Extend a reservation. |
| `python -m vcl down [ID]` | End a reservation. Running it on one that's already ended is not an error. |
| `python -m vcl list` | List your active reservations. |

When you leave out `ID`, the command reads it from `--state-file`, which defaults to `.vcl_state.json`. `up` writes that file.

Add `--json` before the subcommand to get machine-readable output.

### How `up` behaves

- If the instance fails or times out, `up` ends the reservation so GPU time isn't leaked. Pass `--keep-on-failure` to keep it for debugging.
- Inside GitHub Actions, `up` also writes `request_id`, `ip`, `user`, and `port` to `$GITHUB_OUTPUT`.
- VCL lets only the IP that requested connect data reach the instance, which defaults to this machine's IP. So run `up`/`connect` on the same machine that will run ssh/Ansible, or pass `--remote-ip`.
- The password is left out of the output and the state file. Use `--show-password`, or `--include-password` for the inventory, only when key-based SSH isn't available.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | VCL/API error |
| 2 | Configuration or usage error |
| 3 | Timed out waiting for the instance |

## Example pipeline usage

```bash
python -m vcl up gpu-train --length 180 --inventory vcl_inventory.ini
ansible-playbook -i vcl_inventory.ini train.yml
python -m vcl down          # run this in an always() / cleanup step
```

## Library use

```python
from vcl import VCLClient, get_environment, provision, wait_ready, connect_info, teardown, resolve_token

client = VCLClient(resolve_token())
req = provision(client, get_environment("gpu-train"), length_override=120)
wait_ready(client, req["request_id"], timeout_seconds=1800)
info = connect_info(client, req["request_id"])
...
teardown(client, req["request_id"])
```

## Tests

```
python -m unittest tests.test_vcl -v
```

The tests run against an in-process fake XML-RPC server and need no network.
