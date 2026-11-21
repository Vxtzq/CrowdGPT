#!/usr/bin/env python3
"""Push the current client to the ROCm box and pre-flight it against a coordinator.

Why this exists: `rrun.sh` hands the script over as a single ssh argv element, so
a base64-inlined client (633 KB) dies with "Argument list too long" — the shell
ceiling is ~128 KB per argument. Bulk moves go over scp; only a short driver goes
through rrun.

    tools/rocm_client_push.py                          # sync + doctor, no server
    tools/rocm_client_push.py --url https://... --token T   # sync + doctor + dry-run
    tools/rocm_client_push.py --url https://... --register myname --email me@x.io

`--dry-run` exercises auth and `GET /fl/task?skip_weights=true` and returns before
any training or upload, so exit 0 there means a real round will get that far.
"""

import argparse
import os
import pathlib
import shlex
import subprocess
import sys
import tarfile

REPO = pathlib.Path(__file__).resolve().parents[1]
ENV_FILE = pathlib.Path(os.environ.get("CROWDGPT_ENV_FILE", "/app/crowdgpt/.env"))
TARBALL = pathlib.Path("/tmp/opencode/rocm/client.tgz")
RRUN = pathlib.Path(os.environ.get("CROWDGPT_RRUN", "/tmp/opencode/rocm/rrun.sh"))
ASKPASS = pathlib.Path("/tmp/opencode/rocm/askpass.sh")
KNOWN_HOSTS = pathlib.Path("/tmp/opencode/rocm/known_hosts")
REMOTE_DIR = "/tmp/crowdclient"
PY = "/root/rocm-venv/bin/python"


def _env() -> dict:
    out = dict(os.environ)
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return out


def _members() -> list:
    """The client's own import closure: entry point, paradigms, ng helpers."""
    return ["client.py"] + sorted(p.name for p in REPO.glob("crowd_*.py")) + [
        str(p.relative_to(REPO)) for p in sorted((REPO / "crowdgpt_ng").glob("*.py"))
    ]


def pack() -> int:
    TARBALL.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TARBALL, "w:gz") as tar:
        for rel in _members():
            tar.add(REPO / rel, arcname=rel)
    return TARBALL.stat().st_size


def scp(env: dict) -> None:
    host = f"{env['ROCM_USERNAME']}@{env['ROCM_TEST_IP']}"
    cmd = ["scp", "-o", "StrictHostKeyChecking=accept-new",
           "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
           "-o", "NumberOfPasswordPrompts=1", "-o", "ConnectTimeout=10",
           str(TARBALL), f"{host}:/tmp/client.tgz"]
    run_env = dict(env, SSH_ASKPASS=str(ASKPASS), SSH_ASKPASS_REQUIRE="force", DISPLAY=":0")
    subprocess.run(cmd, env=run_env, check=True, stdout=subprocess.DEVNULL)


def driver(args) -> str:
    """Short script (a few hundred bytes) — this is what rrun can carry."""
    lines = [
        "set -u",
        f"rm -rf {REMOTE_DIR} && mkdir -p {REMOTE_DIR}",
        f"tar xzf /tmp/client.tgz -C {REMOTE_DIR}",
        f"cd {REMOTE_DIR}",
        f"echo files=$(ls | wc -l) client=$(wc -c < client.py) bytes",
    ]
    common = "--server {} --no-update --headless".format(args.url) if args.url else ""
    if args.url and args.register:
        lines.append(f"{PY} client.py {common} --register --user {shlex.quote(args.register)}"
                     f" --email {shlex.quote(args.email or args.register + '@example.invalid')}"
                     " || echo REGISTER_FAILED")
    if args.url:
        tok = f"CROWDGPT_TOKEN={shlex.quote(args.token)} " if args.token else ""
        lines += [
            f"{tok}{PY} client.py {common} --doctor --json 2>&1 | tail -40",
            f"{tok}{PY} client.py {common} --dry-run --no-tokenizer; echo DRYRUN_EXIT=$?",
        ]
    else:
        lines.append(f"{PY} client.py --doctor --json 2>&1 | tail -40")
    lines.append("echo PUSH_DONE")
    path = pathlib.Path("/tmp/opencode/rocm/push_driver.sh")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="")
    ap.add_argument("--token", default="")
    ap.add_argument("--register", default="", help="username to create on the coordinator")
    ap.add_argument("--email", default="")
    a = ap.parse_args()

    env = _env()
    for key in ("ROCM_USERNAME", "ROCM_TEST_IP"):
        if not env.get(key):
            sys.exit(f"missing {key} in {ENV_FILE}")

    size = pack()
    print(f"packed {size:,} bytes: {len(_members())} files")
    scp(env)
    print("scp ok")

    script = driver(a)
    proc = subprocess.run(["bash", str(RRUN), script], env=env,
                          capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.stderr.strip():
        sys.stderr.write(proc.stderr[-2000:])
    return proc.returncode


if __name__ == "__main__":
    main()
