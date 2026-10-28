#!/usr/bin/env python3
"""Offline release signing for CrowdGPT's auto-updater.   Dev-machine tool: clients never run this.

    python tools/sign_release.py gen-key --out ~/secure/crowdgpt_release.pem     # once; prints the line for client.py
    python tools/sign_release.py release                                         # every release: hash + sign + self-check
    python tools/sign_release.py verify                                          # re-check what you are about to push
(`release` uses ~/release.pem unless you pass --key.  Asks for the passphrase once.)

`release` reads CLIENT_VERSION, hashes client.py and client.html, writes docs/version.txt
(line 1 = version, then sha256_clientpy:, sha256_clienthtml:, expires:) and signs those exact bytes
into docs/version.txt.sig.
Needs:  pip install cryptography      (not a client dependency)
"""
import argparse, base64, datetime, getpass, hashlib, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIENT, HTML = ROOT / "client.py", ROOT / "client.html"
VERSION_TXT, SIG = ROOT / "docs" / "version.txt", ROOT / "docs" / "version.txt.sig"


def _crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
        return serialization, Ed25519PrivateKey, Ed25519PublicKey
    except ImportError:
        sys.exit("This tool needs the 'cryptography' package:  pip install cryptography")


def _pub_b64(priv):
    ser, *_ = _crypto()
    return base64.b64encode(priv.public_key().public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw)).decode()


def _client_bytes(path=CLIENT):
    return path.read_bytes().replace(b"\r\n", b"\n")      # repo stores LF (.gitattributes)


def _client_version(code: bytes):
    m = re.search(rb'^CLIENT_VERSION\s*=\s*["\']([^"\']+)["\']', code, re.M)
    if not m:
        sys.exit("CLIENT_VERSION not found in client.py")
    return m.group(1).decode()


def cmd_gen_key(a):
    ser, Priv, _ = _crypto()
    out = Path(a.out).expanduser()
    if out.exists():
        sys.exit(f"{out} already exists; refusing to overwrite a signing key.")
    pw = getpass.getpass("Passphrase to encrypt the private key (do not leave empty): ").encode()
    if not pw or getpass.getpass("Repeat: ").encode() != pw:
        sys.exit("Empty or mismatching passphrase.")
    priv = Priv.generate()
    out.write_bytes(priv.private_bytes(ser.Encoding.PEM, ser.PrivateFormat.PKCS8, ser.BestAvailableEncryption(pw)))
    out.chmod(0o600)
    print(f"Private key written to {out}  (keep it OFFLINE, back it up; never commit it).")
    print("Public key line for client.py  UPDATE_PUBKEYS = [ ... ]:\n")
    print(f'    "{_pub_b64(priv)}",')


def _load(path):
    ser, *_ = _crypto()
    pw = getpass.getpass("Key passphrase: ").encode()
    return ser.load_pem_private_key(Path(path).expanduser().read_bytes(), pw)


def _embedded_keys(src):
    try:
        start = src.index("UPDATE_PUBKEYS")
    except ValueError:
        sys.exit("ERROR: 'UPDATE_PUBKEYS' marker not found in client.py — "
                 "update tools/sign_release.py to match the client.")
    try:
        lb = src.index("[", start)
        rb = src.index("]", lb)
    except ValueError:
        sys.exit("ERROR: could not find UPDATE_PUBKEYS list boundaries in client.py")
    body = src[lb + 1:rb]
    # tolerate a missing trailing comma on the last entry
    return set(re.findall(r'"([A-Za-z0-9+/=]{44})",?', body))

def cmd_release(a):
    code = _client_bytes()
    version = _client_version(code)
    if not HTML.exists():
        sys.exit("client.html not found: the client updates both files, so both hashes are required.")
    lines = [version, f"sha256_clientpy: {hashlib.sha256(code).hexdigest()}",
             f"sha256_clienthtml: {hashlib.sha256(_client_bytes(HTML)).hexdigest()}"]
    if not a.no_expiry:
        lines.append("expires: " + (datetime.date.today() + datetime.timedelta(days=a.expires_days)).isoformat())
    manifest = ("\n".join(lines) + "\n").encode()
    priv = _load(a.key)
    if _pub_b64(priv) not in _embedded_keys(CLIENT.read_text(encoding="utf-8")):
        sys.exit("STOP: this key's public half is NOT in UPDATE_PUBKEYS in client.py, so clients would reject "
                 f"the release.\nIts public key is: {_pub_b64(priv)}")
    sig = base64.b64encode(priv.sign(manifest)).decode() + "\n"
    VERSION_TXT.write_bytes(manifest)
    SIG.write_text(sig, encoding="ascii", newline="\n")
    print(f"Signed release {version}\n{manifest.decode()}")
    print(f"signing key: {_pub_b64(priv)}")
    cmd_verify(a)      # prove a real client would accept it before you push
    print("Now: git add client.py client.html docs/version.txt docs/version.txt.sig && git commit && git push  (and merge nothing into client.py afterwards)")


def cmd_verify(a):
    """Verify with the client's own code path, so a green result means a real client would accept it."""
    src = CLIENT.read_text(encoding="utf-8")
    import types
    mod = types.ModuleType("client_verify"); mod.__dict__.update(base64=base64, hashlib=hashlib, datetime=datetime, UPDATE_PUBKEYS=[])
    start, end = src.index("_ED_P = 2 **"), src.index("def _fetch_remote_version")
    exec(src[start:end], mod.__dict__)
    keys = _embedded_keys(src)
    if not keys:
        sys.exit("FAIL: UPDATE_PUBKEYS in client.py is empty.")
    manifest, sig = VERSION_TXT.read_bytes(), SIG.read_text()
    if not mod._verify_manifest(manifest, sig, keys):
        sys.exit("FAIL: docs/version.txt.sig does not verify against the keys in client.py.")
    m = mod._parse_manifest(manifest)
    if (hashlib.sha256(_client_bytes()).hexdigest() != m.get("sha256_clientpy")
            or hashlib.sha256(_client_bytes(HTML)).hexdigest() != m.get("sha256_clienthtml")):
        sys.exit("FAIL: client.py/client.html changed after signing (hash mismatch). Run `release` again.")
    if m["version"] != _client_version(_client_bytes()):
        sys.exit("FAIL: version.txt and CLIENT_VERSION disagree.")
    print(f"OK: release {m['version']} verifies under the keys embedded in client.py.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen-key"); g.add_argument("--out", required=True); g.set_defaults(fn=cmd_gen_key)
    r = sub.add_parser("release"); r.add_argument("--key", default="~/release.pem")
    r.add_argument("--expires-days", type=int, default=180); r.add_argument("--no-expiry", action="store_true"); r.set_defaults(fn=cmd_release)
    v = sub.add_parser("verify"); v.set_defaults(fn=cmd_verify)
    a = ap.parse_args(); a.fn(a)
