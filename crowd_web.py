"""Web UI for the CrowdGPT client: config, token, the FastAPI app, its server thread and the event stream.

client.py imports this lazily, only on the web path; the updater ships only client.py and client.html,
so this file may be missing on an auto-updated install. Never `import client` here: run as
`python client.py` the client is `__main__`, and a second copy would have its own globals. Whatever
the web side needs (the Api instance, the token path, the rendered page) is passed in.

FastAPI and uvicorn are imported on demand too, so importing this module needs only the stdlib.
"""
import asyncio
import contextlib
import hmac
import inspect
import ipaddress
import json
import logging
import os
import re
import secrets
import socket
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from typing import NamedTuple, Optional

log = logging.getLogger(__name__)


def _write_private(path, text):
    """Write text to path readable by the owner only (0600)."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f: f.write(text)
    with contextlib.suppress(OSError): os.chmod(str(path), 0o600)


WEB_AUTH_MODES = ("auto", "token", "none")
WEB_TOKEN_MIN_LEN = 16

class WebConfig(NamedTuple):
    host: str
    port: int
    auth: str
    token: Optional[str]
    allowed_hosts: tuple
    any_host: bool

def _web_port(raw):
    try: port = 8765 if raw is None else int(raw)
    except (TypeError, ValueError): raise ValueError(f"web port must be a number, got {raw!r}") from None
    if not 1 <= port <= 65535: raise ValueError(f"web port must be 1-65535, got {port}")
    return port

def _web_allowed_hosts(raw):
    hosts = []
    for item in (raw or "").split(","):
        h = item.strip().lower()
        if h and h not in hosts: hosts.append(h)
    return tuple(hosts)

def resolve_web_config(args, env):
    """Web UI settings from argparse flags and an env mapping; a flag beats its env variable.

    An empty env value counts as unset. Raises ValueError on a bad auth mode, port or token."""
    def pick(flag, name):
        v = getattr(args, flag, None)
        if v is not None: return v
        return (env.get(name) or "").strip() or None
    host = pick("web_host", "CROWDGPT_WEB_HOST") or "127.0.0.1"
    port = _web_port(pick("web_port", "CROWDGPT_WEB_PORT"))
    auth = str(pick("web_auth", "CROWDGPT_WEB_AUTH") or "auto").strip().lower()
    if auth not in WEB_AUTH_MODES:
        raise ValueError(f"web auth mode must be one of {', '.join(WEB_AUTH_MODES)}, got {auth!r}")
    token = pick("web_token", "CROWDGPT_WEB_TOKEN") or None
    if token is not None and len(token) < WEB_TOKEN_MIN_LEN:
        raise ValueError(f"web token must be at least {WEB_TOKEN_MIN_LEN} characters")
    hosts = _web_allowed_hosts(env.get("CROWDGPT_WEB_ALLOWED_HOSTS"))
    return WebConfig(host, port, auth, token, hosts, "*" in hosts)

def resolve_web_token(override, path):
    """The web UI token: the override, else the saved file, else a new one saved with mode 0600.

    An empty file counts as missing. Read errors other than a missing file propagate."""
    if override: return override
    try: saved = Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError: saved = ""
    if saved: return saved
    token = secrets.token_urlsafe(32)
    _write_private(path, token + "\n")
    return token

def is_loopback(host):
    """True for 127.0.0.0/8, ::1 (also bracketed or IPv4-mapped) and `localhost`. No DNS lookup."""
    h = str(host).strip().strip("[]").lower()
    if h == "localhost": return True
    try: ip = ipaddress.ip_address(h)
    except ValueError: return False
    return (getattr(ip, "ipv4_mapped", None) or ip).is_loopback

def auth_required(mode, host):
    """Whether the web UI asks for a token: always for `token`, never for `none`, off loopback for `auto`."""
    if mode == "token": return True
    if mode == "none": return False
    if mode == "auto": return not is_loopback(host)
    raise ValueError(f"web auth mode must be one of {', '.join(WEB_AUTH_MODES)}, got {mode!r}")

def web_url(sockname):
    """Display URL for a bound socket address (2- or 4-tuple); IPv6 hosts get brackets."""
    host, port = sockname[0], sockname[1]
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"

def _import_web_stack():
    """Import FastAPI and uvicorn on demand: the updater can ship this file to installs without them."""
    try:
        import fastapi
        import uvicorn
    except ImportError as e:
        raise RuntimeError("the web UI needs FastAPI and uvicorn: pip install fastapi uvicorn") from e
    return fastapi, uvicorn

# Stands in for pywebview in a browser: api calls POST to /api/<name>, events arrive over SSE.
# Both the stream and pywebviewready wait for DOMContentLoaded so the page's handlers exist.
_WEB_SHIM = (
    "<script>window.__CROWD_WEB__=%s;(function(){"
    "var api=new Proxy({},{get:function(_,n){if(typeof n!=='string'||n==='then')return undefined;"
    "return function(){return fetch('/api/'+encodeURIComponent(n),{method:'POST',credentials:'same-origin',"
    "headers:{'Content-Type':'application/json'},body:JSON.stringify(Array.prototype.slice.call(arguments))})}}});"
    "window.pywebview={api:api};"
    "document.addEventListener('DOMContentLoaded',function(){"
    "var es=new EventSource('/events');es.onmessage=function(m){var p;try{p=JSON.parse(m.data)}catch(e){return}"
    "if(window.handleEvent)window.handleEvent(p.ev,p.data)};"
    "window.dispatchEvent(new Event('pywebviewready'))})})();</script>"
)

def render_web_html(html, opts):
    """client.html for a browser: html has its placeholders filled already; the bridge shim goes
    before the page's first script."""
    shim = _WEB_SHIM % json.dumps(opts).replace("</", "<\\/")
    m = re.search(r"<script", html, re.IGNORECASE)
    at = m.start() if m else len(html)
    return html[:at] + shim + html[at:]

def _web_status(api):
    """Coarse training state for /healthz: whether the training thread is alive."""
    t = getattr(api, "thread", None)
    return "running" if t is not None and t.is_alive() else "idle"

# The Api methods client.html calls; nothing else on Api is reachable over HTTP.
WEB_API_METHODS = frozenset({"get_backends", "set_gpu", "set_vram", "set_backend", "login",
                             "register", "login_anon", "start", "stop", "logout"})

def _web_api_args(body, fn):
    """Positional args from a JSON array body, or None if it isn't one or doesn't fit fn's signature."""
    try: args = json.loads(body)
    except ValueError: return None
    if not isinstance(args, list): return None
    try: inspect.signature(fn).bind(*args)
    except TypeError: return None
    return args

def _web_lifespan(sink):
    """Binds the sink to the server loop for the app's lifetime, so send() has somewhere to deliver."""
    @contextlib.asynccontextmanager
    async def lifespan(_app):
        if sink: sink.bind(asyncio.get_running_loop())
        try: yield
        finally:
            if sink: sink.unbind()
    return lifespan

def _last_event_id(raw):
    """The Last-Event-ID header as an int, or None when it's missing or not a number."""
    try: return int(raw)
    except (TypeError, ValueError): return None

def _add_events_route(app, sink, heartbeat):
    """GET /events: the sink's frames as Server-Sent Events, with a comment line as heartbeat. A new
    stream starts with the sink's backlog for the browser's Last-Event-ID."""
    from fastapi import Request
    from fastapi.responses import StreamingResponse

    async def stream(last_id):
        q = sink.subscribe()
        backlog = sink.backlog(last_id)  # no await since subscribe: nothing published in between
        try:
            for frame in backlog: yield frame
            while True:
                try: yield await asyncio.wait_for(q.get(), heartbeat)
                except asyncio.TimeoutError: yield ": ping\n\n"
        finally: sink.unsubscribe(q)  # runs when the client disconnects and the generator is cancelled

    @app.get("/events")
    async def events(request: Request):
        last_id = _last_event_id(request.headers.get("last-event-id"))
        return StreamingResponse(stream(last_id), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

WEB_AUTH_COOKIE = "crowdgpt_web_token"
_TOKEN_PAGE = (b"<!doctype html><meta charset=utf-8><title>CrowdGPT</title>"
               b"<form method=get action=/><p>Web UI token: <input name=\"token\" type=password autofocus>"
               b" <button>Open</button></p></form>")

def _token_matches(candidate, token):
    """Constant-time comparison; an empty or non-ASCII candidate never matches."""
    if not candidate: return False
    try: return hmac.compare_digest(candidate.encode("ascii"), token.encode("ascii"))
    except UnicodeEncodeError: return False

def _cookie_value(raw, name):
    """One cookie's value from a raw Cookie header, or None."""
    for part in raw.split(";"):
        k, _, v = part.strip().partition("=")
        if k == name: return v
    return None

class _TokenGate:
    """ASGI middleware: every HTTP request but /healthz needs the token as a Bearer header, the cookie
    or ?token= (which sets the cookie and redirects). Others get the token page on / or a 401."""
    def __init__(self, app, token):
        self.app, self.token = app, token

    def _authorized(self, headers):
        """A matching Bearer header or cookie."""
        scheme, _, cred = headers.get("authorization", "").partition(" ")
        if scheme.lower() == "bearer" and _token_matches(cred.strip(), self.token): return True
        return _token_matches(_cookie_value(headers.get("cookie", ""), WEB_AUTH_COOKIE), self.token)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/healthz":
            return await self.app(scope, receive, send)
        if self._authorized({k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}):
            return await self.app(scope, receive, send)
        query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        if scope["method"] == "GET" and _token_matches((query.get("token") or [""])[0], self.token):
            cookie = f"{WEB_AUTH_COOKIE}={self.token}; HttpOnly; SameSite=Strict; Path=/"
            return await _respond(send, 303, [(b"location", scope["path"].encode()), (b"set-cookie", cookie.encode())])
        if scope["path"] == "/":
            return await _respond(send, 401, [(b"content-type", b"text/html; charset=utf-8")], _TOKEN_PAGE)
        await _respond(send, 401)

async def _respond(send, status, headers=(), body=b""):
    await send({"type": "http.response.start", "status": status,
                "headers": [*headers, (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})

def cli_credentials(args, env):
    """The login the CLI asked for (--anon, --token, --user/--password or their env variables)."""
    return {"anon": bool(args.anon), "token": args.token or env.get("CROWDGPT_TOKEN"),
            "user": args.user, "password": args.password or env.get("CROWDGPT_PASSWORD")}

def _saved_for_server(saved, server):
    """(token, username) from a saved session for this server, or None."""
    saved = saved or {}
    if saved.get("token") and (saved.get("server") or "").rstrip("/") == server.rstrip("/"):
        return saved["token"], saved.get("username")
    return None

def _password_login(creds, server, do_login):
    if not creds.get("password"): return "missing_password"
    tok, err = do_login(server, creds["user"], creds["password"])
    return (tok, creds["user"]) if tok else (err or "login_failed")

def resolve_cli_login(creds, server, saved, do_login):
    """In headless order: anon, token, user + password, then the saved session for this server.
    Returns (token, username), an error code string, or None when nothing was given."""
    if creds.get("anon"): return None, "anonymous"
    if creds.get("token"): return creds["token"], creds.get("user")
    if creds.get("user"): return _password_login(creds, server, do_login)
    return _saved_for_server(saved, server)

def web_autologin(api, creds, saved, do_login, autostart=False):
    """Log the Api in from CLI credentials, without saving them; start training if autostart.
    Emits login_success or login_error like the page's own login. True on login."""
    got = resolve_cli_login(creds, api.server_url, saved, do_login)
    if got is None: return False  # nothing given: the page shows its login screen
    if isinstance(got, str):
        api.emit("login_error", {"code": got}); return False
    api.auth_token, api.username = got
    api.emit("login_success", {"username": api.username or "restored", "autologin": True})
    if autostart: api.start()
    return True

_LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")

def _is_unspecified(host):
    """True for the listen-on-everything addresses (0.0.0.0, ::); False for names."""
    try: return ipaddress.ip_address(host).is_unspecified
    except ValueError: return False

def web_allowed_hosts(bind, extra):
    """Host names the web UI answers to: loopback, the bind address and extra; for 0.0.0.0/:: also
    this machine's hostname and its addresses. None when extra has `*` (the check is off)."""
    if "*" in extra: return None
    bind = str(bind).strip().strip("[]").lower()
    hosts = {*_LOOPBACK_NAMES, *(h.lower() for h in extra)}
    if not _is_unspecified(bind):
        return frozenset(hosts | {bind})
    names = {n.lower() for n in (socket.gethostname(), socket.getfqdn()) if n}
    hosts |= names
    for name in names:
        try: infos = socket.getaddrinfo(name, None)
        except OSError: continue
        hosts |= {info[4][0].lower() for info in infos}
    return frozenset(hosts)

def _hostname(netloc):
    """The lowercased host part of host[:port] or [v6][:port], or None when it doesn't parse."""
    try: return urlsplit("//" + netloc).hostname
    except ValueError: return None

class _HostGate:
    """ASGI middleware against DNS rebinding and CSRF: the Host header must be in `allowed`, and a
    POST's Origin, when sent, must be allowed or the same as Host. `allowed=None` turns both off."""
    def __init__(self, app, allowed):
        self.app, self.allowed = app, allowed

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        host = headers.get("host", "")
        if _hostname(host) not in self.allowed:
            return await _respond(send, 400)
        origin = headers.get("origin")
        if scope["method"] == "POST" and origin is not None and not self._origin_ok(origin, host):
            return await _respond(send, 403)
        await self.app(scope, receive, send)

    def _origin_ok(self, origin, host):
        try: parts = urlsplit(origin)
        except ValueError: return False
        return parts.hostname in self.allowed or (bool(parts.netloc) and parts.netloc.lower() == host.lower())

def create_web_app(api, opts):
    """FastAPI app: opts["html"] at /, a health check, allow-listed Api calls and, with opts["sink"],
    the /events stream. The docs endpoints are off. With opts["token"], every route but /healthz
    needs it (see _TokenGate). With opts["allowed_hosts"] (from web_allowed_hosts), Host and Origin
    are checked first (see _HostGate); missing or None turns that check off."""
    fastapi, _ = _import_web_stack()
    from fastapi.responses import HTMLResponse, Response
    from starlette.concurrency import run_in_threadpool
    sink = opts.get("sink")
    app = fastapi.FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=_web_lifespan(sink))
    html = opts["html"]
    if sink: _add_events_route(app, sink, float(opts.get("heartbeat", 15)))
    if opts.get("token"): app.add_middleware(_TokenGate, token=opts["token"])
    if opts.get("allowed_hosts") is not None: app.add_middleware(_HostGate, allowed=opts["allowed_hosts"])  # added last, runs first

    @app.post("/api/{method}")
    async def call_api(method: str, request: fastapi.Request):
        if method not in WEB_API_METHODS: return Response(status_code=404)
        fn = getattr(api, method)
        args = _web_api_args(await request.body(), fn)
        if args is None: return Response(status_code=400)
        await run_in_threadpool(fn, *args)  # Api methods block; keep them off the event loop
        return Response(status_code=204)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return html

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "status": _web_status(api)}

    return app

class WebServer:
    """uvicorn in a daemon thread. Off the main thread uvicorn leaves signal handling to the caller."""
    def __init__(self, app, host, port):
        _, uvicorn = _import_web_stack()
        self.server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, name="web-server", daemon=True)

    def start(self, timeout=10.0):
        """Start serving; return the bound socket address once listening."""
        self.thread.start()
        deadline = time.monotonic() + timeout
        while not self.server.started:
            if not self.thread.is_alive(): raise RuntimeError("web server failed to start")
            if time.monotonic() > deadline: raise TimeoutError("web server did not start in time")
            time.sleep(0.02)
        return self.server.servers[0].sockets[0].getsockname()

    def stop(self, timeout=10.0):
        self.server.should_exit = True
        if self.thread.is_alive(): self.thread.join(timeout)


# Event -> state group. backlog() replays the latest frame per group; events in one group overwrite
# each other (a logout replaces the login). Events not listed, like login_error, aren't replayed.
WEB_STATE_GROUPS = {
    "login_success": "auth", "logged_out": "auth",
    "started": "train", "stopped": "train", "start_refused": "train",
    "overlay_show": "overlay", "overlay_hide": "overlay", "overlay_progress": "overlay_progress",
    "status": "status", "stats": "stats", "cal_stats": "stats", "model_preview": "preview",
    "backends": "backends", "vram": "vram", "gpus": "gpus", "meta": "meta",
}

class WebSink:
    """Delivers Api events to browsers: send() hands each event to the server loop, which fans it out
    into one bounded queue per subscriber (oldest dropped when full), so a slow browser never blocks emit.

    It also keeps the latest frame per state group and a ring of log frames, so backlog() can bring a
    new or reconnecting browser up to date. All of that state is only touched on the loop thread."""
    def __init__(self, queue_size=256, log_size=500):
        self._queue_size = queue_size
        self._loop = None
        self._subs = set()
        self._seq = 0
        self._state = {}                        # group -> (seq, frame)
        self._log = deque(maxlen=log_size)      # (seq, frame)
        self._evicted = 0                       # highest seq that has left the log ring

    def bind(self, loop): self._loop = loop
    def unbind(self): self._loop = None
    def subscriber_count(self): return len(self._subs)

    def subscribe(self):
        q = asyncio.Queue(maxsize=self._queue_size)
        self._subs.add(q)
        return q

    def unsubscribe(self, q): self._subs.discard(q)

    def backlog(self, last_id=None):
        """Frames a browser that has seen up to last_id is missing, in seq order. The full snapshot
        (state plus log tail) when last_id is None, already out of the log ring, or ahead of the sink."""
        if last_id is None or last_id < self._evicted or last_id > self._seq: last_id = 0
        kept = [*self._state.values(), *self._log]
        return [frame for seq, frame in sorted(kept) if seq > last_id]

    def send(self, ev, data):
        loop = self._loop
        if loop is None or loop.is_closed(): return
        try: payload = json.dumps({"ev": ev, "data": data})
        except (TypeError, ValueError): return
        with contextlib.suppress(RuntimeError):  # the loop closed between the check and the call
            loop.call_soon_threadsafe(self._publish, ev, payload)

    def _publish(self, ev, payload):
        self._seq += 1
        frame = f"id: {self._seq}\ndata: {payload}\n\n"
        self._remember(ev, frame)
        for q in tuple(self._subs):
            if q.full(): q.get_nowait()
            q.put_nowait(frame)

    def _remember(self, ev, frame):
        if ev == "log":
            if len(self._log) == self._log.maxlen: self._evicted = self._log[0][0]
            self._log.append((self._seq, frame))
        elif ev in WEB_STATE_GROUPS:
            self._state[WEB_STATE_GROUPS[ev]] = (self._seq, frame)


def prepare_web(args, env, token_path):
    """(config, token) for run_web, before anything starts; token is None when auth isn't required.
    Raises ValueError on a bad setting."""
    cfg = resolve_web_config(args, env)
    token = resolve_web_token(cfg.token, token_path) if auth_required(cfg.auth, cfg.host) else None
    return cfg, token

def start_web(api, cfg, token, html, login):
    """Wire the Api to browsers and start serving: a WebSink, the page with its shim, the app with
    token and Host/Origin checks, the server thread. Then log in from the CLI without starting
    training. html has its placeholders filled; login is (creds, saved, do_login). Returns the server."""
    sink = WebSink(); api._add_sink(sink)
    page = render_web_html(html, {"auth": cfg.auth, "loopback": is_loopback(cfg.host)})
    app = create_web_app(api, {"html": page, "sink": sink, "token": token,
                               "allowed_hosts": web_allowed_hosts(cfg.host, cfg.allowed_hosts)})
    server = WebServer(app, cfg.host, cfg.port)
    log.info(f"[web] listening on {web_url(server.start())}")
    creds, saved, do_login = login
    web_autologin(api, creds, saved, do_login, autostart=False)
    return server
