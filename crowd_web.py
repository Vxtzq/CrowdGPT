"""Web UI for the CrowdGPT client: config, token, the FastAPI app, its server thread and the event stream.

client.py imports this lazily, only on the web path; the updater ships only client.py and client.html,
so this file may be missing on an auto-updated install. Never `import client` here: run as
`python client.py` the client is `__main__`, and a second copy would have its own globals. Whatever
the web side needs (the Api instance, the token path, the rendered page) is passed in.

FastAPI and uvicorn are imported on demand too, so importing this module needs only the stdlib.
"""
import asyncio
import contextlib
import inspect
import ipaddress
import json
import os
import re
import secrets
import threading
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple, Optional


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

def create_web_app(api, opts):
    """FastAPI app: opts["html"] at /, a health check, allow-listed Api calls and, with opts["sink"],
    the /events stream. The docs endpoints are off."""
    fastapi, _ = _import_web_stack()
    from fastapi.responses import HTMLResponse, Response
    from starlette.concurrency import run_in_threadpool
    sink = opts.get("sink")
    app = fastapi.FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=_web_lifespan(sink))
    html = opts["html"]
    if sink: _add_events_route(app, sink, float(opts.get("heartbeat", 15)))

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
