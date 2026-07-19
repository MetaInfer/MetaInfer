"""FastAPI WebUI server. The long-lived main process.

Responsibilities:
  - Serve the static SPA frontend (from the sys-shell plugin's ``static/``)
  - Mount plugin routers (sys-shell at ``/api/sys-shell``, task types at
    ``/api/{type}/{task_id}``)
  - Push live updates via SSE (routes handled by sys-shell plugin)

No in-memory task state — every read endpoint derives from files on
disk (state_reader), every write goes through the launcher / task
registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import reconcile as _reconcile
from .registry import WebPlugin
from .registry import all_plugins as _all_web_plugins
from .registry import get as _get_web_plugin
from .. import tasks as _task_packages  # noqa: F401 — side-effect: register all task plugins (orchestrator + web)


def _stat_dir() -> Path:
    """Return the frontend static directory from the registered sys-shell plugin."""
    _plugin = _get_web_plugin("sys-shell")
    if _plugin is None or _plugin.frontend_dir is None:
        raise RuntimeError("sys-shell plugin not registered (did you import metainfer.tasks?)")
    return _plugin.frontend_dir


def create_app() -> FastAPI:
    app = FastAPI(title="MetaInfer", docs_url=None, redoc_url=None)

    # Reconcile runtime state with the actual process table. This makes
    # the WebUI crash-safe: on restart, any orchestrator subprocesses
    # that survived from the previous session are picked back up, and
    # stale entries for orchestrators that have died are cleaned.
    try:
        _reconcile.reconcile()
    except Exception as e:  # noqa: BLE001 — startup must not crash
        import sys
        print(f"[metainfer-web] reconciliation failed: {e!r}", file=sys.stderr)

    # ------------------------------------------------------------------ #
    # Plugin routers
    # ------------------------------------------------------------------ #
    # Each plugin's build_router(plugin) returns an APIRouter of relative-path
    # endpoints. sys-shell mounts at /api/sys-shell (no {task_id}); every other
    # task-type plugin mounts at /api/{type}/{task_id}.
    _plugin_by_type: Dict[str, Any] = {}
    for _plugin in _all_web_plugins():
        if _plugin.build_router is None:
            continue
        _router = _plugin.build_router(_plugin)
        if _router is None:
            continue
        _plugin_by_type[_plugin.type] = _plugin
        if _plugin.type == "sys-shell":
            app.include_router(_router, prefix="/api/sys-shell")
        else:
            app.include_router(
                _router,
                prefix=f"/api/{_plugin.type}/{{task_id}}",
            )

    # ------------------------------------------------------------------ #
    # Static frontend
    # ------------------------------------------------------------------ #
    _STATIC_DIR = _stat_dir()

    def _cache_bust_token() -> str:
        """Compute a version token from the latest mtime across all
        static files (web shell + every plugin's frontend dir).
        Embedded into index.html as ?v=<token> on every JS/CSS URL so
        the browser fetches fresh modules after any code change.
        """
        mtimes: List[float] = []
        roots = [_STATIC_DIR] + [
            p.frontend_dir for p in _all_web_plugins()
            if p.frontend_dir and p.frontend_dir.exists()
        ]
        for root in roots:
            try:
                mtimes.extend(pp.stat().st_mtime for pp in root.rglob("*") if pp.is_file())
            except OSError:
                pass
        return str(int(max(mtimes))) if mtimes else "0"

    # ------------------------------------------------------------------ #
    # Importmap + stylesheet merge — server-side single source of truth
    # ------------------------------------------------------------------ #
    #
    # The shell provides default entries for shared widgets (charts /
    # state-graph / iterations-table / etc.) directly inside
    # ``index.html``'s ``<script type="importmap">`` block. Plugins that
    # want to ship a divergent version of one of these widgets simply
    # include the same key in their ``importmap_entries`` — the merge
    # below lets plugin entries OVERRIDE shell entries with the same key.
    #
    # In addition, every ``*.js`` directly under a plugin's
    # ``frontend_dir`` is auto-registered as ``app/<stem>`` →
    # ``/static/plugins/<type>/<file>?v=<token>``. Plugin authors only
    # need to populate ``importmap_entries`` for keys that DON'T follow
    # the ``app/<filename-stem>`` convention, or to override shell
    # entries (e.g. ship a divergent ``app/state-graph``).
    #
    # Plugin stylesheets (``extra_stylesheets``) are injected as
    # ``<link>`` tags right after the shell stylesheet, so a plugin can
    # ship task-type-specific CSS without editing the shell's
    # ``styles.css``.
    #
    # We merge on the SERVER (rather than emitting duplicate keys and
    # relying on browser last-wins semantics) because duplicate importmap
    # keys are spec-undefined and Chrome rejects the entire importmap on
    # duplicates in some versions — server merge is the safe path.

    _SHELL_IMPORTMAP_CACHE: Dict[str, Any] = {}

    def _shell_importmap() -> Dict[str, str]:
        """Parse the shell importmap from index.html once, cache it."""
        if _SHELL_IMPORTMAP_CACHE:
            return _SHELL_IMPORTMAP_CACHE["entries"]
        import re
        html_text = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        m = re.search(
            r'<script[^>]*type="importmap"[^>]*>(.*?)</script>',
            html_text, re.DOTALL,
        )
        if not m:
            _SHELL_IMPORTMAP_CACHE["entries"] = {}
            return {}
        try:
            block = json.loads(m.group(1))
            entries = dict(block.get("imports", {}))
        except (ValueError, TypeError):
            entries = {}
        _SHELL_IMPORTMAP_CACHE["entries"] = entries
        return entries

    def _plugin_auto_importmap(plugin: "WebPlugin") -> Dict[str, str]:
        """Auto-discover importmap entries by globbing the plugin's
        ``frontend_dir`` for ``*.js``. Each file becomes
        ``app/<stem>`` → ``/static/plugins/<type>/<file>`` (CACHE_BUST
        substituted later). Explicit ``importmap_entries`` on the plugin
        override these."""
        out: Dict[str, str] = {}
        if not plugin.frontend_dir or not plugin.frontend_dir.is_dir():
            return out
        prefix = f"/static/plugins/{plugin.type}"
        try:
            files = sorted(plugin.frontend_dir.glob("*.js"))
        except OSError:
            return out
        for p in files:
            key = f"app/{p.stem}"
            out[key] = f"{prefix}/{p.name}?v=CACHE_BUST"
        return out

    def _merged_importmap(token: str) -> Dict[str, str]:
        """Merge shell + plugin importmap entries with a deliberate
        precedence (lowest → highest):

          1. Per-plugin auto-discovered entries (``app/<stem>`` for every
             ``*.js`` under ``frontend_dir``).
          2. Shell entries (the ``<script type="importmap">`` block in
             ``index.html``).
          3. Per-plugin explicit ``importmap_entries``.

        This means a plugin that happens to ship a file named
        ``utils.js`` does NOT silently hijack the shell's ``app/utils``.
        To override a shell entry, a plugin must opt in by listing it
        explicitly in ``importmap_entries`` — accidental filename
        collisions stay safe. Within (3), later-registered plugins win
        over earlier ones, mirroring the historical behavior.
        """
        merged: Dict[str, str] = {}
        for plugin in _all_web_plugins():
            for k, v in _plugin_auto_importmap(plugin).items():
                merged[k] = v.replace("CACHE_BUST", token)
        for k, v in _shell_importmap().items():
            merged[k] = v.replace("CACHE_BUST", token)
        for plugin in _all_web_plugins():
            for k, v in plugin.importmap_entries.items():
                merged[k] = v.replace("CACHE_BUST", token)
        return merged

    def _plugin_stylesheet_links(token: str) -> str:
        """Render ``<link>`` tags for every plugin stylesheet, in
        registration order. Returns the concatenated HTML (may be empty).

        Filename validation: each entry must be a bare filename (no path
        separators, no ``..``) ending in ``.css``. This is a defense-
        in-depth on top of the later ``relative_to`` check — it rejects
        backslash forms and odd inputs early, so a buggy or malicious
        plugin can't get any variety of escape into the rendered HTML.
        """
        import re as _re
        # One filename, ``.css`` suffix, no separators of any kind, no
        # leading dot. Allows ``foo.css`` / ``foo-bar.css`` / ``foo_1.css``.
        _VALID = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*\.css$")
        parts: List[str] = []
        for plugin in _all_web_plugins():
            if not plugin.extra_stylesheets or not plugin.frontend_dir:
                continue
            for fname in plugin.extra_stylesheets:
                if not isinstance(fname, str) or not _VALID.match(fname):
                    continue  # reject malformed entry; never inject raw
                target = plugin.frontend_dir / fname
                if not target.is_file():
                    # Form is valid but the file doesn't exist (typo in
                    # plugin config, or a stale entry). Skip rather than
                    # emit a link that 404s.
                    continue
                try:
                    rel = target.resolve().relative_to(
                        plugin.frontend_dir.resolve()
                    )
                except (ValueError, OSError):
                    continue  # symlink/resolution escape; skip
                url = (
                    f"/static/plugins/{plugin.type}/{rel.as_posix()}?v={token}"
                )
                parts.append(
                    f'  <link rel="stylesheet" href="{url}" />'
                )
        return "\n".join(parts)

    @app.get("/")
    def index() -> HTMLResponse:
        html_text = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        token = _cache_bust_token()
        merged = _merged_importmap(token)
        # Re-serialize the importmap block with the merged entries. This
        # avoids the duplicate-key problem (browsers' importmap spec
        # rejects duplicate top-level keys) and gives plugins a clean
        # override path.
        import re
        new_block = json.dumps({"imports": merged}, indent=2)
        # Preserve the 6-space indent used inside <script> in index.html
        # so the diff stays small. The regex matches the whole script
        # block regardless of original indent.
        indented = "\n".join("      " + line for line in new_block.splitlines())
        html_text = re.sub(
            r'(<script[^>]*type="importmap"[^>]*>)(.*?)(</script>)',
            lambda m: m.group(1) + "\n" + indented + "\n  " + m.group(3),
            html_text, count=1, flags=re.DOTALL,
        )
        # Inject plugin stylesheets right after the shell stylesheet.
        # The shell index.html has exactly one <link rel="stylesheet" href="/static/styles.css?v=CACHE_BUST">.
        plugin_links = _plugin_stylesheet_links(token)
        if plugin_links:
            html_text = html_text.replace(
                '<link rel="stylesheet" href="/static/styles.css?v=CACHE_BUST" />',
                '<link rel="stylesheet" href="/static/styles.css?v=CACHE_BUST" />\n'
                + plugin_links,
                1,
            )
        # Substitute CACHE_BUST anywhere else in the doc (e.g. <link>
        # stylesheet, the bootstrap main.js script tag).
        html_text = html_text.replace("CACHE_BUST", token)
        return HTMLResponse(
            content=html_text,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    # Each plugin with bundled frontend assets gets its own mount point
    # at /static/plugins/<type>/. The plugin's importmap_entries point
    # at URLs under this mount. Mount these BEFORE the generic /static
    # catch-all so the more-specific path wins.
    for _plugin in _all_web_plugins():
        if _plugin.frontend_dir and _plugin.frontend_dir.exists():
            app.mount(
                f"/static/plugins/{_plugin.type}",
                StaticFiles(directory=str(_plugin.frontend_dir)),
                name=f"static_plugin_{_plugin.type}",
            )

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Force no-cache on JS / CSS module responses so the browser always
    # revalidates. ES modules are aggressively cached by the browser; an
    # old state-graph.js with the VNode-concat bug will keep loading
    # from cache even after a Ctrl-Shift-R unless we explicitly send
    # no-cache. Revalidation is cheap (304s when nothing changed).
    @app.middleware("http")
    async def _no_cache_modules(request: Request, call_next):
        resp = await call_next(request)
        path = request.url.path
        if path.startswith("/static/") and (path.endswith(".js") or path.endswith(".css")):
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        # All /api/ endpoints read live state from disk (run.json,
        # agents.json, iterations/, ...). Without no-store the browser
        # applies heuristic caching and the Live sub-agents / Last
        # output columns go stale even though the orchestrator is
        # actively rewriting agents.json. The cost of no-store is one
        # small file read per poll — negligible.
        elif path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # On clean shutdown, clear the WebUI entry in runtime.json so the
    # next boot doesn't think the previous session is still running.
    # (If the WebUI crashes, reconcile() overwrites the stale entry on
    # next start anyway — this is just hygiene.)
    @app.on_event("shutdown")
    async def _on_shutdown():
        try:
            from . import runtime as _runtime
            _runtime.record_webui_exit()
        except Exception:  # noqa: BLE001 — never fail shutdown
            pass

    # Background liveness probe — detects orchestrators that died
    # mid-run while the WebUI was up (crash / OOM kill / agent killing
    # its own parent). reconcile() only runs at startup, so without
    # this the UI would freeze on a stale "running" snapshot until the
    # next WebUI restart. See metainfer.server.liveness.
    from . import liveness as _liveness
    _liveness.attach(app, interval=10.0)

    return app


# Module-level app instance for uvicorn / `metainfer-web` entry point.
app = create_app()


def main() -> int:
    """Entry point for the `metainfer-web` console script."""
    import uvicorn
    import os
    host = os.environ.get("METAINFER_HOST", "127.0.0.1")
    port = int(os.environ.get("METAINFER_PORT", "8765"))
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
