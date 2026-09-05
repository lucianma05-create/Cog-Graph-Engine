"""Minimal HTTP server (stdlib): static viewer + JSON API.

Endpoints:
  GET  /                        viewer page
  GET  /viewer.css|/viewer.js|/vendor/vis-network.min.js   static assets
  GET  /api/seeds               seed list
  GET  /api/state?seed=<id>     full session (auto-Init on first visit)
  POST /api/reset               {seed_id} -> re-run Init
  POST /api/step                {seed_id, mode, agent_reply?, system_prompt?}
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from engine import llm
from engine.prompts import AGENT_DEFAULT
from engine.schema import Seed, StepRequest
from engine import agent as agent_mod

STATIC_DIR = Path(__file__).parent / "static"

STATIC_FILES = {
    "/": "index.html",
    "/viewer.css": "viewer.css",
    "/viewer.js": "viewer.js",
    "/vendor/vis-network.min.js": "vendor/vis-network.min.js",
}

CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "application/javascript; charset=utf-8"}


def _lean_initial(initial: dict) -> dict:
    """API payload keeps the log file as the audit record; llm_raw stays on disk only."""
    return {k: v for k, v in initial.items() if k != "llm_raw"}


def _lean_turn(turn: dict) -> dict:
    return {k: v for k, v in turn.items() if k not in ("llm_raw", "system_prompt_used")}


def state_payload(seed: Seed, log: dict, sim=None) -> dict:
    payload = {
        "seed_id": seed.seed_id,
        "task": seed.task.value,
        "seed": {
            "persona": seed.persona,
            "private_persona": seed.private_persona,
            "agent_private": seed.agent_private,
            "scenario": seed.scenario,
            "u0": seed.u0,
            "pre_context": [u.model_dump() for u in seed.pre_context],
            "reference_transcript": [u.model_dump() for u in seed.reference_transcript],
            # simulator-only character; the viewer is the simulator owner's
            # window — the agent LLM never receives this payload
            "cognitive_style": seed.cognitive_style,
            "cognitive_profile": seed.cognitive_profile,
        },
        "initial": _lean_initial(log["initial"]),
        "turns": [_lean_turn(t) for t in log["turns"]],
        "busy": False,
        "agent_prompt_default": AGENT_DEFAULT[seed.task],
        "llm": {"model": llm.MODEL},
    }
    if sim is not None:
        done, reason = sim.done
        payload["done"] = {"done": done, "done_reason": reason}
    else:
        last = log["turns"][-1] if log["turns"] else None
        payload["done"] = {"done": bool(last and last.get("done")),
                           "done_reason": (last or {}).get("done_reason")}
    return payload


def make_handler(store) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "CogGraphEngine/0.1"

        # ---------------------------------------------------------- plumbing

        def log_message(self, fmt: str, *args) -> None:  # keep stderr readable
            sys.stderr.write("[server] %s - %s\n" % (self.address_string(), fmt % args))

        def _send_json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_static(self, name: str) -> None:
            path = STATIC_DIR / name
            if not path.is_file():
                self._send_json(404, {"error": f"static file missing: {name}"})
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPES[path.suffix])
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ------------------------------------------------------------ routes

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in STATIC_FILES:
                self._send_static(STATIC_FILES[parsed.path])
            elif parsed.path == "/api/report":
                self._handle_report(parse_qs(parsed.query))
            elif parsed.path == "/api/seeds":
                self._send_json(200, {"seeds": store.list_seeds()})
            elif parsed.path == "/api/state":
                self._handle_state(parse_qs(parsed.query))
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/step":
                    self._handle_step()
                elif parsed.path == "/api/reset":
                    self._handle_reset()
                else:
                    self._send_json(404, {"error": "not found"})
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON body"})

        # ---------------------------------------------------------- actions

        def _handle_report(self, query: dict) -> None:
            seed_id = (query.get("seed") or [""])[0]
            path = store.sessions_dir / "reports" / f"{seed_id}.md"
            if not path.is_file():
                self._send_json(404, {"error": "no trajectory report yet; run a step first"})
                return
            body = path.read_text(encoding="utf-8").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{seed_id}_trajectory.md"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_state(self, query: dict) -> None:
            seed_id = (query.get("seed") or [""])[0]
            if not seed_id:
                self._send_json(400, {"error": "missing ?seed=<id>"})
                return
            lock = store.acquire(seed_id)
            if lock is None:
                self._send_json(409, {"error": "busy", "busy": True})
                return
            try:
                sim = store.get_sim(seed_id)  # may run Init (one LLM call)
                self._send_json(200, state_payload(sim.seed, sim.log(), sim))
            except KeyError:
                self._send_json(404, {"error": f"unknown seed {seed_id!r}"})
            except llm.ConfigError as e:
                self._send_json(500, {"error": str(e)})
            except llm.SchemaError as e:
                self._send_json(422, {"error": f"LLM schema failure: {e}"})
            except Exception as e:  # pragma: no cover - last resort
                self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
            finally:
                lock.release()

        def _handle_reset(self) -> None:
            body = self._read_json()
            seed_id = body.get("seed_id") or ""
            regenerate = bool(body.get("reinit"))
            lock = store.acquire(seed_id)
            if lock is None:
                self._send_json(409, {"error": "busy", "busy": True})
                return
            try:
                # re-Init; G0 reused from its persisted artifact unless reinit=true
                sim = store.reset_sim(seed_id, regenerate=regenerate)
                self._send_json(200, state_payload(sim.seed, sim.log(), sim))
            except KeyError:
                self._send_json(404, {"error": f"unknown seed {seed_id!r}"})
            except llm.ConfigError as e:
                self._send_json(500, {"error": str(e)})
            except llm.SchemaError as e:
                self._send_json(422, {"error": f"LLM schema failure: {e}"})
            except Exception as e:
                self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
            finally:
                lock.release()

        def _handle_step(self) -> None:
            try:
                req = StepRequest.model_validate(self._read_json())
            except Exception as e:
                self._send_json(400, {"error": f"invalid request: {e}"})
                return
            lock = store.acquire(req.seed_id)
            if lock is None:
                self._send_json(409, {"error": "busy", "busy": True})
                return
            try:
                sim = store.get_sim(req.seed_id)
                done, reason = sim.done
                if done:
                    self._send_json(409, {"error": f"session already done: {reason or 'no reason'}",
                                          "done": True})
                    return
                if req.mode == "manual":
                    agent_reply = (req.agent_reply or "").strip()
                    prompt_used = None
                else:
                    prompt_used = (req.system_prompt or "").strip() or AGENT_DEFAULT[sim.seed.task]
                    agent_reply = agent_mod.generate_agent_reply(
                        sim.seed, sim.history, prompt_used
                    )
                turn = sim.step(agent_reply, mode=req.mode, system_prompt_used=prompt_used)
                payload = state_payload(sim.seed, sim.log(), sim)
                payload["last_turn"] = _lean_turn(turn)
                self._send_json(200, payload)
            except KeyError:
                self._send_json(404, {"error": f"unknown seed {req.seed_id!r}"})
            except llm.ConfigError as e:
                self._send_json(500, {"error": str(e)})
            except llm.SchemaError as e:
                self._send_json(422, {"error": f"LLM schema failure: {e}"})
            except Exception as e:
                self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
            finally:
                lock.release()

    return Handler


def make_server(host: str, port: int, store) -> ThreadingHTTPServer:
    handler = make_handler(store)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd
