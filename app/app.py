"""Plan Reader: upload a floor plan, pick a model, find doors or ask a question.

Everything runs on this machine: the page talks to this server (localhost:7860), which starts and forwards to
llama-server (localhost:8080). Standard library only, no pip packages.

Models come in two setups, because a model whose vision side was trained (fpcvis) needs its own vision file
(mmproj) and llama-server loads one at a time:
  standard  official Qwen3-VL-4B vision file; the fpc adapter is loaded but applied per request, so the same
            server also answers as the untrained base model
  fpcvis    its own vision file and adapter (only if downloaded with `bash setup.sh --full`)
Picking a model from the other setup restarts llama-server (~10-30 s). An answer cut off at the length limit (the
model kept repeating boxes) is asked again once with the DRY sampler, which discourages repeats.

    bash run.sh          # then open http://localhost:7860
"""
from __future__ import annotations

import atexit
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("APP_PORT", "7860"))
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
LLAMA = f"http://localhost:{LLAMA_PORT}"
IMAGE_MAX_TOKENS = os.environ.get("IMAGE_MAX_TOKENS", "1024")   # about 1 megapixel, as in training
GPU_LAYERS = os.environ.get("GPU_LAYERS", "99")                 # ignored by CPU-only builds
BASE_GGUF = "models/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
SETUPS = {   # first available setup's first model is the default
    "fpcvis": {"mmproj": "models/mmproj-fpcvis-F16.gguf", "adapters": ["fpcvis"]},
    "standard": {"mmproj": "models/mmproj-Qwen3VL-4B-Instruct-F16.gguf", "adapters": ["fpc"], "base": True},
}
DRY_RETRY = {"dry_multiplier": 0.8, "cache_prompt": False}
LOCATE_Q = ('Locate every door in this floor plan. Answer with a JSON list, one entry per door: '
            '{"bbox_2d": [x1, y1, x2, y2], "label": "door"}, with coordinates from 0 to 1000 relative to '
            'the image width and height. Answer [] if there is no door.')
COUNT_Q = ("How many doors are shown in this floor plan? Count the TOTAL number of doors (all types).\n"
           "Give only the final answer, without explanation.")
_BOX = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def strip_reasoning(text):
    """Use only the final answer if the model wrapped it in tags or wrote reasoning first."""
    if text is None:
        return ""
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()


def boxes(s):
    """Every [x1, y1, x2, y2] in the text (bbox_2d entries of a JSON list, or loose)."""
    return [[float(v) for v in m.groups()] for m in _BOX.finditer(s)]


def lora_path(name):
    return f"models/qwen3vl4b_lora_{name}-F16.gguf"


def find_server():
    if os.environ.get("LLAMA_SERVER"):
        return os.environ["LLAMA_SERVER"]
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    hits = glob.glob(os.path.join(ROOT, "third_party", "llama.cpp", "**", name), recursive=True)
    return hits[0] if hits else None


def available():
    """Setups whose files exist, and model -> setup."""
    def ok(p):
        return os.path.exists(os.path.join(ROOT, p))
    setups = {k: v for k, v in SETUPS.items()
              if ok(BASE_GGUF) and ok(v["mmproj"]) and all(ok(lora_path(a)) for a in v["adapters"])}
    models = {}
    for name, s in setups.items():
        for a in s["adapters"]:
            models[a] = name
        if s.get("base"):
            models["base"] = name
    return setups, models


SERVER_BIN = find_server()
SETUPS_OK, MODELS = available()
EXAMPLES = sorted(f for f in os.listdir(os.path.join(ROOT, "examples")) if f.lower().endswith((".png", ".jpg")))
PID_FILE = os.path.join(ROOT, "logs", "llama-server.pid")
_server = {"proc": None, "setup": None}
_lock = threading.Lock()     # one request at a time: llama-server runs with -np 1, and switching must not overlap


def _stop():
    p = _server["proc"]
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    _server.update(proc=None, setup=None)


def _healthy():
    try:
        with urllib.request.urlopen(f"{LLAMA}/health", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def ensure(setup):
    """Make llama-server run `setup`; restart it if another setup is loaded."""
    p = _server["proc"]
    if _server["setup"] == setup and p and p.poll() is None:
        return
    _stop()
    if _healthy():
        raise RuntimeError(f"port {LLAMA_PORT} is already used by another llama-server; stop it first "
                           "or set LLAMA_PORT")
    s = SETUPS_OK[setup]
    cmd = [SERVER_BIN, "-m", BASE_GGUF, "--mmproj", s["mmproj"]]
    for a in s["adapters"]:
        cmd += ["--lora", lora_path(a)]
    cmd += ["--lora-init-without-apply", "--port", str(LLAMA_PORT), "-ngl", GPU_LAYERS, "-c", "4096", "-np", "1",
            "--image-max-tokens", IMAGE_MAX_TOKENS, "--temp", "0"]
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log = open(os.path.join(ROOT, "logs", "llama-server.log"), "a", encoding="utf-8")
    print(f"app: starting llama-server with the {setup} setup (log: logs/llama-server.log)", flush=True)
    _server.update(proc=subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), setup=setup)
    with open(PID_FILE, "w") as f:    # run.sh stops this process if the app is killed without cleaning up
        f.write(str(_server["proc"].pid))
    for _ in range(300):
        if _healthy():
            return
        if _server["proc"].poll() is not None:
            break
        time.sleep(1)
    _stop()
    raise RuntimeError("llama-server did not start; see logs/llama-server.log")


def _post(body):
    req = urllib.request.Request(f"{LLAMA}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"] or "", out["choices"][0].get("finish_reason")


def ask(image_data_url, question, model, max_tokens):
    setup = MODELS[model]
    with _lock:
        ensure(setup)
        adapters = SETUPS_OK[setup]["adapters"]
        lora = [{"id": i, "scale": 1.0 if name == model else 0.0} for i, name in enumerate(adapters)]
        body = {"model": "app", "temperature": 0, "max_tokens": max_tokens, "lora": lora,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": question}]}]}
        t = time.time()
        text, finish = _post(body)
        retried = None
        if finish == "length":   # stuck repeating: ask once more with the DRY sampler
            retried = {"first_boxes": len(boxes(strip_reasoning(text)))}
            text, finish = _post({**body, **DRY_RETRY})
    return {"answer": text, "boxes": boxes(strip_reasoning(text)), "seconds": round(time.time() - t, 1),
            "finish": finish, "retried": retried}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, data, ctype="application/json"):
        raw = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(os.path.dirname(__file__), "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if self.path == "/models":
            return self._send(200, {"models": list(MODELS), "default": next(iter(MODELS)),
                                    "setup": dict(MODELS), "loaded": _server["setup"], "examples": EXAMPLES})
        if self.path.startswith("/examples/") and self.path[len("/examples/"):] in EXAMPLES:
            name = self.path[len("/examples/"):]
            with open(os.path.join(ROOT, "examples", name), "rb") as f:
                return self._send(200, f.read(), "image/png" if name.lower().endswith(".png") else "image/jpeg")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/ask":
            return self._send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            mode = req.get("mode", "locate")
            question = {"locate": LOCATE_Q, "count": COUNT_Q}.get(mode) or req.get("question", "").strip()
            if not question:
                return self._send(400, {"error": "empty question"})
            model = req.get("model") or next(iter(MODELS))
            if model not in MODELS:
                return self._send(400, {"error": f"unknown model {model}"})
            res = ask(req["image"], question, model, 1500 if mode == "locate" else int(req.get("max_tokens", 512)))
            res["question"] = question
            self._send(200, res)
        except Exception as e:  # shown in the page
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write("app: " + fmt % args + "\n")


if __name__ == "__main__":
    if not SERVER_BIN or not os.path.exists(SERVER_BIN):
        sys.exit("llama-server not found in third_party/llama.cpp (run: bash setup.sh)")
    if not MODELS:
        sys.exit("model files missing in models/ (run: bash setup.sh)")
    atexit.register(_stop)
    first = next(iter(MODELS))
    ensure(MODELS[first])     # load the default model before opening the page
    print(f"\nopen http://localhost:{PORT}  (models: {', '.join(MODELS)}; Ctrl+C to stop)\n", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop()
