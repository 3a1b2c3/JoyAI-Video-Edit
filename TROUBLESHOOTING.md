# JoyAI-Video-Edit — known issues and fixes

Consolidated from WEBSOCKET_DEBUG.md / WEBSOCKET_FIX.md (now removed) plus
findings from later sessions. Update this file in place rather than
spinning up new scratch `.md` files per issue.

## 1. WebSocket unreachable: server blocked on synchronous model loading (RESOLVED)

**Symptom:** browser couldn't connect to `ws://<host>:8080/ws`; server
process was running (high CPU/memory, chunks visibly generating in logs)
but nothing was listening on port 8080 — `netstat`/`curl` both showed no
listener, and the "Application startup complete" line never printed.

**Root cause:** FastAPI's `lifespan` preload called `get_runtime()`
synchronously, loading DiT/VAE/text-encoder models (30-60s) *before*
uvicorn started accepting connections at all.

**Fix** (`deploy/xvideo/serving/serve_joyomni_streaming.py`, lines
~607-780): load models in a background thread via
`loop.run_in_executor(None, load_and_signal)` so uvicorn starts listening
immediately; the `/ws` handler does `await app.state.runtime_ready.wait()`
before processing anything, so a client connecting before models finish
loading just waits there instead of getting refused.

**Result:** server startup dropped from 60+s to ~2-3s; WebSocket available
immediately, first request waits for models if they're not ready yet,
subsequent requests are fast.

## 2. `"start"` message payload schema (working, exact format)

The client sends one JSON text message to kick off a session
(`websocket_endpoint`'s `elif msg_type == "start":` branch, line ~1338):

```json
{
  "type": "start",
  "prompt": "a person walking through a city street",
  "ref_image": "data:image/png;base64,<...>",
  "output_codec": "h264",
  "input_codec": "h264",
  "output_quality": 60,
  "gate_enabled": true
}
```

- `ref_image` is optional — a bare base64 string or a `data:...;base64,`
  URL both work (`_decode_ref_image`, line 119); omit or set `null` to
  start without a reference frame.
- Other optional fields the handler reads with sane defaults:
  `kv_reset_frames`, `max_temporal_ids`, `freeze_kv_on_static`,
  `static_diff_thresh`, `max_inflight_chunks`, `use_pe`, `source` (set to
  `"file"` for lossless mode).

**Verified server responses** on a successful handshake:
```
{'type': 'session_granted'}
{'type': 'started', 'frames_per_next_chunk': 1, 'height': 480, 'width': 840,
 'output_codec': 'h264', 'input_codec': 'h264', 'ref_image': true, ...}
```

**Local test client:** `test_websocket_client.py` (repo root) connects,
sends exactly this payload, and prints every message the server sends
back — a fast way to confirm the endpoint is healthy without a browser:

```bash
python test_websocket_client.py --host <bare-ip-no-scheme> --port 8080
```

**Common mistake:** `--host` takes a bare IP/hostname, not a URL. Passing
`--host http://10.x.x.x/` builds a malformed
`ws://http://10.x.x.x/:8080/ws` and fails DNS resolution
(`getaddrinfo failed`) before ever reaching the network. No scheme, no
trailing slash — just `10.x.x.x`.

## 3. Multiple remote hosts in play — verify which one before testing

This project has been run/tested across at least three different remote
identities in past sessions: `10.57.233.141` (older WEBSOCKET_FIX.md
notes), `pmgb300ws-0304` / `10.74.11.118` (a *different* project's
machine — flashdreams/LingBot — easy to confuse since both were worked on
in the same day), and `kschmid-4vvboh` ("horde"). **Always confirm the
current host with `hostname -I` on the box actually running the server**
before pointing a client at an IP from memory — testing against the wrong
machine's IP produces `ConnectionRefusedError`, which looks identical to
a real server-down problem.

## 4. Server bound to `127.0.0.1` — unreachable except from itself

**Symptom:** the websocket test (or browser) works when run *on* the
server's own host against `127.0.0.1`, but any other machine gets
`ConnectionRefusedError` even with the correct IP.

**Cause:** `serve_joyomni_streaming.py --host` defaults to `127.0.0.1`
(line ~2004). `deploy/run_server.sh` overrides this correctly
(`HOST="${JOYOMNI_HOST:-0.0.0.0}"`, line 55, passed as `--host "$HOST"`)
— but only if the server was actually launched *through* that script (or
`run_server_best.sh`/`run_server_fp4.sh`, which both chain into it). If
something invoked `serve_joyomni_streaming.py` directly, or an env var
leaked `JOYOMNI_HOST=127.0.0.1` in, the override never applies.

**Fix:** always launch via `bash run_server_fp4.sh` (or
`run_server_best.sh`), never the raw Python entrypoint directly. Confirm
the actual bind address before testing remotely:

```bash
ss -ltnp 2>/dev/null | grep 8080   # look for 0.0.0.0:8080 or *:8080, not 127.0.0.1:8080
```

## 5. `run_server_fp4.sh` hardcoded low-VRAM mode on any GPU (FIXED)

**Symptom:** on a 95GB card with only ~41GB actually in use (no VRAM
pressure at all), the server still paid the low-VRAM tax: block-by-block
DiT staging + text-encoder CPU offload (added H2D transfer latency per
prompt encode) — pure downside, no benefit.

**Cause:** `run_server_fp4.sh` had `export JOYOMNI_LOW_VRAM=1` hardcoded
unconditionally, which shadowed `run_server_best.sh`'s own GPU
auto-detection (`run_server_best.sh` already picks `JOYOMNI_LOW_VRAM=0`
on >48 GiB cards, `=1` on ≤48 GiB, per `DEPLOYMENT.md` §4) — the
auto-detect logic never got a chance to run because the variable was
already set before `run_server_best.sh` was even invoked.

**Fix:** removed the hardcoded export from `run_server_fp4.sh`; GPU
auto-detection in `run_server_best.sh` now actually applies. Still
overridable by exporting `JOYOMNI_LOW_VRAM` yourself before running
either script.

## 6. `ltx_causal` import error in Echo-WM Gradio UI (separate JoyAI-Echo repo)

Not this repo — `~/JoyAI-Echo/echo_wm/gradio_echo_wm.py` failing with
`ModuleNotFoundError: No module named 'ltx_causal'`. The package lives at
`echo_wm/ltx-causal/src/ltx_causal/` (`src`-layout) but isn't wired into
any setup script's install step. Fix:

```bash
cd ~/JoyAI-Echo/echo_wm && source .venv/bin/activate
pip install -e ltx-causal --no-deps
```

## 7. GPU startup warmup cost (see also flashdream_public's LingBot notes)

Not specific to this repo, but the same lesson applies here: the first
generation(s) after server startup pay a one-time CUDA kernel
JIT/cuDNN-autotune cost that's much higher than steady-state — plan
around it (warm up before exposing the UI to real users, or accept the
first request being slow) rather than assuming steady-state timing
applies to the first request.

## 8. `no chunk in UI`: lossless/file-source frames silently dropped (RESOLVED)

**Symptom:** server logs show healthy generation (`#####[STREAM] chunk=N
in_frames=8 out_frames=8 elapsed=0.5s ...` repeating normally, `chunk_done`
firing in the browser console) but nothing ever renders in the UI, and the
browser console never shows a single `[WS] Binary frame: ...` line.

**How it was found:** the `/debug` HTTP endpoint
(`serve_joyomni_streaming.py`'s `@app.get("/debug")`) exposes
`ws_debug.frames_out`/`ws_debug.output_bytes`. Checking it mid-session
showed `frames_out: 186` (climbing normally) alongside `output_bytes: 0`
(flat, never moved) — proof frames were being counted as "sent" without a
single byte actually leaving the server. That pointed straight at
`_send_encoded_frames`'s per-frame loop instead of the more commonly
suspected face/person-presence gate (`gate_state.absent_hold` — checked
first via a temporary `/debug/gate` endpoint + loud log, both since
removed once ruled out; see `9adde56`/`2754066` history for that dead end
if it resurfaces).

**Root cause:** in lossless/file-source mode
(`payload.source == "file"` client-side -> `lossless_mode = True`
server-side), `joyomni_streaming.py`'s frame-encode step
(`get_output_frames`) intentionally returns **raw numpy frames** instead
of JPEG bytes, to preserve full quality for disk recording. But
`_send_encoded_frames` in `serve_joyomni_streaming.py` treated any
non-`bytes` frame as "nothing to send" and `continue`'d straight past the
websocket send, only ever submitting to `rec_output` (disk). Every
lossless-mode frame got silently eaten before it could reach the browser.

**Fix** (`serve_joyomni_streaming.py`, the `elif not isinstance(encoded,
bytes):` branch): JPEG-encode a *preview* copy of the raw frame with
`cv2.imencode` (same pattern used elsewhere in this file) and fall through
into the existing shared send path instead of `continue`-ing past it. Disk
recording is untouched — the shared success path already separately
submits the original raw `encoded` frame to `rec_output`, so recording
keeps full lossless quality completely independent of what gets
JPEG-compressed for the live preview.

**Lesson:** when "the counter says N happened but the effect never shows
up," check whether the counter increments on a code path that never
actually performs the visible side effect — `frames_out += 1` living
right next to a `continue` that skips the send is exactly that trap.

## 9. `joyomni_ops` build/venv mismatches (multiple issues, watch for both)

Two distinct, easily-conflated problems hit together on `pmgb300ws-0304`:

**(a) Wrong venv active.** `deploy/run_server.sh` and its wrapper scripts
(`run_server_fp4.sh`, `run_server_fp8.sh`, `run_server_best.sh`) do **not**
activate a `.venv` themselves — they only handle `conda activate` and only
if `JOYOMNI_CONDA_ENV` is set. They rely entirely on whatever Python venv
is already active in the calling shell. Since venv activation is pure
shell state (unaffected by `cd`), it's easy to have a *different* project's
venv (e.g. `~/JoyAI-Echo/.venv`) still active from earlier in the session
and not notice — the traceback will show `torch`/other packages loading
from the wrong repo's `site-packages` path. Always verify before running
any `run_server_*.sh`:
```bash
which python && python -c "import torch; print(torch.__file__)"
```

**(b) Stale/wrong-target `joyomni_ops` build.** Even with the correct venv
active, `deploy/joyomni_ops/build.log` (if present) records exactly what
the *last* build was compiled for — check it before assuming a "missing
symbol" `ImportError` (e.g. `cannot import name 'fused_norm_scale_shift'`)
is a code bug. One observed instance: the log showed the extension had
been built via **WSL2 on the local Windows machine**, targeting an **RTX
5090 (compute capability 12.0)**, at an old git commit -- completely
wrong-target for a GB300 datacenter card (different compute capability
entirely), and possibly from before that symbol existed in source at all.
`pip install --no-build-isolation ./deploy/joyomni_ops` (DEPLOYMENT.md's
documented command) must be re-run **on the actual target machine**, with
the correct venv active, against current source, before the extension can
be trusted to match the GPU you're actually running on.

Use `bash build_joyomni_ops.sh` (repo root) rather than the raw `pip install`
command — it wraps that exact command with the checks above (venv, torch/GPU
compute_cap printed before building) plus a post-build import/symbol check,
so a wrong-target or stale build fails loudly here instead of surfacing later
as a confusing runtime error. `JOYOMNI_OPS_CUTLASS_DIR` (default
`deploy/tmp/cutlass`) still selects the cutlass checkout for full FP8 kernel
support; `JOYOMNI_OPS_NO_FP8=1` builds the light variant instead. See
DEPLOYMENT.md's GB300/GB200 section for a case where the cutlass checkout
itself needs to be at a specific pinned commit or a required header goes
missing.
