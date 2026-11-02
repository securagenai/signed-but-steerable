"""llama.cpp server helpers. The chat request reproduces the prior study exactly (same system prompt,
max_tokens=64, temperature 0, seed 42), so new configurations are comparable with the released runs.
stream_completion() timestamps every streamed token for the contention experiment (E4)."""
import json, subprocess, time

import requests

SYSTEM_PROMPT = ("Answer the question using ONLY the numbered context passages. "
                 "Reply with a short answer phrase and the passage number in brackets. "
                 "If the context does not contain the answer, reply exactly: UNANSWERABLE")


def build_messages(question, blocks):
    ctx = "\n\n".join(f"[{j + 1}] {b}" for j, b in enumerate(blocks))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}]


def chat(server, messages, max_tokens=64, timeout=1800):
    t0 = time.perf_counter()
    r = requests.post(f"{server}/v1/chat/completions",
                      json={"messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 42},
                      timeout=timeout).json()
    ms = (time.perf_counter() - t0) * 1e3
    ans = r["choices"][0]["message"]["content"].strip()
    return ans, r.get("timings", {}), r.get("usage", {}), ms


def wait_health(server, tries=300, delay=2.0):
    for _ in range(tries):
        try:
            if requests.get(f"{server}/health", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(delay)
    return False


def start_server(binary, model, threads, port=8080, ctx=4096, log_path=None, cpus=None, extra=()):
    """Starts llama-server with one slot (as in the prior study). `cpus` (e.g. '0-2') pins it with taskset."""
    cmd = [str(binary), "-m", str(model), "-t", str(threads), "-c", str(ctx), "-np", "1",
           "--host", "127.0.0.1", "--port", str(port), *extra]
    if cpus:
        cmd = ["taskset", "-c", cpus] + cmd
    log = open(log_path, "w") if log_path else subprocess.DEVNULL
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    if not wait_health(f"http://127.0.0.1:{port}"):
        p.kill()
        raise RuntimeError(f"llama-server did not become healthy: {' '.join(cmd)}")
    return p


def stop_server(p, settle_s=5):
    p.terminate()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
    time.sleep(settle_s)


def stream_completion(server, prompt, n_predict=128, timeout=1800):
    """Streams /completion with a fixed number of generated tokens (ignore_eos) and no prompt cache,
    returning time to first token, inter-token intervals and the server's own timings."""
    payload = {"prompt": prompt, "n_predict": n_predict, "ignore_eos": True, "cache_prompt": False,
               "temperature": 0, "seed": 42, "stream": True}
    t0 = time.perf_counter()
    stamps, timings, text = [], {}, []
    with requests.post(f"{server}/completion", json=payload, stream=True, timeout=timeout) as r:
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            ev = json.loads(raw[6:])
            if ev.get("content"):
                stamps.append(time.perf_counter())
                text.append(ev["content"])
            if ev.get("stop"):
                timings = ev.get("timings", {})
                break
    if not stamps:
        return {"ok": False}
    itl = [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]
    return {"ok": True, "ttft_ms": (stamps[0] - t0) * 1e3, "itl_ms": itl, "n_tokens": len(stamps),
            "total_ms": (stamps[-1] - t0) * 1e3, "timings": timings, "chars": len("".join(text))}
