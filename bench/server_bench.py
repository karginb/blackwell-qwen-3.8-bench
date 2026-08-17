#!/usr/bin/env python3
"""
Benchmark llama-server configurations with tight error bars and thermal telemetry.

WHY NOT llama-cli
Driving llama-cli and scraping its terminal output has three problems we hit in
practice:

  1. No repetitions -> single-shot numbers with roughly 10% run-to-run noise,
     enough to make an ngl sweep come out non-monotonic (physically impossible).
  2. No thermal visibility -> on a laptop, a long benchmark session quietly
     throttles and later configurations look worse than earlier ones purely
     because the GPU got hot.
  3. No speculative-decoding telemetry -> llama-cli runs the model in an internal
     server process whose stats never reach stdout, so draft acceptance (the one
     number that decides whether speculative decoding is winning) is invisible.

llama-server solves all three: every /completion response carries a `timings`
object with exact token counts and, when drafting is active, `draft_n` and
`draft_n_accepted`.

WHAT THIS SCRIPT DOES
For each configuration, in isolation:
  start server -> wait for health -> warmup request -> N measured requests
  -> stop server -> cool down -> next configuration

It reports the median (robust against a single slow outlier), the observed
spread, and the GPU clock/temperature range each configuration ran at, so a
throttled result is visible instead of silently polluting the comparison.

USAGE
    python3 bench/server_bench.py \\
      --model models/Qwen3.8-27B-UD-IQ2_XXS.gguf \\
      --reps 5 --cooldown 30 \\
      --config "baseline:-ngl 50 -fa on" \\
      --config "mtp-2:-ngl 50 -fa on --spec-type draft-mtp --spec-draft-n-max 2"
"""

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_PROMPT = (
    "Write a Python function that performs binary search on a sorted list, "
    "then explain step by step how it works, why its complexity is O(log n), "
    "and what happens when the target value is absent from the list."
)


# --------------------------------------------------------------------------
# GPU telemetry
# --------------------------------------------------------------------------

def gpu_sample():
    """Read SM clock (MHz), temperature (C), power draw (W) and used VRAM (MiB).

    VRAM usage matters as much as speed here: configurations do not cost the same
    amount of memory, so "tokens per second" is only half of the comparison. A
    config that is 5% faster while eating 1 GB more VRAM is not obviously better.

    Returns None when nvidia-smi is unavailable, so the script still runs on
    machines without an NVIDIA GPU.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,temperature.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().split("\n")[0]
        clock, temp, power, mem = (p.strip() for p in out.split(","))
        return {"clock_mhz": int(float(clock)),
                "temp_c": int(float(temp)),
                "power_w": float(power),
                "vram_mib": int(float(mem))}
    except Exception:
        return None


# --------------------------------------------------------------------------
# Server lifecycle
# --------------------------------------------------------------------------

class Server:
    """Runs one llama-server process for the lifetime of a single configuration."""

    def __init__(self, binary, model, flags, port, log_path):
        tokens = shlex.split(flags)
        # A config may name its own model, which lets one run compare different
        # quantizations of the same model under identical thermal conditions --
        # the only way to make a quality/speed trade-off comparison honest.
        own_model = any(t in ("-m", "--model") for t in tokens)
        base = [binary] if own_model else [binary, "-m", model]
        self.cmd = [*base, "--host", "127.0.0.1", "--port", str(port), *tokens]
        self.port = port
        self.log_path = log_path
        self.proc = None

    def __enter__(self):
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.log.close()

    def wait_ready(self, timeout=600):
        """Poll /health until the model is loaded. Returns elapsed seconds, or None."""
        url = f"http://127.0.0.1:{self.port}/health"
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            if self.proc.poll() is not None:
                return None  # process died during load, most likely OOM
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if json.load(r).get("status") == "ok":
                        return time.perf_counter() - start
            except Exception:
                pass
            time.sleep(1.0)
        return None

    def failure_reason(self):
        """Extract the most useful line from the server log after a failed start."""
        try:
            with open(self.log_path, errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            return "server log unreadable"
        for needle in ("out of memory", "failed to allocate",
                       "failed to create context", "failed to load model"):
            for line in lines:
                if needle in line.lower():
                    return line.strip()[:160]
        return lines[-1].strip()[:160] if lines else "server exited without output"

    def completion(self, prompt, n_predict, timeout=900):
        """Send one /completion request and return its timings object."""
        payload = json.dumps({
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,     # deterministic: every rep does identical work
            "cache_prompt": False,  # no prompt-cache reuse between reps
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion",
            data=payload, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r).get("timings", {})


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

@dataclass
class Result:
    name: str
    flags: str
    ok: bool = False
    error: str = ""
    load_s: float = 0.0
    discard: int = 0
    decode_tps: list = field(default_factory=list)
    prefill_tps: list = field(default_factory=list)
    draft_n: int = 0
    draft_accepted: int = 0
    clocks: list = field(default_factory=list)
    temps: list = field(default_factory=list)
    vram: list = field(default_factory=list)

    # A laptop GPU answers the first request on boost clocks and then settles into
    # a lower sustained state once it hits its thermal/power limit. Only the settled
    # number describes what a user actually gets from a long generation, so the
    # headline statistics are computed after dropping the first `discard` reps.

    @property
    def steady(self):
        return self.decode_tps[self.discard:] or self.decode_tps

    @property
    def decode_median(self):
        return statistics.median(self.steady) if self.steady else 0.0

    @property
    def decode_peak(self):
        """Best single rep, normally the first one on boost clocks."""
        return max(self.decode_tps) if self.decode_tps else 0.0

    @property
    def boost_drop_pct(self):
        """How much throughput was lost between the peak and the settled state."""
        if not self.decode_tps or not self.decode_median:
            return 0.0
        return (self.decode_peak / self.decode_median - 1) * 100

    @property
    def prefill_median(self):
        vals = self.prefill_tps[self.discard:] or self.prefill_tps
        return statistics.median(vals) if vals else 0.0

    @property
    def spread_pct(self):
        """Half-range as a percentage of the median -- how much to distrust the number."""
        vals = self.steady
        if len(vals) < 2:
            return 0.0
        return (max(vals) - min(vals)) / 2 / self.decode_median * 100

    @property
    def acceptance(self):
        """Fraction of drafted tokens the target model accepted. None if not drafting."""
        return self.draft_accepted / self.draft_n if self.draft_n else None

    @property
    def vram_peak(self):
        return max(self.vram) if self.vram else 0


def check_flags(flags):
    """Reject flag combinations that llama.cpp silently degrades.

    Passing -ngl together with --fit-target makes the fitter abort (see
    common/fit.cpp: "n_gpu_layers already set by user"), leaving the explicit
    -ngl in charge. The run still succeeds, so the mistake is easy to miss --
    you just quietly benchmark something other than what you meant to.
    """
    tokens = shlex.split(flags)
    has_ngl = any(t in ("-ngl", "--gpu-layers", "--n-gpu-layers") for t in tokens)
    has_fit = any(t in ("-fitt", "--fit-target") for t in tokens)
    if has_ngl and has_fit:
        return ("-ngl and --fit-target are mutually exclusive: the fitter aborts "
                "when n_gpu_layers is set explicitly")
    return None


def run_config(name, flags, args):
    result = Result(name=name, flags=flags, discard=args.discard)
    log_path = os.path.join(args.raw_dir, f"server_{name}.log")

    print(f"\n=== {name} ===")
    print(f"    flags: {flags}")

    conflict = check_flags(flags)
    if conflict:
        result.error = conflict
        print(f"    SKIPPED: {conflict}")
        return result

    with Server(args.binary, args.model, flags, args.port, log_path) as server:
        load_s = server.wait_ready(timeout=args.load_timeout)
        if load_s is None:
            result.error = server.failure_reason()
            print(f"    FAILED: {result.error}")
            return result

        result.load_s = load_s
        print(f"    server ready in {load_s:.1f}s")

        try:
            server.completion(args.prompt, args.warmup_tokens)  # warmup, discarded
        except Exception as e:
            result.error = f"warmup request failed: {e}"
            print(f"    FAILED: {result.error}")
            return result

        for i in range(args.reps):
            try:
                t = server.completion(args.prompt, args.tokens)
            except Exception as e:
                result.error = f"request {i + 1} failed: {e}"
                print(f"    FAILED: {result.error}")
                return result

            decode = t.get("predicted_per_second", 0.0)
            prefill = t.get("prompt_per_second", 0.0)
            result.decode_tps.append(decode)
            result.prefill_tps.append(prefill)
            result.draft_n += int(t.get("draft_n", 0) or 0)
            result.draft_accepted += int(t.get("draft_n_accepted", 0) or 0)

            gpu = gpu_sample()
            note = ""
            if gpu:
                result.clocks.append(gpu["clock_mhz"])
                result.temps.append(gpu["temp_c"])
                result.vram.append(gpu["vram_mib"])
                note = (f"   [{gpu['clock_mhz']} MHz, {gpu['temp_c']}C, "
                        f"{gpu['power_w']:.0f}W, {gpu['vram_mib']} MiB]")
            tag = " (warm-up, discarded)" if i < args.discard else ""
            print(f"    rep {i + 1}/{args.reps}: {decode:6.2f} tok/s{note}{tag}")

    result.ok = True
    acc = result.acceptance
    acc_str = f", acceptance {acc * 100:.1f}%" if acc is not None else ""
    print(f"    steady {result.decode_median:.2f} tok/s "
          f"(spread +/-{result.spread_pct:.1f}%){acc_str}")
    if result.boost_drop_pct > 3.0:
        print(f"    peak was {result.decode_peak:.2f} tok/s on boost clocks "
              f"({result.boost_drop_pct:.0f}% above the settled state)")
    return result


def cooldown(seconds, target_temp):
    """Idle until the GPU cools down, so the next config starts from the same state.

    Without this, a long sweep measures thermal decay as much as it measures the
    configurations themselves.
    """
    if seconds <= 0:
        return
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        gpu = gpu_sample()
        if gpu and target_temp and gpu["temp_c"] <= target_temp:
            print(f"    cooled to {gpu['temp_c']}C after "
                  f"{time.perf_counter() - start:.0f}s")
            return
        time.sleep(2.0)
    gpu = gpu_sample()
    print(f"    cooldown {seconds}s done" + (f" ({gpu['temp_c']}C)" if gpu else ""))


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_table(results):
    print("\n" + "=" * 96)
    print("SUMMARY  (decode = steady state, after discarding boost-clock reps)")
    print("=" * 96)
    header = (f"{'config':<16}{'decode t/s':>12}{'spread':>9}{'prefill t/s':>13}"
              f"{'accept':>9}{'VRAM MiB':>10}{'peak C':>8}{'min MHz':>9}")
    print(header)
    print("-" * len(header))

    baseline = next((r for r in results if r.ok), None)

    for r in results:
        if not r.ok:
            print(f"{r.name:<16}  FAILED: {r.error[:70]}")
            continue
        acc = r.acceptance
        acc_str = f"{acc * 100:.1f}%" if acc is not None else "-"
        temp = f"{max(r.temps)}" if r.temps else "-"
        mhz = f"{min(r.clocks)}" if r.clocks else "-"
        vram = f"{r.vram_peak}" if r.vram else "-"
        print(f"{r.name:<16}{r.decode_median:>12.2f}{r.spread_pct:>8.1f}%"
              f"{r.prefill_median:>13.1f}{acc_str:>9}{vram:>10}{temp:>8}{mhz:>9}")

    if baseline and len(results) > 1:
        print("-" * len(header))
        print(f"relative to '{baseline.name}' ({baseline.decode_median:.2f} tok/s, "
              f"{baseline.vram_peak} MiB):")
        for r in results:
            if not r.ok or r is baseline:
                continue
            delta = (r.decode_median / baseline.decode_median - 1) * 100
            dv = r.vram_peak - baseline.vram_peak
            print(f"  {r.name:<16}{delta:+7.1f}% speed   {dv:+6d} MiB VRAM")

    # A config whose spread exceeds its advantage has not actually proven anything.
    noisy = [r for r in results if r.ok and r.spread_pct > 5.0]
    if noisy:
        print("\nWARNING: high run-to-run spread in: "
              + ", ".join(f"{r.name} (+/-{r.spread_pct:.1f}%)" for r in noisy))
        print("         Raise --reps or --cooldown before trusting these numbers.")

    throttled = [r for r in results
                 if r.ok and r.clocks and max(r.clocks) - min(r.clocks) > 200]
    if throttled:
        print("\nWARNING: clock drop over 200 MHz during: "
              + ", ".join(r.name for r in throttled))
        print("         Thermal throttling is confounding these results.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="default GGUF model; a config may override it "
                                   "by naming its own -m")
    p.add_argument("--binary", default="./llama.cpp/build/bin/llama-server")
    p.add_argument("--config", action="append", required=True, metavar="NAME:FLAGS",
                   help="a configuration to benchmark; repeatable")
    p.add_argument("--reps", type=int, default=6, help="requests per config")
    p.add_argument("--discard", type=int, default=2,
                   help="reps to drop before computing statistics; the GPU answers "
                        "the first requests on boost clocks and only then settles "
                        "into the sustained state that reflects real usage")
    p.add_argument("--tokens", type=int, default=256, help="tokens to generate per request")
    p.add_argument("--warmup-tokens", type=int, default=128,
                   help="tokens for the unmeasured warm-up request")
    p.add_argument("--cooldown", type=int, default=30, help="seconds between configs")
    p.add_argument("--cool-to", type=int, default=0,
                   help="end cooldown early once GPU reaches this temperature (C)")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--load-timeout", type=int, default=600)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--out", default="bench/results/server_bench.json")
    p.add_argument("--raw-dir", default="bench/results/raw")
    args = p.parse_args()

    if args.model and not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}")
    if not os.path.exists(args.binary):
        sys.exit(f"llama-server not found: {args.binary}")
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    configs = []
    for spec in args.config:
        if ":" not in spec:
            sys.exit(f"--config must be NAME:FLAGS, got: {spec}")
        name, flags = spec.split(":", 1)
        configs.append((name.strip(), flags.strip()))

    if args.discard >= args.reps:
        sys.exit(f"--discard ({args.discard}) must be smaller than --reps ({args.reps})")

    gpu = gpu_sample()
    print(f"model  : {args.model}")
    print(f"configs: {len(configs)}   reps: {args.reps} "
          f"(first {args.discard} discarded)   tokens/request: {args.tokens}")
    if gpu:
        print(f"gpu    : {gpu['clock_mhz']} MHz, {gpu['temp_c']}C, "
              f"{gpu['vram_mib']} MiB in use at start")

    results = []
    for i, (name, flags) in enumerate(configs):
        results.append(run_config(name, flags, args))
        if i < len(configs) - 1:
            cooldown(args.cooldown, args.cool_to)

    print_table(results)

    with open(args.out, "w") as f:
        json.dump({
            "model": args.model,
            "reps": args.reps,
            "discard": args.discard,
            "tokens": args.tokens,
            "prompt": args.prompt,
            "results": [
                {"name": r.name, "flags": r.flags, "ok": r.ok, "error": r.error,
                 "load_s": round(r.load_s, 1),
                 "decode_tps": r.decode_tps, "prefill_tps": r.prefill_tps,
                 "decode_median": round(r.decode_median, 3),
                 "decode_peak": round(r.decode_peak, 3),
                 "boost_drop_pct": round(r.boost_drop_pct, 2),
                 "spread_pct": round(r.spread_pct, 2),
                 "draft_n": r.draft_n, "draft_accepted": r.draft_accepted,
                 "acceptance": r.acceptance,
                 "clocks": r.clocks, "temps": r.temps, "vram_mib": r.vram}
                for r in results
            ],
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
