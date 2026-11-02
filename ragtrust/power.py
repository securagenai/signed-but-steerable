"""System telemetry on the Raspberry Pi 5: PMIC power estimate (as in the prior study), CPU
utilisation (system and per process), SoC temperature, ARM clock and throttling flags.
The PMIC estimate covers the board's rails measured by the PMIC; it is not a substitute for an
external USB-C power meter. Log a meter alongside when available."""
import re, shutil, subprocess, threading, time

import numpy as np

try:
    import psutil
except ImportError:  # psutil is optional on the desktop
    psutil = None


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:
        return None


def pmic_watts():
    out = _run(["vcgencmd", "pmic_read_adc"]) if shutil.which("vcgencmd") else None
    if not out:
        return None
    cur, volt = {}, {}
    for n, v, u in re.findall(r"(\S+)\s+\w+\(\d+\)=([\d.]+)([AV])", out):
        (cur if u == "A" else volt)[n[:-2]] = float(v)
    w = sum(cur[k] * volt[k] for k in cur if k in volt)
    return w if w > 0 else None


def soc_temp_c():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0
    except Exception:
        return None


def arm_freq_mhz():
    try:
        return int(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read()) / 1000.0
    except Exception:
        return None


def throttled():
    return _run(["vcgencmd", "get_throttled"]) if shutil.which("vcgencmd") else None


class Telemetry(threading.Thread):
    """Background sampler. Use as a context manager; .window(a, b) summarises samples between two
    time.time() stamps. `pids` maps a label to a process id for per-process CPU percentages."""

    def __init__(self, period=0.5, pids=None):
        super().__init__(daemon=True)
        self.period, self.samples, self._run = period, [], True
        self.procs = {}
        if psutil:
            for k, pid in (pids or {}).items():
                try:
                    p = psutil.Process(pid)
                    p.cpu_percent(None)
                    self.procs[k] = p
                except Exception:
                    pass
            psutil.cpu_percent(None)

    def run(self):
        while self._run:
            s = {"t": time.time(), "w": pmic_watts(), "temp": soc_temp_c(), "mhz": arm_freq_mhz()}
            if psutil:
                s["cpu"] = psutil.cpu_percent(None)
                for k, p in self.procs.items():
                    try:
                        s[f"cpu_{k}"] = p.cpu_percent(None)
                    except Exception:
                        s[f"cpu_{k}"] = None
            self.samples.append(s)
            time.sleep(self.period)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self._run = False
        self.join(timeout=5)

    def window(self, a, b):
        S = [s for s in self.samples if a <= s["t"] <= b]
        if not S:
            return {}
        out = {}
        for k in S[0]:
            if k == "t":
                continue
            v = [s[k] for s in S if s.get(k) is not None]
            if v:
                out[f"{k}_mean"] = float(np.mean(v))
                out[f"{k}_max"] = float(np.max(v))
        out["n_samples"] = len(S)
        return out
