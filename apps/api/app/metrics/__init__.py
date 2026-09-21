"""In-memory metrics primitives.

Tiny, dependency-free Prometheus text-format emitter. Replace
:func:`set_backend` with a real Prometheus client at deploy time if
you need push-gateway or exemplars.

Three types:
- :class:`Counter` — monotonic, labelled
- :class:`Histogram` — fixed buckets, exposes ``_sum``, ``_count``, ``_bucket``
- :class:`Gauge` — arbitrary set

The render() function emits one line per series, including the
``HELP`` and ``TYPE`` headers that the Prometheus text format v0.0.4
spec requires.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


_lock = threading.Lock()
_metrics: Dict[str, "Metric"] = {}


def set_backend(backend: "MetricsBackend") -> None:
    """Swap the backend. Currently only the in-memory backend is
    supported, so this is a placeholder for the Prometheus client
    hookup."""
    # Phase 6: only the in-memory backend ships. ``set_backend`` is
    # a no-op for now; we keep the symbol so call sites don't have
    # to change when we wire prometheus_client later.
    return None


def reset() -> None:
    """Drop all metrics. Tests use this between scenarios."""
    with _lock:
        _metrics.clear()


def _get_or_create(name: str, kind: str) -> "Metric":
    with _lock:
        if name not in _metrics:
            if kind == "counter":
                _metrics[name] = Counter(name)
            elif kind == "histogram":
                _metrics[name] = Histogram(name)
            elif kind == "gauge":
                _metrics[name] = Gauge(name)
            else:
                raise ValueError(f"unknown metric kind: {kind!r}")
        return _metrics[name]


# ---- Counter ----------------------------------------------------------


class Counter:
    def __init__(self, name: str) -> None:
        self.name = name
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = defaultdict(float)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with _lock:
            self._values[key] += amount

    def render(self) -> List[str]:
        out = [
            f"# HELP {self.name} counter",
            f"# TYPE {self.name} counter",
        ]
        for key, value in sorted(self._values.items()):
            label_str = _format_labels(key)
            out.append(f"{self.name}{label_str} {value}")
        return out


def Counter(name: str) -> Counter:  # type: ignore[no-redef]
    return _get_or_create(name, "counter")  # type: ignore[return-value]


# ---- Histogram ----------------------------------------------------------


_BUCKETS: Tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, +float("inf"))


class Histogram:
    def __init__(self, name: str) -> None:
        self.name = name
        # key -> [bucket_counts (cumulative over the global _BUCKETS), sum, count]
        self._values: Dict[Tuple[Tuple[str, str], ...], List[float]] = {}

    def observe(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with _lock:
            if key not in self._values:
                # +1 trailing slot for the +Inf bucket and a final slot for _sum/_count.
                self._values[key] = [0.0] * (len(_BUCKETS) + 2)
            bucket = self._values[key]
            for i, b in enumerate(_BUCKETS):
                if value <= b:
                    bucket[i] += 1
            bucket[-2] += value  # sum
            bucket[-1] += 1  # count

    def render(self) -> List[str]:
        out = [
            f"# HELP {self.name} histogram (seconds)",
            f"# TYPE {self.name} histogram",
        ]
        for key, counts in sorted(self._values.items()):
            label_str_base = _format_labels(key)
            # Cumulative counts.
            for i, b in enumerate(_BUCKETS):
                if math.isinf(b):
                    le = "+Inf"
                else:
                    le = str(b)
                merged = dict(key)
                merged["le"] = le
                lbl = _format_labels(tuple(sorted(merged.items())))
                out.append(f"{self.name}_bucket{lbl} {counts[i]}")
            # Sum and count.
            lbl = _format_labels(key)
            out.append(f"{self.name}_sum{lbl} {counts[-2]}")
            out.append(f"{self.name}_count{lbl} {counts[-1]}")
        return out


def Histogram(name: str) -> Histogram:  # type: ignore[no-redef]
    return _get_or_create(name, "histogram")  # type: ignore[return-value]


# ---- Gauge --------------------------------------------------------------


class Gauge:
    def __init__(self, name: str) -> None:
        self.name = name
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}

    def set(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with _lock:
            self._values[key] = value

    def render(self) -> List[str]:
        out = [
            f"# HELP {self.name} gauge",
            f"# TYPE {self.name} gauge",
        ]
        for key, value in sorted(self._values.items()):
            lbl = _format_labels(key)
            out.append(f"{self.name}{lbl} {value}")
        return out


def Gauge(name: str) -> Gauge:  # type: ignore[no-redef]
    return _get_or_create(name, "gauge")  # type: ignore[return-value]


# ---- render --------------------------------------------------------------


def render() -> str:
    """Render every registered metric in Prometheus text format v0.0.4."""
    with _lock:
        snapshot = list(_metrics.values())
    lines: List[str] = []
    for metric in snapshot:
        lines.extend(metric.render())
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _format_labels(items: Iterable[Tuple[str, str]]) -> str:
    items = list(items)
    if not items:
        return ""
    parts = ",".join(f'{k}="{v}"' for k, v in items)
    return "{" + parts + "}"


import math  # used by Histogram (bucket parsing)

# public API together at the top of the file)
