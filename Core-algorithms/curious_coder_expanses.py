#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curious Coder — Advanced Python Concepts (Explained by Code)
============================================================
Run:
  python curious_coder_expanses.py                 # interactive menu
  python curious_coder_expanses.py --demo all      # run all demos non-interactively
  python curious_coder_expanses.py --demo async    # run a specific section

Sections
--------
1) Closures & Decorators (with timing + memoization)
2) Generators & Pipelines (lazy processing + streaming)
3) Context Managers (class-based and contextlib)
4) Descriptors & Data Validation (clean domain models)
5) Metaclasses (enforce class contracts at import-time)
6) Structural Pattern Matching (PEP 634)
7) Asyncio (tasks, gather, backoff, cancellation)
8) Concurrency in Practice (threads vs processes, GIL, CPU vs I/O)
9) Type Hints & Protocols (static duck typing, mypy-friendly)
10) Caching & LRU (functools + manual TTL cache)
11) Testing by Example (doctest in docstrings)
12) Optional: Vectorization Taste (NumPy fallback demo)

Notes
-----
- Each demo prints a clear explanation + the output so it's teachable in class.
- No internet calls; async demos simulate I/O with asyncio.sleep.
- Safe to run multiple times; no external files created.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import functools
import inspect
import math
import os
import random
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Protocol, Tuple

#  Pretty printing helpers 

def banner(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)

def explain(s: str) -> None:
    print(textwrap.dedent(s).strip() + "\n")

#  1) Closures & Decorators: timing + memoization

def timed(fn: Callable) -> Callable:
    """Decorator: prints runtime. Demonstrates closure capturing 'fn' & inner wrapper.
    >>> @timed
    ... def slow(): 
    ...     _=sum(range(10000)); return 42
    ...
    ... # doctest: +ELLIPSIS
    ... slow()
    42
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dur = (time.perf_counter() - start) * 1000
            print(f"[timed] {fn.__name__} ran in {dur:.2f} ms")
    return wrapper

def memoize(fn: Callable) -> Callable:
    """Simple memoization decorator using a closure over 'cache' dict."""
    cache: Dict[Tuple[Any, ...], Any] = {}
    @functools.wraps(fn)
    def wrapped(*args):
        if args in cache:
            return cache[args]
        out = fn(*args)
        cache[args] = out
        return out
    return wrapped

@timed
def fib_demo(n: int = 30) -> int:
    @memoize
    def fib(k: int) -> int:
        return k if k < 2 else fib(k-1) + fib(k-2)
    return fib(n)

# 2) Generators & Pipelines: lazy streaming

def numbers(start: int = 1) -> Iterator[int]:
    """Infinite generator of natural numbers."""
    i = start
    while True:
        yield i
        i += 1

def take(n: int, it: Iterable[Any]) -> List[Any]:
    out = []
    for _ in range(n):
        out.append(next(it))
    return out

def gen_pipeline_demo() -> List[int]:
    """Pipeline: (numbers -> squares -> evens -> take)
    Shows memory-constant streaming.
    """
    def squares(seq: Iterable[int]) -> Iterator[int]:
        for x in seq:
            yield x * x
    def evens(seq: Iterable[int]) -> Iterator[int]:
        for x in seq:
            if x % 2 == 0:
                yield x
    return take(8, evens(squares(numbers(1))))

# 3) Context Managers: class-based & contextlib 

class timeblock:
    """Class-based context manager: measures code block time.
    with timeblock("phase"):
        ...work...
    """
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = (time.perf_counter() - self.t0) * 1000
        print(f"[timeblock] {self.name}: {dt:.2f} ms")
        # swallow nothing; propagate exceptions
        return False

@contextlib.contextmanager
def suppress_and_report(*exceptions):
    """contextlib-based: selectively suppress, with notice."""
    try:
        yield
    except exceptions as e:
        print(f"[suppress] suppressed: {type(e).__name__}: {e}")

# 4) Descriptors & Data Validation 

class NonEmptyStr:
    def __set_name__(self, owner, name):
        self.private = "_" + name
    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private)
    def __set__(self, obj, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        setattr(obj, self.private, value)

class Percent:
    def __set_name__(self, owner, name):
        self.private = "_" + name
    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private)
    def __set__(self, obj, value):
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError("percent must be in [0,1]")
        setattr(obj, self.private, float(value))

class Metric:
    name = NonEmptyStr()
    weight = Percent()
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight

# 5) Metaclasses: enforce class contracts 

class RequireEvaluate(type):
    """Metaclass that ensures subclasses define 'evaluate(self, x)'."""
    def __new__(mcls, name, bases, ns):
        cls = super().__new__(mcls, name, bases, ns)
        if name != "BaseModel" and "evaluate" not in ns:
            raise TypeError(f"{name} must implement 'evaluate(self, x)'")
        return cls

class BaseModel(metaclass=RequireEvaluate):
    pass

class LogisticModel(BaseModel):
    def __init__(self, w: float, b: float):
        self.w, self.b = w, b
    def evaluate(self, x: float) -> float:
        return 1 / (1 + math.exp(-(self.w * x + self.b)))

#  6) Structural Pattern Matching (PEP 634)

def route(event: dict) -> str:
    """Match on dict structure to route events."""
    match event:
        case {"type": "click", "x": int(x), "y": int(y)} if x >= 0 and y >= 0:
            return f"Click at ({x},{y})"
        case {"type": "key", "key": str(k)}:
            return f"Keypress '{k}'"
        case {"type": "resize", "w": int(w), "h": int(h)}:
            return f"Resize to {w}x{h}"
        case _:
            return "Unknown event"

#  7) Asyncio: tasks, gather, backoff, cancellation

async def fake_io(name: str, secs: float) -> str:
    await asyncio.sleep(secs)
    return f"{name} done in {secs:.2f}s"

async def with_backoff(task_name: str, attempts: int = 3) -> str:
    for i in range(attempts):
        try:
            # simulate flaky: 50% failure until last try
            if i < attempts - 1 and random.random() < 0.5:
                raise RuntimeError("transient")
            return await fake_io(task_name, 0.1 + 0.05 * i)
        except RuntimeError:
            await asyncio.sleep(0.05 * (2**i))
    return f"{task_name} failed after retries"

async def async_demo() -> List[str]:
    tasks = [with_backoff(f"T{i}") for i in range(3)]
    done = await asyncio.gather(*tasks, return_exceptions=False)
    return done

# 8) Concurrency: threads vs processes 

def cpu_bound(n: int) -> int:
    # intentionally heavy-ish: sum of primes up to n (simple check)
    def is_prime(x: int) -> bool:
        if x < 2: 
            return False
        r = int(x**0.5)
        for k in range(2, r+1):
            if x % k == 0:
                return False
        return True
    return sum(i for i in range(n) if is_prime(i))

def io_bound(secs: float) -> str:
    time.sleep(secs); return f"slept {secs}s"

def concurrency_demo() -> Dict[str, float]:
    out: Dict[str, float] = {}
    # CPU — compare threads vs processes
    N = 10000  # small to keep runtime snappy
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(cpu_bound, [N]*4))
    out["cpu_threads_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(cpu_bound, [N]*4))
    out["cpu_processes_ms"] = (time.perf_counter() - t0) * 1000

    # I/O — threads shine
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(io_bound, [0.2, 0.2, 0.2, 0.2]))
    out["io_threads_ms"] = (time.perf_counter() - t0) * 1000
    return out

#  9) Type Hints & Protocols 

class SupportsLen(Protocol):
    def __len__(self) -> int: ...

def length_okay(x: SupportsLen, min_len: int = 1) -> bool:
    """Static duck typing: any object with __len__ qualifies."""
    return len(x) >= min_len

# 10) Caching: LRU + TTL 

@functools.lru_cache(maxsize=128)
def heavy_convert(x: int) -> int:
    # pretend slow transform
    time.sleep(0.01)
    return x * x * x

def ttl_cache_demo(ttl_seconds: float = 0.2) -> None:
    """Manual TTL cache example with closure state."""
    cache: Dict[int, Tuple[float, Any]] = {}
    def convert(x: int) -> int:
        now = time.time()
        if x in cache and (now - cache[x][0]) < ttl_seconds:
            return cache[x][1]
        val = heavy_convert(x)
        cache[x] = (now, val)
        return val
    # warm and hit
    a = convert(7); b = convert(7)
    time.sleep(ttl_seconds)
    c = convert(7)
    print(f"TTL cache sequence -> {a}, {b}, {c}")

# -11) Testing by Example: doctest 

def normalize_space(s: str) -> str:
    """
    Collapse internal whitespace; strip ends.
    >>> normalize_space('  Hello   curious   coder  ')
    'Hello curious coder'
    """
    return " ".join(s.split())

#  12) Optional: Vectorization Taste (NumPy) -

def numpy_demo() -> Optional[Tuple[int, float]]:
    try:
        import numpy as np  # local import
    except Exception:
        print("[numpy] not available, skipping vectorization demo.")
        return None
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=1_0000)
    y = rng.normal(0, 1, size=1_0000)
    # vectorized correlation vs pure-Python loop (speed intuition)
    t0 = time.perf_counter()
    r = float(np.corrcoef(x, y)[0,1])
    v_ms = (time.perf_counter() - t0) * 1000

    # naive python loop
    t0 = time.perf_counter()
    m = sum((float(xi) - float(x.mean()))*(float(yi) - float(y.mean())) for xi, yi in zip(x, y)) / len(x)
    d = math.sqrt(float(((x - x.mean())**2).sum() / len(x))) * math.sqrt(float(((y - y.mean())**2).sum() / len(y)))
    r_naive = float(m / d)
    p_ms = (time.perf_counter() - t0) * 1000
    print(f"[numpy] corr={r:.4f}  vectorized={v_ms:.2f} ms  naive={p_ms:.2f} ms")
    return len(x), r

# - Driver -

def run_all() -> None:
    banner("1) Closures & Decorators")
    explain("""Decorators are closures that take a function and return a new function.
They capture state (like 'cache' or start time) without changing call sites.""")
    print("fib_demo(30) ->", fib_demo(30))

    banner("2) Generators & Pipelines")
    explain("""Generators stream data lazily so you can chain infinite sequences safely.""")
    print("First 8 even squares:", gen_pipeline_demo())

    banner("3) Context Managers")
    with timeblock("toy work"):
        sum(range(100000))
    with suppress_and_report(ZeroDivisionError):
        _ = 1 / 0

    banner("4) Descriptors & Validation")
    m = Metric("Accuracy", 0.9)
    print("Metric:", m.name, m.weight)

    banner("5) Metaclasses")
    lm = LogisticModel(1.7, -0.2)
    print("Logistic(0.5) ->", f"{lm.evaluate(0.5):.3f}")

    banner("6) Structural Pattern Matching")
    for ev in ({"type":"click","x":3,"y":5}, {"type":"key","key":"K"}, {"type":"noop"}):
        print(ev, "->", route(ev))

    banner("7) Asyncio")
    print("Running 3 tasks with jitter/backoff...")
    print(asyncio.run(async_demo()))

    banner("8) Concurrency: threads vs processes")
    stats = concurrency_demo()
    for k,v in stats.items():
        print(f"{k}: {v:.2f} ms")

    banner("9) Type Hints & Protocols")
    print("length_okay on list ->", length_okay([1,2,3], 2))

    banner("10) Caching: LRU + TTL")
    print("heavy_convert(10) twice (LRU hits on second):", heavy_convert(10), heavy_convert(10))
    ttl_cache_demo(0.2)

    banner("11) Doctest example")
    import doctest; doctest.testmod(verbose=False)
    print("normalize_space OK")

    banner("12) NumPy vectorization (optional)")
    numpy_demo()

def run_one(name: str) -> None:
    mapping = {
        "closures": lambda: print("fib_demo(30) ->", fib_demo(30)),
        "generators": lambda: print("even squares:", gen_pipeline_demo()),
        "context": lambda: (timeblock("demo").__enter__(), sum(range(100000)), timeblock("demo").__exit__(None,None,None)),
        "descriptors": lambda: print("Metric:", Metric("Precision", 0.8).__dict__),
        "metaclass": lambda: print("logistic:", LogisticModel(2.0, -1.0).evaluate(0.2)),
        "match": lambda: print(route({"type":"resize","w":800,"h":600})),
        "async": lambda: print(asyncio.run(async_demo())),
        "concurrency": lambda: print(concurrency_demo()),
        "types": lambda: print("SupportsLen(list)?", length_okay([1,2,3])),
        "cache": lambda: (print(heavy_convert(9), heavy_convert(9)), ttl_cache_demo(0.1)),
        "doctest": lambda: (print("Running doctest..."), __import__("doctest").testmod()),
        "numpy": numpy_demo,
    }
    if name == "all":
        run_all()
    elif name in mapping:
        banner(f"Running demo: {name}")
        mapping[name]()
    else:
        raise SystemExit(f"Unknown demo '{name}'. Choices: {', '.join(sorted(mapping))} or 'all'.")

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Curious Coder — Advanced Python Concepts")
    p.add_argument("--demo", default=None, help="Run a specific demo (or 'all').")
    args = p.parse_args(argv)

    if args.demo:
        run_one(args.demo)
    else:
        # Simple interactive menu
        options = [
            "closures","generators","context","descriptors","metaclass",
            "match","async","concurrency","types","cache","doctest","numpy","all","quit"
        ]
        while True:
            print("\nChoose a demo:", ", ".join(options))
            choice = input("> ").strip().lower()
            if choice in ("quit","q","exit"):
                break
            try:
                run_one(choice)
            except SystemExit as e:
                print(e)
            except Exception as exc:
                print("[error]", type(exc).__name__, exc)

if __name__ == "__main__":
    main()
