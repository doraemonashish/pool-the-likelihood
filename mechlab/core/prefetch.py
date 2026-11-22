"""Order-preserving background batch generation with a bounded queue.

``Prefetcher(fn, start, stop, workers, ahead)`` returns ``fn(step)`` for consecutive steps,
computing up to ``ahead`` future steps in worker processes. ``fn`` must be picklable (a
top-level function or functools.partial). With workers=0 everything runs inline.
"""
from __future__ import annotations

import collections
import multiprocessing as mp
from typing import Callable


class Prefetcher:
    def __init__(self, fn: Callable[[int], dict], start: int, stop: int, workers: int = 2, ahead: int = 8):
        self.fn, self.stop, self.workers, self.ahead = fn, stop, workers, max(1, ahead)
        self._pool = None
        self._q: collections.deque = collections.deque()
        self._next_submit = start
        self._next_get = start

    def __enter__(self):
        if self.workers > 0:
            try:
                self._pool = mp.get_context("fork").Pool(self.workers)
            except Exception:  # pragma: no cover - restricted environments
                self._pool = None
        self._fill()
        return self

    def __exit__(self, *exc):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()

    def _fill(self):
        if self._pool is None:
            return
        while len(self._q) < self.ahead and self._next_submit < self.stop:
            self._q.append((self._next_submit, self._pool.apply_async(self.fn, (self._next_submit,))))
            self._next_submit += 1

    def get(self, step: int) -> dict:
        assert step == self._next_get, f"Prefetcher expects sequential steps (got {step}, expected {self._next_get})"
        self._next_get += 1
        if self._pool is None:
            return self.fn(step)
        s, res = self._q.popleft()
        assert s == step
        out = res.get()
        self._fill()
        return out
