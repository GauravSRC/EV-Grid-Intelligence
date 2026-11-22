"""Tiny vectorised sparse LP builder on top of scipy.optimize.linprog(method="highs").

Variables are created in blocks (one index per time step); constraints are added row-blocks at a time
as sums of ``coef * x[idx]`` terms, which keeps model construction O(nnz) and free of Python loops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

Term = Tuple[np.ndarray, Union[float, np.ndarray]]


@dataclass
class LPResult:
    status: int
    message: str
    objective: float
    x: np.ndarray
    blocks: Dict[str, np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.x[self.blocks[name]]


@dataclass
class LPBuilder:
    c: List[np.ndarray] = field(default_factory=list)
    lb: List[np.ndarray] = field(default_factory=list)
    ub: List[np.ndarray] = field(default_factory=list)
    blocks: Dict[str, np.ndarray] = field(default_factory=dict)
    n: int = 0
    _eq: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    _le: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    _rhs_eq: List[np.ndarray] = field(default_factory=list)
    _rhs_le: List[np.ndarray] = field(default_factory=list)
    _neq: int = 0
    _nle: int = 0

    def var(self, name: str, size: int, lb: Union[float, np.ndarray] = 0.0, ub: Union[float, np.ndarray] = np.inf,
            cost: Union[float, np.ndarray] = 0.0) -> np.ndarray:
        idx = np.arange(self.n, self.n + size)
        self.n += size
        self.c.append(np.broadcast_to(np.asarray(cost, float), (size,)).copy())
        self.lb.append(np.broadcast_to(np.asarray(lb, float), (size,)).copy())
        self.ub.append(np.broadcast_to(np.asarray(ub, float), (size,)).copy())
        self.blocks[name] = idx
        return idx

    def _rows(self, terms: Sequence[Term], nrows: int, row0: int):
        r, cidx, v = [], [], []
        for idx, coef in terms:
            idx = np.asarray(idx)
            coef = np.broadcast_to(np.asarray(coef, float), idx.shape)
            r.append(row0 + np.arange(nrows))
            cidx.append(idx)
            v.append(coef)
        return np.concatenate(r), np.concatenate(cidx), np.concatenate(v)

    def eq(self, terms: Sequence[Term], rhs: Union[float, np.ndarray]) -> None:
        nrows = len(np.asarray(terms[0][0]))
        self._eq.append(self._rows(terms, nrows, self._neq))
        self._rhs_eq.append(np.broadcast_to(np.asarray(rhs, float), (nrows,)).copy())
        self._neq += nrows

    def le(self, terms: Sequence[Term], rhs: Union[float, np.ndarray]) -> None:
        nrows = len(np.asarray(terms[0][0]))
        self._le.append(self._rows(terms, nrows, self._nle))
        self._rhs_le.append(np.broadcast_to(np.asarray(rhs, float), (nrows,)).copy())
        self._nle += nrows

    def add_cost(self, idx: np.ndarray, cost: Union[float, np.ndarray]) -> None:
        c = np.concatenate(self.c)
        c[idx] += cost
        self.c = [c]

    def _mat(self, parts, nrows):
        if not parts:
            return None
        r = np.concatenate([p[0] for p in parts])
        ci = np.concatenate([p[1] for p in parts])
        v = np.concatenate([p[2] for p in parts])
        return coo_matrix((v, (r, ci)), shape=(nrows, self.n)).tocsr()

    def solve(self) -> LPResult:
        c = np.concatenate(self.c)
        bounds = np.column_stack([np.concatenate(self.lb), np.concatenate(self.ub)])
        bounds = [(lo, None if np.isinf(hi) else hi) for lo, hi in bounds]
        A_eq = self._mat(self._eq, self._neq)
        A_ub = self._mat(self._le, self._nle)
        res = linprog(c, A_ub=A_ub, b_ub=np.concatenate(self._rhs_le) if self._rhs_le else None,
                      A_eq=A_eq, b_eq=np.concatenate(self._rhs_eq) if self._rhs_eq else None,
                      bounds=bounds, method="highs")
        x = res.x if res.x is not None else np.zeros(self.n)
        return LPResult(res.status, res.message, float(res.fun) if res.fun is not None else np.nan, x, self.blocks)
