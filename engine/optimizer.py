from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from .config import AuctionConfig
from .tactics import MANTRA_FORMATIONS
from .utils import has_any_role, parse_roles, robust_minmax

# Minimum department coverage across the whole 25-man squad (bench included).
SQUAD_ROLE_MINIMUMS = [
    ({"Dc"}, 4),
    ({"Dd"}, 2),
    ({"Ds"}, 2),
    ({"E", "W"}, 2),
    ({"M"}, 2),
    ({"C"}, 3),
    ({"Pc"}, 2),
    ({"W", "T", "A"}, 3),
]


@dataclass
class OptimizationResult:
    selected_names: list[str]
    objective: float
    feasible: bool
    reason: str = ""
    formation: str = ""
    starters: list[str] = field(default_factory=list)


def _current_count(my_roster: pd.DataFrame, roles: set[str]) -> int:
    if my_roster.empty:
        return 0
    return sum(has_any_role(r, roles) for r in my_roster["roles"])


class _SquadModel:
    """Best-XI squad model.

    The squad is worth what its fielded eleven scores, not the sum of all 25 players,
    so a bench player only counts for `config.bench_weight` of their expected points.

    Variables, concatenated into a single vector:
      x_i          player i is in the 25-man squad
      m_f          formation f is the one being built towards (exactly one)
      w_(i,g)      player i fills a slot of requirement group g (of some formation)

    A player is a starter iff they fill a slot, and `sum_g w_(i,g) <= x_i` keeps each
    player in at most one slot: this is a bipartite b-matching, the same rule the roster
    page uses, so the optimizer and the UI agree on what a fieldable XI is.
    """

    def __init__(self, pool: pd.DataFrame, owned_mask: np.ndarray, my_roster: pd.DataFrame,
                 budget_left: float, config: AuctionConfig, relax: bool = False):
        self.pool = pool
        self.config = config
        self.n = len(pool)
        self.roles = [set(parse_roles(r)) for r in pool["roles"].tolist()]
        self.fp = pd.to_numeric(pool["expected_fp"], errors="coerce").fillna(0).to_numpy(float)
        self.price = pd.to_numeric(pool["dynamic_price"], errors="coerce").fillna(1).clip(lower=1).to_numpy(float)
        self.owned = owned_mask

        self.groups: list[tuple[int, set[str], int]] = []
        self.formations: list[str] = []
        for key, fmt in MANTRA_FORMATIONS.items():
            f_idx = len(self.formations)
            self.formations.append(key)
            for roles, count, _label in fmt["needs"]:
                self.groups.append((f_idx, roles, count))

        # Only build a variable where the player is actually eligible for the group.
        self.pairs = [
            (i, g_idx)
            for g_idx, (_f, g_roles, _c) in enumerate(self.groups)
            for i in range(self.n)
            if self.roles[i] & g_roles
        ]
        self.off_m = self.n
        self.off_w = self.n + len(self.formations)
        self.size = self.off_w + len(self.pairs)
        self._build(my_roster, budget_left, relax)

    def _build(self, my_roster: pd.DataFrame, budget_left: float, relax: bool) -> None:
        cfg = self.config
        n, N = self.n, self.size
        rows, lows, highs = [], [], []

        def add(idx, vals, lo, hi):
            idx = np.asarray(idx, dtype=int)
            rows.append(sp.csr_matrix((np.asarray(vals, float), (np.zeros(len(idx), int), idx)), shape=(1, N)))
            lows.append(lo)
            highs.append(hi)

        def squad_mask(target: set[str]) -> np.ndarray:
            return np.array([1.0 if self.roles[i] & target else 0.0 for i in range(n)])

        add(np.arange(n), np.ones(n), cfg.roster_size, cfg.roster_size)
        # Owned players are sunk cost: only the still-available ones consume the budget.
        add(np.arange(n), self.price * (~self.owned), -np.inf, budget_left)
        add(self.off_m + np.arange(len(self.formations)), np.ones(len(self.formations)), 1, 1)
        add(np.arange(n), squad_mask({"Por"}), cfg.target_goalkeepers, cfg.target_goalkeepers)

        if not relax:
            for target, count in SQUAD_ROLE_MINIMUMS:
                add(np.arange(n), squad_mask(target), count, np.inf)
            if cfg.max_players_per_real_team > 0 and "team" in self.pool:
                for _team, idx in self.pool.groupby("team").groups.items():
                    positions = [self.pool.index.get_loc(i) for i in idx]
                    add(positions, np.ones(len(positions)), -np.inf, cfg.max_players_per_real_team)

        # Each requirement group must be filled exactly when its formation is the chosen one.
        by_group: dict[int, list[int]] = {}
        by_player: dict[int, list[int]] = {}
        for w_idx, (i, g_idx) in enumerate(self.pairs):
            by_group.setdefault(g_idx, []).append(w_idx)
            by_player.setdefault(i, []).append(w_idx)
        for g_idx, (f_idx, _roles, count) in enumerate(self.groups):
            w_idx = np.array(by_group.get(g_idx, []), dtype=int) + self.off_w
            add(np.concatenate([w_idx, [self.off_m + f_idx]]),
                np.concatenate([np.ones(len(w_idx)), [-float(count)]]), 0, 0)
        # A player fills at most one slot, and only if they are in the squad.
        for i, w_list in by_player.items():
            w_idx = np.array(w_list, dtype=int) + self.off_w
            add(np.concatenate([w_idx, [i]]), np.concatenate([np.ones(len(w_idx)), [-1.0]]), -np.inf, 0)

        self.constraints = LinearConstraint(sp.vstack(rows).tocsc(), np.array(lows, float), np.array(highs, float))
        # Players already bought are fixed in the squad.
        lb = self.owned.astype(float)
        self.bounds = Bounds(np.concatenate([lb, np.zeros(N - n)]), np.ones(N))

    def objective(self, score_noise: np.ndarray | None = None) -> np.ndarray:
        lam = float(np.clip(self.config.bench_weight, 0.0, 1.0))
        cfg = self.config
        risk = pd.to_numeric(self.pool["risk"], errors="coerce").fillna(0.5).to_numpy(float)
        upside = pd.to_numeric(self.pool["upside"], errors="coerce").fillna(0.5).to_numpy(float)
        value = robust_minmax(pd.Series(self.fp / np.maximum(self.price, 1))).to_numpy(float)
        # Squad-wide preferences (risk, upside, value for money) stay on the squad variable.
        squad_score = (
            lam * self.fp
            - cfg.risk_aversion * 28.0 * risk
            + cfg.upside_weight * 18.0 * upside
            + cfg.value_weight * 18.0 * value
        )
        if score_noise is not None:
            squad_score = squad_score + score_noise
        c = np.zeros(self.size)
        c[: self.n] = -squad_score
        starter_bonus = (1.0 - lam) * self.fp
        for w_idx, (i, _g) in enumerate(self.pairs):
            c[self.off_w + w_idx] = -starter_bonus[i]
        return c

    def solve(self, score_noise: np.ndarray | None = None, integer: bool = True, time_limit: float = 4.0):
        c = self.objective(score_noise)
        integrality = np.ones(self.size) if integer else np.zeros(self.size)
        options = {"time_limit": time_limit}
        if integer:
            options["mip_rel_gap"] = 0.02
        return milp(c=c, integrality=integrality, bounds=self.bounds, constraints=self.constraints, options=options)

    def read_solution(self, result) -> OptimizationResult:
        if result.x is None or not result.success:
            return OptimizationResult([], 0.0, False, str(result.message))
        x = result.x[: self.n]
        chosen = self.pool.iloc[np.where(x > 0.5)[0]]
        f_vals = result.x[self.off_m : self.off_w]
        formation = self.formations[int(np.argmax(f_vals))] if len(f_vals) else ""
        starter_idx = sorted({self.pairs[w][0] for w in range(len(self.pairs)) if result.x[self.off_w + w] > 0.5})
        return OptimizationResult(
            chosen["name_norm"].tolist(),
            float(-result.fun),
            True,
            formation=formation,
            starters=self.pool.iloc[starter_idx]["name_norm"].tolist(),
        )


def optimize_portfolio(
    players: pd.DataFrame,
    events: list[dict],
    config: AuctionConfig,
    simulations: int | None = None,
) -> tuple[pd.DataFrame, OptimizationResult]:
    sold = {e["name_norm"] for e in events}
    mine = {e["name_norm"] for e in events if e.get("mine")}
    my_roster = players[players["name_norm"].isin(mine)].copy()
    spent = sum(float(e["price"]) for e in events if e.get("mine"))
    budget_left = config.starting_budget - spent
    # The squad model reasons over the final 25, so it needs the players already bought.
    pool = players[~players["name_norm"].isin(sold - mine)].copy().reset_index(drop=True)
    owned = pool["name_norm"].isin(mine).to_numpy()

    if len(pool) < config.roster_size:
        empty = pd.DataFrame({"name_norm": pool["name_norm"], "exposure": 0.0})
        return empty, OptimizationResult([], 0.0, False, "Giocatori disponibili insufficienti")

    model = _SquadModel(pool, owned, my_roster, budget_left, config, relax=False)
    base = model.read_solution(model.solve())
    if not base.feasible:
        model = _SquadModel(pool, owned, my_roster, budget_left, config, relax=True)
        base = model.read_solution(model.solve())

    # Exposure comes from the LP relaxation under perturbed projections: on this
    # assignment-structured model it matches the integer optimum but solves ~15x faster,
    # and a fractional weight is itself a natural exposure reading.
    sims = simulations if simulations is not None else config.simulations
    sd = pd.to_numeric(pool["projection_sd"], errors="coerce").fillna(25).to_numpy(float)
    rng = np.random.default_rng(42 + len(events))
    weights = np.zeros(len(pool))
    successful = 0
    for _ in range(max(1, sims)):
        res = model.solve(score_noise=rng.normal(0, sd * 0.12), integer=False, time_limit=2.0)
        if res.x is not None and res.success:
            successful += 1
            weights += np.clip(res.x[: len(pool)], 0.0, 1.0)
    if successful == 0:
        weights = np.array([1.0 if name in set(base.selected_names) else 0.0 for name in pool["name_norm"]])
        successful = 1
    exposure = pd.DataFrame({"name_norm": pool["name_norm"], "exposure": weights / successful})
    return exposure, base
