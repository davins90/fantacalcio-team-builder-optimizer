from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .config import AuctionConfig
from .utils import has_any_role, robust_minmax


@dataclass
class OptimizationResult:
    selected_names: list[str]
    objective: float
    feasible: bool
    reason: str = ""


def _current_count(my_roster: pd.DataFrame, roles: set[str]) -> int:
    if my_roster.empty:
        return 0
    return sum(has_any_role(r, roles) for r in my_roster["roles"])


def _solve_once(
    available: pd.DataFrame,
    my_roster: pd.DataFrame,
    budget_left: float,
    config: AuctionConfig,
    score_noise: np.ndarray | None = None,
    relax: bool = False,
) -> OptimizationResult:
    n = len(available)
    need = config.roster_size - len(my_roster)
    if need <= 0:
        return OptimizationResult([], 0.0, True)
    if n < need:
        return OptimizationResult([], 0.0, False, "Giocatori disponibili insufficienti")

    fp = pd.to_numeric(available["expected_fp"], errors="coerce").fillna(0).to_numpy(float)
    risk = pd.to_numeric(available["risk"], errors="coerce").fillna(0.5).to_numpy(float)
    upside = pd.to_numeric(available["upside"], errors="coerce").fillna(0.5).to_numpy(float)
    price = pd.to_numeric(available["dynamic_price"], errors="coerce").fillna(1).clip(lower=1).to_numpy(float)
    value = robust_minmax(pd.Series(fp / np.maximum(price, 1))).to_numpy(float)
    score = fp - config.risk_aversion * 28.0 * risk + config.upside_weight * 18.0 * upside + config.value_weight * 18.0 * value
    if score_noise is not None:
        score = score + score_noise

    # scipy.milp minimizes; negate the utility.
    c = -score
    constraints = []
    # Exact remaining roster size.
    constraints.append(LinearConstraint(np.ones((1, n)), [need], [need]))
    # Budget.
    constraints.append(LinearConstraint(price.reshape(1, -1), [-np.inf], [budget_left]))

    # Goalkeeper constraint: strictly fill needed goalkeepers up to target, never exceed.
    cur_por = _current_count(my_roster, {"Por"})
    needed_por = max(0, config.target_goalkeepers - cur_por)
    needed_por = min(needed_por, need)
    por_mask = np.array([1.0 if has_any_role(v, {"Por"}) else 0.0 for v in available["roles"]])
    if needed_por > 0:
        constraints.append(LinearConstraint(por_mask.reshape(1, -1), [needed_por], [needed_por]))
    else:
        constraints.append(LinearConstraint(por_mask.reshape(1, -1), [-np.inf], [0]))

    if not relax:
        role_requirements = [
            ({"Dc"}, 4),
            ({"Dd"}, 2),
            ({"Ds"}, 2),
            ({"E", "W"}, 2),
            ({"M"}, 2),
            ({"C"}, 3),
            ({"Pc"}, 2),
            ({"W", "T", "A"}, 3),
        ]
        for roles, target in role_requirements:
            deficit = max(0, target - _current_count(my_roster, roles))
            if deficit <= 0:
                continue
            deficit = min(deficit, need - (needed_por if "Por" not in roles else 0))
            mask = np.array([1.0 if has_any_role(v, roles) else 0.0 for v in available["roles"]])
            if mask.sum() >= deficit:
                constraints.append(LinearConstraint(mask.reshape(1, -1), [deficit], [np.inf]))

        # Diversification cap by real club.
        if config.max_players_per_real_team > 0 and "team" in available:
            current_counts = my_roster["team"].value_counts().to_dict() if not my_roster.empty else {}
            for team, idx in available.groupby("team").groups.items():
                cap = max(0, config.max_players_per_real_team - int(current_counts.get(team, 0)))
                mask = np.zeros(n)
                positions = [available.index.get_loc(i) for i in idx]
                mask[positions] = 1
                constraints.append(LinearConstraint(mask.reshape(1, -1), [-np.inf], [cap]))
    else:
        # Soft relaxed constraints: maintain minimal balance across departments
        non_por_need = need - needed_por
        if non_por_need >= 3:
            for roles in [{"Dc", "Dd", "Ds", "B", "E"}, {"M", "C"}, {"W", "T", "A", "Pc"}]:
                deficit = max(0, 1 - _current_count(my_roster, roles))
                if deficit > 0:
                    mask = np.array([1.0 if has_any_role(v, roles) else 0.0 for v in available["roles"]])
                    if mask.sum() >= deficit:
                        constraints.append(LinearConstraint(mask.reshape(1, -1), [deficit], [np.inf]))


    result = milp(
        c=c,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints,
        options={"time_limit": 1.5, "mip_rel_gap": 0.015},
    )
    if result.x is None or not result.success:
        return OptimizationResult([], 0.0, False, str(result.message))
    chosen = available.iloc[np.where(result.x > 0.5)[0]]
    return OptimizationResult(chosen["name_norm"].tolist(), float(-result.fun), True)


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
    available = players[~players["name_norm"].isin(sold)].copy().reset_index(drop=True)

    base = _solve_once(available, my_roster, budget_left, config, relax=False)
    if not base.feasible:
        base = _solve_once(available, my_roster, budget_left, config, relax=True)

    sims = simulations if simulations is not None else config.simulations
    counts = {name: 0 for name in available["name_norm"]}
    rng = np.random.default_rng(42 + len(events))
    successful = 0
    for _ in range(max(1, sims)):
        sd = pd.to_numeric(available["projection_sd"], errors="coerce").fillna(25).to_numpy(float)
        noise = rng.normal(0, sd * 0.12)
        sim = _solve_once(available, my_roster, budget_left, config, score_noise=noise, relax=not base.feasible)
        if sim.feasible:
            successful += 1
            for name in sim.selected_names:
                counts[name] = counts.get(name, 0) + 1
    denom = max(1, successful)
    exposure = pd.DataFrame({"name_norm": list(counts), "exposure": [counts[x] / denom for x in counts]})
    return exposure, base
