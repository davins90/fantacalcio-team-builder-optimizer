from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import AuctionConfig
from .utils import parse_roles, primary_market_role


def base_prices(players: pd.DataFrame, config: AuctionConfig) -> pd.Series:
    fvm = pd.to_numeric(players["fvm"], errors="coerce").fillna(1).clip(lower=1)
    return (fvm * config.starting_budget / 1000.0).clip(lower=1.0)


def _event_lookup(players: pd.DataFrame) -> dict[str, dict]:
    return players.set_index("name_norm").to_dict("index")


def role_inflation_posteriors(players: pd.DataFrame, events: list[dict], config: AuctionConfig) -> dict[str, dict]:
    lookup = _event_lookup(players)
    logs: dict[str, list[float]] = defaultdict(list)
    overall = []
    prior_strength = 5.0

    for ev in events:
        row = lookup.get(ev.get("name_norm", ""))
        if not row:
            continue
        base = max(1.0, float(row.get("fvm") or 1) * config.starting_budget / 1000.0)
        price = max(1.0, float(ev.get("price", 1)))
        # Cheap players create unstable ratios, so cap individual evidence.
        log_ratio = float(np.clip(math.log(price / base), -0.70, 0.70))
        overall.append(log_ratio)
        group = primary_market_role(row.get("roles", ""))
        logs[group].append(log_ratio)

    overall_post = sum(overall) / (prior_strength + len(overall)) if overall else 0.0
    result = {"ALL": {"n": len(overall), "multiplier": math.exp(overall_post)}}
    for role, vals in logs.items():
        post = (0.35 * overall_post * prior_strength + sum(vals)) / (prior_strength + len(vals))
        result[role] = {"n": len(vals), "multiplier": float(math.exp(post))}
    return result


def scarcity_multipliers(players: pd.DataFrame, sold_names: set[str]) -> dict[str, float]:
    initial: dict[str, float] = defaultdict(float)
    remaining: dict[str, float] = defaultdict(float)
    for _, row in players.iterrows():
        fvm = float(row.get("fvm") or 1)
        if fvm < 2.0:
            continue
        weight = math.sqrt(fvm)
        group = primary_market_role(row.get("roles", ""))
        initial[group] += weight
        if row["name_norm"] not in sold_names:
            remaining[group] += weight
    out = {}
    for role, total in initial.items():
        frac = remaining[role] / max(1.0, total)
        # Scarcity rises up to +22% as starting-caliber talent in that role depletes
        out[role] = float(1.0 + 0.22 * ((1.0 - frac) ** 1.5))
    return out


def liquidity_multiplier(events: list[dict], config: AuctionConfig) -> float:
    spent = sum(float(e.get("price", 0)) for e in events)
    sold = len(events)
    budget_left = max(1.0, config.total_market_budget - spent)
    slots_left = max(1, config.total_roster_slots - sold)
    initial_per_slot = config.total_market_budget / config.total_roster_slots
    current_per_slot = budget_left / slots_left
    return float(np.clip((current_per_slot / initial_per_slot) ** 0.22, 0.88, 1.15))


def fit_multiplier(row: pd.Series, my_roster: pd.DataFrame, config: AuctionConfig) -> float:
    roles = set(parse_roles(row.get("roles", "")))
    if not roles:
        return 0.90

    def count_any(target):
        if my_roster.empty:
            return 0
        return sum(bool(set(parse_roles(r)).intersection(target)) for r in my_roster["roles"])

    cur_por = count_any({"Por"})
    if roles == {"Por"} and cur_por >= config.target_goalkeepers:
        return 0.15  # Goalkeeper department already full

    needs = {
        "Por": max(0, config.target_goalkeepers - cur_por),
        "Dc": max(0, 4 - count_any({"Dc"})),
        "Dd": max(0, 2 - count_any({"Dd"})),
        "Ds": max(0, 2 - count_any({"Ds"})),
        "E": max(0, 2 - count_any({"E"})),
        "M": max(0, 2 - count_any({"M"})),
        "C": max(0, 3 - count_any({"C"})),
        "Pc": max(0, 2 - count_any({"Pc"})),
        "CREATIVE": max(0, 3 - count_any({"W", "T", "A"})),
    }
    hit = 0
    if "Por" in roles and needs["Por"] > 0: hit += 1
    if "Dc" in roles and needs["Dc"] > 0: hit += 1
    if "Dd" in roles and needs["Dd"] > 0: hit += 1
    if "Ds" in roles and needs["Ds"] > 0: hit += 1
    if "E" in roles and needs["E"] > 0: hit += 1
    if "M" in roles and needs["M"] > 0: hit += 1
    if "C" in roles and needs["C"] > 0: hit += 1
    if "Pc" in roles and needs["Pc"] > 0: hit += 1
    if roles.intersection({"W", "T", "A"}) and needs["CREATIVE"] > 0: hit += 1

    flexibility = min(0.04, max(0, len(roles) - 1) * 0.02)
    if hit == 0:
        return float(0.85 + flexibility)
    return float(1.0 + min(0.10, hit * 0.03) + flexibility)



def dynamic_market_values(players: pd.DataFrame, events: list[dict], my_roster: pd.DataFrame, config: AuctionConfig) -> pd.DataFrame:
    work = players.copy()
    sold = {e.get("name_norm") for e in events}
    inflation = role_inflation_posteriors(work, events, config)
    scarcity = scarcity_multipliers(work, sold)
    liquidity = liquidity_multiplier(events, config)
    work["base_price"] = base_prices(work, config)

    dyn = []
    inflation_mult = []
    scarcity_mult = []
    fit_mult = []
    for _, row in work.iterrows():
        role = primary_market_role(row.get("roles", ""))
        im = inflation.get(role, inflation.get("ALL", {"multiplier": 1.0}))["multiplier"]
        sm = scarcity.get(role, 1.0)
        fm = fit_multiplier(row, my_roster, config)
        value = float(row["base_price"] * im * sm * liquidity)
        dyn.append(max(1.0, value))
        inflation_mult.append(im)
        scarcity_mult.append(sm)
        fit_mult.append(fm)
    work["market_inflation"] = inflation_mult
    work["scarcity_mult"] = scarcity_mult
    work["fit_mult"] = fit_mult
    work["dynamic_price"] = dyn
    work["liquidity_mult"] = liquidity
    return work
