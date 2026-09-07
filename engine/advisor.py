from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AuctionConfig
from .market import dynamic_market_values, role_inflation_posteriors
from .optimizer import optimize_portfolio
from .projections import project_players
from .utils import primary_market_role, robust_minmax


def state_frames(players: pd.DataFrame, events: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sold = {e.get("name_norm") for e in events}
    mine = {e.get("name_norm") for e in events if e.get("mine")}
    return players[~players["name_norm"].isin(sold)].copy(), players[players["name_norm"].isin(mine)].copy()


def analyze_auction(raw_players: pd.DataFrame, events: list[dict], config: AuctionConfig, manual_overweights: list[str] | None = None) -> dict:
    if "expected_fp" in raw_players.columns and "projected_fm" in raw_players.columns:
        projected = raw_players
    else:
        projected = project_players(raw_players, manual_overweights, config.manual_team_premium)
    _, my_roster = state_frames(projected, events)
    market = dynamic_market_values(projected, events, my_roster, config)

    exposure, opt = optimize_portfolio(market, events, config)
    market = market.merge(exposure, on="name_norm", how="left")
    market["exposure"] = market["exposure"].fillna(0.0)

    spend_me = sum(float(e.get("price", 0)) for e in events if e.get("mine"))
    spend_all = sum(float(e.get("price", 0)) for e in events)
    my_budget_left = max(0, config.starting_budget - int(spend_me))
    slots_needed = max(0, config.roster_size - len(my_roster))

    # Absolute auction budget ceiling:
    # To fill K remaining slots, you must keep at least 1 credit for each remaining slot.
    # Therefore, maximum single bid is: my_budget_left - (slots_needed - 1).
    if slots_needed <= 0:
        max_possible_bid = 0
    else:
        max_possible_bid = max(0, my_budget_left - (slots_needed - 1))

    # Portfolio premium is deliberately modest: optimizer signal influences the bid, not vice versa.
    portfolio_mult = 0.94 + 0.14 * market["exposure"]
    model_edge = robust_minmax(market["expected_fp"] / market["dynamic_price"].clip(lower=1))
    quality_mult = 0.96 + 0.09 * model_edge
    market["fair_value"] = market["dynamic_price"] * market["fit_mult"] * portfolio_mult
    raw_bid = np.maximum(1, np.floor(market["fair_value"] * quality_mult + 0.5)).astype(int)
    if slots_needed <= 0 or max_possible_bid <= 0:
        market["max_bid"] = 0
    else:
        market["max_bid"] = np.minimum(raw_bid, max_possible_bid)
    market["value_index"] = model_edge

    sold = {e.get("name_norm") for e in events}
    available = market[~market["name_norm"].isin(sold)].copy()
    available["nomination_score"] = (
        0.45 * available["exposure"]
        + 0.25 * available["value_index"]
        + 0.15 * (available["fit_mult"] - 0.85).clip(lower=0) / 0.25
        + 0.15 * (available["scarcity_mult"] - 1.0).clip(lower=0) / 0.22
    )
    # Prefer targets that are viable / playable (FVM >= 3) and within legal bid
    candidate_targets = available[(available["max_bid"] > 0) & (available["fvm"] >= 3)]
    if candidate_targets.empty:
        candidate_targets = available[available["max_bid"] > 0]
    if candidate_targets.empty:
        candidate_targets = available
    next_targets = candidate_targets.sort_values(["nomination_score", "exposure", "expected_fp"], ascending=False).head(15)

    # Budget-drain nomination: expensive/high-FVM player with low optimizer exposure.
    drain = available.copy()
    drain["drain_score"] = robust_minmax(drain["dynamic_price"]) * (1.0 - drain["exposure"].clip(0, 1))
    budget_drain = drain.sort_values("drain_score", ascending=False).head(5)

    inflation = role_inflation_posteriors(market, events, config)
    state = {
        "players": market,
        "available": available,
        "my_roster": my_roster,
        "exposure": exposure,
        "optimizer": opt,
        "next_targets": next_targets,
        "budget_drain": budget_drain,
        "inflation": inflation,
        "my_spend": spend_me,
        "my_budget_left": my_budget_left,
        "slots_needed": slots_needed,
        "max_possible_bid": max_possible_bid,
        "market_spend": spend_all,
        "market_budget_left": config.total_market_budget - spend_all,
        "sold_count": len(events),
    }
    return state


def player_advice(analysis: dict, name_norm: str) -> dict | None:
    rows = analysis["players"][analysis["players"]["name_norm"] == name_norm]
    if rows.empty:
        return None
    row = rows.iloc[0]
    bid = int(row["max_bid"])
    exposure = float(row["exposure"])
    slots_needed = analysis.get("slots_needed", 1)
    if slots_needed <= 0:
        decision = "ROSTER_FULL"
    elif bid <= 0:
        decision = "OUT_OF_BUDGET"
    elif exposure >= 0.35:
        decision = "BUY"
    elif exposure >= 0.12 or float(row["value_index"]) >= 0.65:
        decision = "VALUE"
    else:
        decision = "DISCIPLINE"
    return {
        "name": row["name"],
        "team": row.get("team", ""),
        "roles": row.get("roles", ""),
        "expected_fp": float(row["expected_fp"]),
        "projected_fm": float(row["projected_fm"]),
        "risk": float(row["risk"]),
        "upside": float(row["upside"]),
        "base_price": float(row["base_price"]),
        "fair_value": float(row["fair_value"]),
        "max_bid": bid,
        "exposure": exposure,
        "inflation": float(row["market_inflation"]),
        "scarcity": float(row["scarcity_mult"]),
        "decision": decision,
        "market_role": primary_market_role(row.get("roles", "")),
    }

