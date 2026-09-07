from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import robust_minmax


def _safe_col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def compute_team_factors(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    fm25 = _safe_col(work, "fm_25")
    fm26 = _safe_col(work, "fm_26")
    pv26 = _safe_col(work, "pv_26", 0).fillna(0)
    work["delta"] = (fm26 - fm25).clip(-3, 3)
    work["weight"] = pv26.clip(0, 5)

    rows = []
    for team, group in work.groupby("team", dropna=False):
        valid = group["delta"].notna() & (group["weight"] > 0)
        evidence = float(group.loc[valid, "weight"].sum())
        raw = 0.0
        if evidence > 0:
            raw = float(np.average(group.loc[valid, "delta"], weights=group.loc[valid, "weight"]))
        # Bayesian-style shrinkage toward 0 with ~20 pseudo-player-games of prior strength.
        posterior = raw * evidence / (evidence + 20.0)
        rows.append({"team": team, "team_delta_raw": raw, "team_evidence": evidence, "team_factor": posterior})
    return pd.DataFrame(rows)


def project_players(df: pd.DataFrame, manual_overweights: list[str] | None = None, manual_premium: float = 0.05) -> pd.DataFrame:
    work = df.copy()
    manual_overweights = set(manual_overweights or [])

    for col in ["fvm", "quote", "pv_26", "fm_26", "pv_25", "fm_25", "pv_24", "fm_24"]:
        if col not in work:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # Historical baseline: last season dominant, two seasons ago as stabilizer.
    hist = work["fm_25"].copy()
    both = work["fm_25"].notna() & work["fm_24"].notna()
    hist.loc[both] = 0.70 * work.loc[both, "fm_25"] + 0.30 * work.loc[both, "fm_24"]
    hist = hist.fillna(work["fm_24"])

    # Market prior converts FVM percentile into a plausible fantasy-average anchor.
    fvm_rank = robust_minmax(work["fvm"].fillna(work["fvm"].median()))
    market_implied_fm = 5.65 + 2.10 * fvm_rank
    hist = hist.fillna(market_implied_fm)
    hist = 0.82 * hist + 0.18 * market_implied_fm

    current_weight = (work["pv_26"].fillna(0) / (work["pv_26"].fillna(0) + 14.0)).clip(0, 0.28)
    current_fm = work["fm_26"].fillna(hist)
    projected_fm = hist * (1 - current_weight) + current_fm * current_weight

    team_factors = compute_team_factors(work)
    team_map = team_factors.set_index("team")["team_factor"].to_dict() if not team_factors.empty else {}
    work["team_factor"] = work["team"].map(team_map).fillna(0.0)
    projected_fm = projected_fm + 0.12 * work["team_factor"]
    projected_fm = projected_fm * np.where(work["team"].isin(manual_overweights), 1.0 + manual_premium, 1.0)

    # Expected availability. New arrivals without history are shrunk to 27 appearances.
    pv25 = work["pv_25"].fillna(27)
    pv24 = work["pv_24"].fillna(pv25)
    expected_apps = (0.72 * pv25 + 0.28 * pv24).clip(10, 36)
    # Small current-season availability update, deliberately weak after only a few rounds.
    expected_apps = (expected_apps + (work["pv_26"].fillna(0) - 1.5) * 0.35).clip(10, 36)

    experience = ((work["pv_25"].fillna(0) + work["pv_24"].fillna(0)) / 60.0).clip(0, 1)
    current_gap = (current_fm - hist).fillna(0).clip(-2.5, 2.5)
    risk = (0.70 * (1 - experience) + 0.30 * (work["pv_25"].fillna(0) < 15).astype(float)).clip(0, 1)
    upside = (0.45 * robust_minmax(current_gap) + 0.35 * fvm_rank + 0.20 * risk).clip(0, 1)

    work["projected_fm"] = projected_fm.clip(5.0, 9.5)
    work["expected_apps"] = expected_apps
    work["expected_fp"] = work["projected_fm"] * work["expected_apps"]
    work["projection_sd"] = 12 + 38 * risk
    work["risk"] = risk
    work["upside"] = upside
    work["quality"] = robust_minmax(work["expected_fp"])
    work["market_quality"] = fvm_rank
    return work
