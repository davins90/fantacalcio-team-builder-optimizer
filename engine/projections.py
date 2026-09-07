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

    # Expected availability.
    fvm_val = work["fvm"].fillna(1.0).clip(lower=1.0)
    # Market prior: reserve players (FVM ~ 1-2) expect 1-3 appearances; regular starters expect 25-33
    market_implied_pv = np.clip(1.5 + 30.0 * (fvm_val / (fvm_val + 35.0)), 1.0, 34.0)

    has_pv25 = work["pv_25"].notna()
    has_pv24 = work["pv_24"].notna()
    hist_pv = pd.Series(np.nan, index=work.index, dtype=float)
    both_pv = has_pv25 & has_pv24
    hist_pv.loc[both_pv] = 0.72 * work.loc[both_pv, "pv_25"] + 0.28 * work.loc[both_pv, "pv_24"]
    hist_pv.loc[has_pv25 & ~has_pv24] = work.loc[has_pv25 & ~has_pv24, "pv_25"]
    hist_pv.loc[~has_pv25 & has_pv24] = work.loc[~has_pv25 & has_pv24, "pv_24"]

    # Bayesian shrinkage toward market implied appearances:
    # Use historical evidence when present (shrunk slightly to prior); otherwise use prior
    pv_weight = np.where(hist_pv.notna(), 0.85, 0.0)
    expected_apps = np.where(hist_pv.notna(), hist_pv * pv_weight + market_implied_pv * (1.0 - pv_weight), market_implied_pv)
    expected_apps = pd.Series(expected_apps, index=work.index, dtype=float)

    # Small current-season availability update
    pv26 = work["pv_26"].fillna(0)
    expected_apps = (expected_apps + (pv26 - 1.0).clip(lower=-2.0, upper=4.0) * 0.35).clip(1.0, 36.0)

    total_pv = work["pv_25"].fillna(0) + work["pv_24"].fillna(0)
    experience = (total_pv / 60.0).clip(0, 1)
    current_gap = (current_fm - hist).fillna(0).clip(-2.5, 2.5)
    fvm_stabilizer = (fvm_val / (fvm_val + 50.0)).clip(0, 0.5)
    risk = (0.65 * (1.0 - experience) + 0.20 * (work["pv_25"].fillna(0) < 15).astype(float) - 0.15 * fvm_stabilizer).clip(0.08, 0.95)
    upside = (0.40 * robust_minmax(current_gap) + 0.40 * fvm_rank + 0.20 * risk).clip(0, 1)

    work["projected_fm"] = projected_fm.clip(5.0, 9.5)
    work["expected_apps"] = expected_apps
    work["expected_fp"] = work["projected_fm"] * work["expected_apps"]
    work["projection_sd"] = 12 + 38 * risk
    work["risk"] = risk
    work["upside"] = upside
    work["quality"] = robust_minmax(work["expected_fp"])
    work["market_quality"] = fvm_rank
    return work
