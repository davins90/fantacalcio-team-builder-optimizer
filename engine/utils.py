from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd

MANTRA_ROLES = ["Por", "Dc", "Dd", "Ds", "B", "E", "M", "C", "W", "T", "A", "Pc"]
ROLE_SET = set(MANTRA_ROLES)


def normalize_name(value: str) -> str:
    value = "" if value is None else str(value)
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return " ".join(value.split())


def parse_roles(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        tokens = [str(x) for x in value]
    else:
        text = str(value).replace("\\", "/").replace("-", "/")
        tokens = re.split(r"[/,;| +]+", text)
    normalized = []
    aliases = {
        "P": "Por", "POR": "Por", "PORTIERE": "Por",
        "DC": "Dc", "DD": "Dd", "DS": "Ds", "B": "B", "E": "E",
        "M": "M", "C": "C", "W": "W", "T": "T", "A": "A", "PC": "Pc",
    }
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        role = aliases.get(tok.upper(), tok)
        if role in ROLE_SET and role not in normalized:
            normalized.append(role)
    return normalized


def roles_text(value) -> str:
    return "/".join(parse_roles(value))


def has_any_role(value, roles: Iterable[str]) -> bool:
    mine = set(parse_roles(value))
    return bool(mine.intersection(set(roles)))


def primary_market_role(value) -> str:
    roles = parse_roles(value)
    # Ordered by auction scarcity / tactical specificity, not by football hierarchy.
    priority = ["Pc", "A", "T", "W", "E", "M", "C", "Dc", "Dd", "Ds", "B", "Por"]
    for role in priority:
        if role in roles:
            return role
    return roles[0] if roles else "UNK"


def coerce_num(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(
        series.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )


def robust_minmax(series: pd.Series, low_q: float = 0.05, high_q: float = 0.95) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)
    lo = s.quantile(low_q)
    hi = s.quantile(high_q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=series.index)
    return ((s - lo) / (hi - lo)).clip(0, 1).fillna(0.5)
