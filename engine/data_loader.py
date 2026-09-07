from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import BinaryIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .utils import MANTRA_ROLES, coerce_num, normalize_name, parse_roles, roles_text

QUOTES_URLS = [
    "https://www.fantacalcio.it/quotazioni-fantacalcio/2026-27",
    "https://www.fantacalcio.it/quotazioni-fantacalcio/mantra",
]
STATS_URLS = {
    "26": "https://www.fantacalcio.it/statistiche-serie-a/2026-27/fantacalcio/riepilogo",
    "25": "https://www.fantacalcio.it/statistiche-serie-a/2025-26/fantacalcio/riepilogo",
    "24": "https://www.fantacalcio.it/statistiche-serie-a/2024-25/fantacalcio/riepilogo",
}
BUNDLED_LIST_PATH = Path(__file__).resolve().parents[1] / "data" / "Quotazioni_Fantacalcio_Stagione_2026_27.xlsx"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FantaMantraPortfolio/1.0; +personal-use)",
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
}


@dataclass
class DataHealth:
    source: str
    players: int
    roles_coverage: float
    stats_25_coverage: float
    warnings: list[str]


def _fetch(url: str, timeout: int = 18) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" ".join(str(x) for x in col if str(x) != "nan").strip() for col in df.columns]
    else:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_table(html: str, required_tokens: list[str]) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html), decimal=",", thousands=".")
    for table in tables:
        table = _flatten_columns(table)
        joined = " ".join(table.columns).lower()
        if all(token.lower() in joined for token in required_tokens):
            return table
    raise ValueError(f"Nessuna tabella compatibile trovata ({required_tokens}).")


def _pick_column(df: pd.DataFrame, tokens: list[str], occurrence: int = 0) -> str | None:
    matches = []
    for col in df.columns:
        lower = col.lower()
        if all(t.lower() in lower for t in tokens):
            matches.append(col)
    if not matches:
        return None
    return matches[min(occurrence, len(matches) - 1)]


def _extract_roles_from_html(html: str) -> dict[str, str]:
    """Best-effort parser for Mantra role badges embedded in the public table HTML."""
    soup = BeautifulSoup(html, "lxml")
    result: dict[str, str] = {}
    role_pattern = re.compile(r"(?<![A-Za-z])(Por|Dc|Dd|Ds|B|E|M|C|W|T|A|Pc)(?![A-Za-z])", re.I)

    for row in soup.find_all("tr"):
        # Player names are normally links in the quotation table.
        candidate_name = None
        for a in row.find_all("a"):
            text = " ".join(a.stripped_strings).strip()
            href = str(a.get("href") or "")
            if text and ("gioc" in href.lower() or "calci" in href.lower()):
                candidate_name = text
                break
        if not candidate_name:
            cells = row.find_all(["td", "th"])
            # On the rendered page the player name is commonly the longest non-numeric cell.
            text_cells = [" ".join(c.stripped_strings).strip() for c in cells]
            plausible = [x for x in text_cells if len(x) >= 3 and not re.fullmatch(r"[A-Z]{2,4}|[-+]?\d+(?:[,.]\d+)?", x)]
            candidate_name = plausible[0] if plausible else None
        if not candidate_name:
            continue

        attr_text = []
        for el in row.find_all(True):
            for attr in ("alt", "title", "aria-label", "data-role", "data-ruolo", "data-position", "class"):
                value = el.get(attr)
                if isinstance(value, list):
                    value = " ".join(map(str, value))
                if value:
                    attr_text.append(str(value))
        # Include compact visible role badges, but avoid the whole row because player names can contain A/C/etc.
        for cell in row.find_all(["td", "span", "div"]):
            txt = " ".join(cell.stripped_strings).strip()
            if 0 < len(txt) <= 10:
                attr_text.append(txt)

        roles: list[str] = []
        for chunk in attr_text:
            for match in role_pattern.findall(chunk):
                parsed = parse_roles(match)
                for role in parsed:
                    if role not in roles:
                        roles.append(role)
        if roles:
            result[normalize_name(candidate_name)] = "/".join(roles)
    return result


def _parse_quotes_html(html: str) -> pd.DataFrame:
    table = _find_table(html, ["calciatore", "fvm"])
    name_col = _pick_column(table, ["calciatore"]) or table.columns[0]
    team_col = _pick_column(table, ["sq"])
    fvm_cols = [c for c in table.columns if "fvm" in c.lower()]
    qa_cols = [c for c in table.columns if re.search(r"(^|\s)qa($|\s)", c.lower())]
    qi_cols = [c for c in table.columns if re.search(r"(^|\s)qi($|\s)", c.lower())]

    out = pd.DataFrame({
        "name": table[name_col].astype(str).str.strip(),
        "team": table[team_col].astype(str).str.strip() if team_col else "",
        "fvm": coerce_num(table[fvm_cols[-1]]) if fvm_cols else np.nan,
        "quote": coerce_num(table[qa_cols[-1]]) if qa_cols else np.nan,
        "quote_initial": coerce_num(table[qi_cols[-1]]) if qi_cols else np.nan,
    })
    roles_map = _extract_roles_from_html(html)
    out["name_norm"] = out["name"].map(normalize_name)
    out["roles"] = out["name_norm"].map(roles_map).fillna("")
    out = out[out["name"].str.len() > 1]
    out = out.drop_duplicates("name_norm", keep="first")
    return out


def _find_player_name_column(table: pd.DataFrame) -> str:
    candidate_cols = []
    for col in table.columns:
        s = table[col].dropna().astype(str).str.strip()
        if len(s) == 0:
            continue
        valid = s[s.str.match(r"^[A-Za-zÀ-ÿ\s\.\'\-]+$") & (s.str.len() >= 3)]
        if len(valid) >= len(table) * 0.4 and s.nunique() > 25:
            candidate_cols.append((col, len(valid)))
    if candidate_cols:
        return sorted(candidate_cols, key=lambda x: x[1], reverse=True)[0][0]
    return _pick_column(table, ["calciatore", "nome", "player"]) or table.columns[0]


def _parse_stats_html(html: str, suffix: str) -> pd.DataFrame:
    table = _find_table(html, ["calciatore", "pv", "fm"])
    name_col = _find_player_name_column(table)
    team_col = _pick_column(table, ["sq"])

    def c(token: str):
        return _pick_column(table, [token])

    cols = {
        "name": table[name_col].astype(str).str.strip(),
        f"team_{suffix}": table[team_col].astype(str).str.strip() if team_col else "",
        f"pv_{suffix}": coerce_num(table[c("pv")]) if c("pv") else np.nan,
        f"mv_{suffix}": coerce_num(table[c("mv")]) if c("mv") else np.nan,
        f"fm_{suffix}": coerce_num(table[c("fm")]) if c("fm") else np.nan,
        f"gol_{suffix}": coerce_num(table[c("gol")]) if c("gol") else np.nan,
        f"ass_{suffix}": coerce_num(table[c("ass")]) if c("ass") else np.nan,
    }
    out = pd.DataFrame(cols)
    out["name_norm"] = out["name"].map(normalize_name)
    out = out[out["name_norm"] != "nan"]
    return out.drop_duplicates("name_norm", keep="first")



def _load_remote() -> tuple[pd.DataFrame, DataHealth]:
    warnings: list[str] = []
    quote_html = None
    last_exc = None
    for url in QUOTES_URLS:
        try:
            quote_html = _fetch(url)
            quotes = _parse_quotes_html(quote_html)
            if len(quotes) >= 100:
                break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    else:
        raise RuntimeError(f"Impossibile leggere le quotazioni pubbliche: {last_exc}")

    base = quotes.copy()
    for suffix, url in STATS_URLS.items():
        try:
            stats = _parse_stats_html(_fetch(url), suffix)
            keep = [c for c in stats.columns if c != "name"]
            base = base.merge(stats[keep], on="name_norm", how="left")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Statistiche {suffix}: {type(exc).__name__}")

    role_cov = float((base["roles"].str.len() > 0).mean()) if len(base) else 0.0
    if role_cov < 0.80:
        warnings.append(
            "Copertura ruoli Mantra bassa dal parsing pubblico. Puoi caricare il listone ufficiale Excel/CSV dalla pagina Dati."
        )
    stat_cov = float(base.get("fm_25", pd.Series(index=base.index, dtype=float)).notna().mean()) if len(base) else 0.0
    health = DataHealth("fantacalcio.it (pagine pubbliche)", len(base), role_cov, stat_cov, warnings)
    return base.reset_index(drop=True), health


def _column_by_alias(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {re.sub(r"\s+", " ", str(c).strip().lower()): c for c in df.columns}
    for alias in aliases:
        key = alias.lower()
        if key in normalized:
            return normalized[key]
    for key, original in normalized.items():
        if any(alias.lower() in key for alias in aliases):
            return original
    return None


def _read_excel_with_detected_header(file_or_path) -> pd.DataFrame:
    """Read Fantacalcio's official workbook, whose real header is below a title row.

    We scan the first rows instead of hard-coding `header=1`, so this keeps working
    if Fantacalcio adds/removes a title line in a future export.
    """
    preview = pd.read_excel(file_or_path, sheet_name="Tutti", header=None, nrows=12)
    header_row = None
    for idx, row in preview.iterrows():
        vals = {str(v).strip().lower() for v in row.tolist() if pd.notna(v)}
        if "nome" in vals and ("fvm m" in vals or "fvm" in vals) and ("rm" in vals or "ruolo mantra" in vals):
            header_row = int(idx)
            break
    if header_row is None:
        # Generic Excel fallback for user-provided files that already have a normal header.
        return pd.read_excel(file_or_path)
    return pd.read_excel(file_or_path, sheet_name="Tutti", header=header_row)


def _normalize_listone(raw: pd.DataFrame) -> pd.DataFrame:
    raw = _flatten_columns(raw)
    name_col = _column_by_alias(raw, ["nome", "calciatore", "player"])
    team_col = _column_by_alias(raw, ["squadra", "sq", "team"])
    role_col = _column_by_alias(raw, ["rm", "ruolo mantra", "mantra", "roles"])
    fvm_col = _column_by_alias(raw, ["fvm m", "fvm mantra", "fvm / 1000", "fvm"])
    quote_col = _column_by_alias(raw, ["qt.a m", "qa m", "quotazione mantra", "qt.a", "qa"])
    quote_initial_col = _column_by_alias(raw, ["qt.i m", "qi m", "quotazione iniziale mantra", "qt.i", "qi"])
    id_col = _column_by_alias(raw, ["id"])

    if not name_col or not fvm_col:
        raise ValueError("Il file deve contenere almeno Nome/Calciatore e FVM M/FVM.")

    out = pd.DataFrame({
        "player_id": coerce_num(raw[id_col]).astype("Int64") if id_col else pd.Series(pd.NA, index=raw.index, dtype="Int64"),
        "name": raw[name_col].astype(str).str.strip(),
        "team": raw[team_col].astype(str).str.strip() if team_col else "",
        "roles": raw[role_col].map(roles_text) if role_col else "",
        "fvm": coerce_num(raw[fvm_col]),
        "quote": coerce_num(raw[quote_col]) if quote_col else np.nan,
        "quote_initial": coerce_num(raw[quote_initial_col]) if quote_initial_col else np.nan,
    })
    out["name_norm"] = out["name"].map(normalize_name)
    out = out[(out["name"].str.len() > 1) & out["fvm"].notna()]
    return out.drop_duplicates("name_norm", keep="first").reset_index(drop=True)


def _enrich_with_remote_stats(out: pd.DataFrame, timeout: int = 8) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    enriched = out.copy()
    for suffix, url in STATS_URLS.items():
        try:
            stats = _parse_stats_html(_fetch(url, timeout=timeout), suffix)
            keep = [c for c in stats.columns if c != "name"]
            enriched = enriched.merge(stats[keep], on="name_norm", how="left")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Storico {suffix} non disponibile: {type(exc).__name__}")
    return enriched, warnings


def load_uploaded(file: BinaryIO, filename: str, enrich_online: bool = True) -> tuple[pd.DataFrame, DataHealth]:
    filename = filename.lower()
    if filename.endswith((".xlsx", ".xls")):
        # UploadedFile/BytesIO must be rewound because header detection reads it twice.
        try:
            file.seek(0)
        except Exception:
            pass
        raw = _read_excel_with_detected_header(file)
    else:
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig", errors="replace")
        raw = pd.read_csv(io.StringIO(content), sep=None, engine="python")

    out = _normalize_listone(raw)
    warnings: list[str] = []
    if enrich_online:
        out, warnings = _enrich_with_remote_stats(out)

    role_cov = float((out["roles"].str.len() > 0).mean()) if len(out) else 0.0
    stat_cov = float(out.get("fm_25", pd.Series(index=out.index, dtype=float)).notna().mean()) if len(out) else 0.0
    source = "listone ufficiale caricato"
    if enrich_online:
        source += " + storico online"
    return out, DataHealth(source, len(out), role_cov, stat_cov, warnings)


def load_bundled_dataset(enrich_online: bool = True) -> tuple[pd.DataFrame, DataHealth]:
    """Primary V1 source: the official Mantra workbook shipped with the app."""
    if not BUNDLED_LIST_PATH.exists():
        raise FileNotFoundError(f"Listone incluso non trovato: {BUNDLED_LIST_PATH}")
    raw = _read_excel_with_detected_header(BUNDLED_LIST_PATH)
    out = _normalize_listone(raw)
    warnings: list[str] = []
    if enrich_online:
        out, warnings = _enrich_with_remote_stats(out)
    role_cov = float((out["roles"].str.len() > 0).mean()) if len(out) else 0.0
    stat_cov = float(out.get("fm_25", pd.Series(index=out.index, dtype=float)).notna().mean()) if len(out) else 0.0
    source = "listone ufficiale incluso (Fantacalcio 2026/27)"
    if enrich_online:
        source += " + storico online"
    return out, DataHealth(source, len(out), role_cov, stat_cov, warnings)


def load_remote_dataset() -> tuple[pd.DataFrame, DataHealth]:
    """Fallback only: scrape the public quotation page when no official workbook exists."""
    return _load_remote()
