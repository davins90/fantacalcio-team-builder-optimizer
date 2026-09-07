import io
from pathlib import Path

import pandas as pd

from engine.data_loader import BUNDLED_LIST_PATH, load_uploaded


def test_upload_mapping_without_network(monkeypatch):
    monkeypatch.setattr("engine.data_loader._fetch", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    raw = pd.DataFrame({
        "Nome": ["Alpha", "Beta"],
        "Squadra": ["AAA", "BBB"],
        "RM": ["Pc", "M/C"],
        "FVM M": [200, 80],
        "Qt.A M": [25, 12],
    })
    data = raw.to_csv(index=False).encode()
    out, health = load_uploaded(io.BytesIO(data), "listone.csv")
    assert len(out) == 2
    assert out.loc[0, "roles"] == "Pc"
    assert out.loc[1, "roles"] == "M/C"
    assert health.roles_coverage == 1.0


def test_official_excel_upload_detects_title_row():
    with Path(BUNDLED_LIST_PATH).open("rb") as f:
        payload = io.BytesIO(f.read())
    out, health = load_uploaded(payload, "Quotazioni_Fantacalcio_Stagione_2026_27.xlsx", enrich_online=False)
    assert len(out) >= 500
    assert health.roles_coverage > 0.99
    assert int(out.loc[out["name_norm"].eq("calhanoglu"), "fvm"].iloc[0]) == 273
