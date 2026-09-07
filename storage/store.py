from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


class LocalStateStore:
    def __init__(self, path: str = ".state/fantamantra.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_state(self, state: dict) -> None:
        current = self.load_state()
        merged = dict(current)
        merged.update(state)
        self.path.write_text(json.dumps(merged, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


    def load_dataset(self) -> pd.DataFrame | None:
        state = self.load_state()
        rows = state.get("dataset")
        return pd.DataFrame(rows) if rows else None

    def save_dataset(self, df: pd.DataFrame) -> None:
        state = self.load_state()
        clean = df.astype(object).where(pd.notna(df), None)
        state["dataset"] = clean.to_dict("records")
        self.save_state(state)


class FirestoreStateStore:
    def __init__(self, project_id: str | None = None, database: str = "(default)", auction_id: str = "default"):
        from google.cloud import firestore

        kwargs = {"database": database}
        if project_id:
            kwargs["project"] = project_id
        self.client = firestore.Client(**kwargs)
        self.state_ref = self.client.collection("fantamantra").document(f"auction-{auction_id}")
        self.data_ref = self.client.collection("fantamantra").document("dataset-current")

    def load_state(self) -> dict:
        snap = self.state_ref.get()
        return snap.to_dict() if snap.exists else {}

    def save_state(self, state: dict) -> None:
        # Dataset is stored separately to keep the auction document small.
        compact = {k: v for k, v in state.items() if k != "dataset"}
        self.state_ref.set(compact)

    def load_dataset(self) -> pd.DataFrame | None:
        snap = self.data_ref.get()
        if not snap.exists:
            return None
        raw = snap.to_dict() or {}
        rows = raw.get("rows")
        return pd.DataFrame(rows) if rows else None

    def save_dataset(self, df: pd.DataFrame) -> None:
        clean = df.astype(object).where(pd.notna(df), None)
        rows = clean.to_dict("records")
        # 500-ish rows with compact columns remain under Firestore's 1 MiB document limit in normal use.
        self.data_ref.set({"rows": rows, "count": len(rows)})


def get_store():
    if os.getenv("USE_FIRESTORE", "0") == "1":
        try:
            return FirestoreStateStore(
                project_id=os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"),
                database=os.getenv("FIRESTORE_DATABASE", "(default)"),
                auction_id=os.getenv("AUCTION_ID", "default"),
            )
        except Exception:
            pass
    return LocalStateStore()
