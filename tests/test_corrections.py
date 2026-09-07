import numpy as np
import pandas as pd
import pytest

from engine.advisor import analyze_auction, player_advice
from engine.config import AuctionConfig
from engine.data_loader import load_bundled_dataset
from engine.projections import project_players
from engine.utils import coerce_num, has_any_role
from storage.store import LocalStateStore


def test_coerce_num_preserves_floats_and_formats():
    # Numeric float series
    s_float = pd.Series([6.75, 5.5, 8.0])
    res_float = coerce_num(s_float)
    assert res_float.iloc[0] == 6.75
    assert res_float.iloc[1] == 5.5
    assert res_float.iloc[2] == 8.0

    # String with comma decimal
    s_comma = pd.Series(["6,75", "5,5", "-1,25"])
    res_comma = coerce_num(s_comma)
    assert res_comma.iloc[0] == 6.75
    assert res_comma.iloc[1] == 5.5
    assert res_comma.iloc[2] == -1.25

    # String with dot decimal and thousands
    s_mixed = pd.Series(["1.000", "6.75", "12.0", "1.250,50"])
    res_mixed = coerce_num(s_mixed)
    assert res_mixed.iloc[0] == 1000.0
    assert res_mixed.iloc[1] == 6.75
    assert res_mixed.iloc[2] == 12.0
    assert res_mixed.iloc[3] == 1250.50


def test_projections_do_not_overvalue_low_fvm_reserves():
    df, _ = load_bundled_dataset(enrich_online=False)
    proj = project_players(df)
    reserves = proj[proj["fvm"] <= 1]
    assert not reserves.empty
    # A player with FVM=1 must have very few expected appearances (1-4), not 27!
    assert reserves["expected_apps"].max() < 5.0
    # Expected FP must be low (~10-25), not 150!
    assert reserves["expected_fp"].max() < 30.0


def test_optimizer_strictly_respects_goalkeeper_target():
    df, _ = load_bundled_dataset(enrich_online=False)
    cfg = AuctionConfig(target_goalkeepers=3, roster_size=25, simulations=2)
    analysis = analyze_auction(df, [], cfg)
    opt = analysis["optimizer"]
    assert opt.feasible
    selected = df[df["name_norm"].isin(opt.selected_names)]
    gk_count = sum(has_any_role(r, {"Por"}) for r in selected["roles"])
    assert gk_count == 3


def test_optimizer_includes_center_forwards_pc():
    df, _ = load_bundled_dataset(enrich_online=False)
    cfg = AuctionConfig(target_goalkeepers=3, roster_size=25, simulations=2)
    analysis = analyze_auction(df, [], cfg)
    opt = analysis["optimizer"]
    selected = df[df["name_norm"].isin(opt.selected_names)]
    pc_count = sum(has_any_role(r, {"Pc"}) for r in selected["roles"])
    assert pc_count >= 2


def test_max_bid_is_capped_by_legal_budget_ceiling():
    df, _ = load_bundled_dataset(enrich_online=False)
    # Total budget = 500. Spent = 485. Remaining budget = 15.
    # 20 players bought, 5 slots needed.
    # Max legal bid = 15 - (5 - 1) = 11.
    events = []
    for i in range(19):
        events.append({
            "name_norm": df.iloc[i]["name_norm"],
            "name": df.iloc[i]["name"],
            "price": 24,
            "mine": True,
        })
    events.append({
        "name_norm": df.iloc[19]["name_norm"],
        "name": df.iloc[19]["name"],
        "price": 29,  # 19 * 24 + 29 = 485
        "mine": True,
    })
    cfg = AuctionConfig(starting_budget=500, roster_size=25, simulations=1)
    analysis = analyze_auction(df, events, cfg)
    assert analysis["my_budget_left"] == 15
    assert analysis["slots_needed"] == 5
    assert analysis["max_possible_bid"] == 11
    assert (analysis["players"]["max_bid"] <= 11).all()


def test_max_bid_is_zero_when_roster_is_full():
    df, _ = load_bundled_dataset(enrich_online=False)
    events = []
    for i in range(25):
        events.append({
            "name_norm": df.iloc[i]["name_norm"],
            "name": df.iloc[i]["name"],
            "price": 10,
            "mine": True,
        })
    cfg = AuctionConfig(starting_budget=500, roster_size=25, simulations=1)
    analysis = analyze_auction(df, events, cfg)
    assert analysis["slots_needed"] == 0
    assert (analysis["players"]["max_bid"] == 0).all()


def test_local_state_store_preserves_dataset(tmp_path):
    store = LocalStateStore(str(tmp_path / "state.json"))
    sample_df = pd.DataFrame([{"name": "test", "fvm": 100}])
    store.save_dataset(sample_df)
    store.save_state({"config": {"participants": 10}, "events": []})
    loaded_df = store.load_dataset()
    assert loaded_df is not None
    assert len(loaded_df) == 1
    assert loaded_df.iloc[0]["name"] == "test"


def test_tactics_depth_and_formations():
    from engine.tactics import compute_roster_depth, evaluate_formations
    roster = pd.DataFrame([
        {"name": "Svilar", "roles": "Por", "team": "Roma", "fvm": 80, "expected_fp": 140},
        {"name": "Bastoni", "roles": "Dc", "team": "Inter", "fvm": 50, "expected_fp": 120},
        {"name": "Di Lorenzo", "roles": "Dd/E", "team": "Napoli", "fvm": 40, "expected_fp": 110},
        {"name": "Tavares", "roles": "Ds/E", "team": "Lazio", "fvm": 30, "expected_fp": 90},
        {"name": "Calhanoglu", "roles": "M/C", "team": "Inter", "fvm": 70, "expected_fp": 150},
        {"name": "Samardzic", "roles": "C/T", "team": "Atalanta", "fvm": 50, "expected_fp": 120},
        {"name": "Politano", "roles": "W", "team": "Napoli", "fvm": 40, "expected_fp": 110},
        {"name": "Zaccagni", "roles": "W/A", "team": "Lazio", "fvm": 60, "expected_fp": 130},
        {"name": "Kean", "roles": "Pc", "team": "Fiorentina", "fvm": 70, "expected_fp": 140},
    ])
    depth = compute_roster_depth(roster)
    assert len(depth["Por"]) == 1
    assert len(depth["Dd"]) == 1
    assert len(depth["Ds"]) == 1
    assert len(depth["Pc"]) == 1

    fmts = evaluate_formations(roster)
    assert len(fmts) >= 5
    assert any("4-3-3" in f["name"] for f in fmts)
    # Check that missing positions are properly flagged when squad has only 1 Dc
    f433 = next(f for f in fmts if f["name"] == "4-3-3")
    assert any("Dc" in m for m in f433["missing"])

