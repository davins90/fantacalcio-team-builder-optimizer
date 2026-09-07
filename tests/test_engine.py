import pandas as pd

from engine.advisor import analyze_auction, player_advice
from engine.config import AuctionConfig
from engine.projections import project_players


def sample_players():
    rows = []
    role_cycle = ["Por", "Dc", "Dd/E", "Ds/E", "M/C", "C/T", "W/A", "T/A", "A/Pc", "Pc"]
    for i in range(80):
        rows.append({
            "name": f"Player {i}",
            "name_norm": f"player {i}",
            "team": f"T{i%20:02d}",
            "roles": role_cycle[i % len(role_cycle)],
            "fvm": max(5, 260 - i * 3),
            "quote": max(1, 30 - i // 3),
            "pv_26": i % 3,
            "fm_26": 6.0 + (i % 8) * 0.2,
            "pv_25": 18 + (i % 18),
            "fm_25": 5.8 + (i % 10) * 0.18,
            "pv_24": 16 + (i % 20),
            "fm_24": 5.7 + (i % 9) * 0.17,
        })
    return pd.DataFrame(rows)


def test_projection_columns():
    out = project_players(sample_players())
    assert {"expected_fp", "risk", "upside", "projected_fm"}.issubset(out.columns)
    assert out["expected_fp"].notna().all()


def test_live_analysis_reacts_to_event():
    cfg = AuctionConfig(roster_size=20, target_goalkeepers=2, simulations=2, max_players_per_real_team=0)
    df = sample_players()
    before = analyze_auction(df, [], cfg)
    target = before["available"].iloc[10]
    events = [{"name_norm": target["name_norm"], "name": target["name"], "price": 50, "mine": False}]
    after = analyze_auction(df, events, cfg)
    assert target["name_norm"] not in set(after["available"]["name_norm"])
    assert after["market_budget_left"] == cfg.total_market_budget - 50


def test_my_purchase_reduces_budget_and_advice_exists():
    cfg = AuctionConfig(roster_size=20, target_goalkeepers=2, simulations=2, max_players_per_real_team=0)
    df = sample_players()
    events = [{"name_norm": "player 0", "name": "Player 0", "price": 15, "mine": True}]
    analysis = analyze_auction(df, events, cfg)
    assert analysis["my_budget_left"] == cfg.starting_budget - 15
    candidate = analysis["available"].iloc[0]["name_norm"]
    advice = player_advice(analysis, candidate)
    assert advice is not None
    assert advice["max_bid"] >= 1
