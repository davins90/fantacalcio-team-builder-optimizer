import numpy as np
import pandas as pd
import pytest

from engine.advisor import analyze_auction, player_advice
from engine.config import AuctionConfig
from engine.tactics import MANTRA_FORMATIONS
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


def test_formations_do_not_double_count_players():
    """A player must fill at most one slot of the XI."""
    from engine.tactics import evaluate_formations

    def p(name, roles):
        return {"name": name, "roles": roles, "team": "X", "fvm": 10, "expected_fp": 100}

    # 11 players but only two midfielders, while 4-3-3 needs M/C x1 + C x2.
    short_midfield = pd.DataFrame([
        p("Por1", "Por"), p("Dd1", "Dd"), p("Dc1", "Dc"), p("Dc2", "Dc"), p("Ds1", "Ds"),
        p("C1", "C"), p("C2", "C"), p("W1", "W"), p("W2", "W"), p("Pc1", "Pc"), p("Dc3", "Dc"),
    ])
    f433 = next(f for f in evaluate_formations(short_midfield) if f["name"] == "4-3-3")
    assert not f433["playable"]
    assert f433["starters_covered"] == 10
    assert f433["missing"]

    # Adding a genuine third midfielder makes the XI fieldable.
    full = pd.DataFrame(short_midfield.to_dict("records")[:-1] + [p("M1", "M")])
    f433_ok = next(f for f in evaluate_formations(full) if f["name"] == "4-3-3")
    assert f433_ok["playable"]
    assert f433_ok["missing"] == []

    # Multi-role players must still be usable for whichever slot the XI needs.
    versatile = pd.DataFrame([
        p("Por1", "Por"), p("Dd1", "Dd"), p("Dc1", "Dc"), p("Dc2", "Dc"), p("Ds1", "Ds"),
        p("MC1", "M/C"), p("C1", "C"), p("C2", "C"), p("W1", "W"), p("W2", "W/A"), p("APc", "A/Pc"),
    ])
    assert next(f for f in evaluate_formations(versatile) if f["name"] == "4-3-3")["playable"]


def _squad_pool(n_per_role=6):
    """Synthetic pool wide enough to satisfy every squad and XI constraint."""
    from engine.projections import project_players
    rows = []
    roles = ["Por", "Dc", "Dd", "Ds", "E", "M", "C", "W", "T", "A", "Pc"]
    for r in roles:
        for k in range(n_per_role):
            rows.append({
                "name": f"{r}{k}", "name_norm": f"{r}{k}".lower(), "roles": r,
                "team": f"T{(len(rows)) % 12}", "fvm": 10 + 4 * k,
                "pv_25": 30, "fm_25": 6.0 + 0.25 * k, "pv_24": 30, "fm_24": 6.0,
            })
    return project_players(pd.DataFrame(rows))


def test_optimizer_builds_a_fieldable_eleven():
    from engine.market import dynamic_market_values
    from engine.optimizer import optimize_portfolio
    from engine.tactics import evaluate_formations

    cfg = AuctionConfig()
    pool = dynamic_market_values(_squad_pool(), [], pd.DataFrame(columns=["roles", "team"]), cfg)
    exposure, opt = optimize_portfolio(pool, [], cfg, simulations=2)

    assert opt.feasible
    assert len(opt.selected_names) == cfg.roster_size
    assert len(opt.starters) == 11
    assert opt.formation in MANTRA_FORMATIONS
    # Starters must be part of the squad, and the XI must really be fieldable.
    assert set(opt.starters).issubset(set(opt.selected_names))
    squad = pool[pool["name_norm"].isin(opt.selected_names)]
    assert next(f for f in evaluate_formations(squad) if f["key"] == opt.formation)["playable"]
    assert exposure["exposure"].between(0, 1).all()


def test_bench_weight_shifts_budget_towards_starters():
    """A lower bench weight must concentrate spending on fewer, better starters."""
    from engine.market import dynamic_market_values
    from engine.optimizer import optimize_portfolio

    pool_raw = _squad_pool(8)
    spend = {}
    for lam in (0.05, 0.60):
        cfg = AuctionConfig(bench_weight=lam)
        pool = dynamic_market_values(pool_raw, [], pd.DataFrame(columns=["roles", "team"]), cfg)
        _, opt = optimize_portfolio(pool, [], cfg, simulations=1)
        assert opt.feasible
        xi = pool[pool["name_norm"].isin(opt.starters)]
        spend[lam] = float(xi["dynamic_price"].sum())
    assert spend[0.05] > spend[0.60]


def test_optimizer_keeps_players_already_bought():
    from engine.market import dynamic_market_values
    from engine.optimizer import optimize_portfolio

    cfg = AuctionConfig()
    pool_raw = _squad_pool()
    pool = dynamic_market_values(pool_raw, [], pd.DataFrame(columns=["roles", "team"]), cfg)
    events = [{"name_norm": "pc0", "price": 30, "mine": True},
              {"name_norm": "dc0", "price": 12, "mine": False}]
    _, opt = optimize_portfolio(pool, events, cfg, simulations=1)
    assert opt.feasible
    assert "pc0" in opt.selected_names       # mio: resta in rosa
    assert "dc0" not in opt.selected_names   # venduto ad altri: fuori dal pool
    assert len(opt.selected_names) == cfg.roster_size
