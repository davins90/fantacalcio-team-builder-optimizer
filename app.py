from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from engine.advisor import analyze_auction, player_advice
from engine.config import AuctionConfig
from engine.data_loader import DataHealth, load_bundled_dataset, load_remote_dataset, load_uploaded
from engine.projections import compute_team_factors
from storage.store import get_store

st.set_page_config(page_title="FantaMantra Portfolio", page_icon="⚽", layout="wide")


def _auth_gate():
    pin = os.getenv("APP_PIN", "").strip()
    if not pin:
        return
    if st.session_state.get("authenticated"):
        return
    st.title("⚽ FantaMantra Portfolio")
    entered = st.text_input("PIN", type="password")
    if st.button("Entra"):
        if entered == pin:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("PIN non corretto")
    st.stop()


_auth_gate()
store = get_store()


def _default_state() -> dict:
    return {
        "config": AuctionConfig().to_dict(),
        "events": [],
        "manual_overweights": [],
        "data_health": None,
    }


if "bootstrapped" not in st.session_state:
    persisted = store.load_state() or {}
    state = _default_state()
    state.update({k: v for k, v in persisted.items() if k in state})
    st.session_state["app_state"] = state
    cached_df = store.load_dataset()
    if cached_df is not None and not cached_df.empty:
        st.session_state["players_raw"] = cached_df
    st.session_state["bootstrapped"] = True

state = st.session_state["app_state"]
config = AuctionConfig.from_dict(state.get("config"))
events: list[dict] = state.get("events", [])


def persist_state():
    store.save_state(st.session_state["app_state"])


def save_dataset(df: pd.DataFrame, health: DataHealth | None = None):
    st.session_state["players_raw"] = df
    store.save_dataset(df)
    if health:
        state["data_health"] = {
            "source": health.source,
            "players": health.players,
            "roles_coverage": health.roles_coverage,
            "stats_25_coverage": health.stats_25_coverage,
            "warnings": health.warnings,
        }
    persist_state()


def reset_auction():
    state["events"] = []
    persist_state()
    st.rerun()


st.sidebar.title("⚽ FantaMantra")
page = st.sidebar.radio("Sezione", ["Asta live", "Chi chiamo?", "La mia rosa", "Mercato", "Dati & setup", "Modello"])
st.sidebar.caption("Portfolio construction adattiva · Mantra")

players_raw = st.session_state.get("players_raw")

# Auto-load on first use if no persisted snapshot exists. The official Mantra workbook
# shipped in data/ is the primary source; public web pages only enrich its history.
if players_raw is None and page != "Dati & setup":
    with st.spinner("Carico il listone ufficiale incluso e provo ad arricchirlo con lo storico…"):
        try:
            bundled, health = load_bundled_dataset(enrich_online=True)
            save_dataset(bundled, health)
            players_raw = bundled
        except Exception as bundled_exc:
            # Last-resort fallback: the app can still bootstrap from public pages.
            try:
                remote, health = load_remote_dataset()
                health.warnings.insert(0, f"Listone incluso non disponibile: {type(bundled_exc).__name__}")
                save_dataset(remote, health)
                players_raw = remote
            except Exception as remote_exc:
                st.error(f"Non riesco a inizializzare i dati: {remote_exc}")
                st.info("Vai in **Dati & setup** e carica un listone Excel/CSV ufficiale.")
                st.stop()

analysis = None
if players_raw is not None and not players_raw.empty and page not in ["Dati & setup", "Modello"]:
    with st.spinner("Ricalcolo portafogli e stato dell'asta…"):
        analysis = analyze_auction(players_raw, events, config, state.get("manual_overweights", []))


if page == "Asta live":
    st.title("Asta live")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Budget", f"{analysis['my_budget_left']:.0f}/{config.starting_budget}")
    c2.metric("Miei giocatori", f"{len(analysis['my_roster'])}/{config.roster_size}")
    c3.metric("Giocatori venduti", len(events))
    c4.metric("Crediti mercato residui", f"{analysis['market_budget_left']:.0f}")

    available = analysis["available"].sort_values("name")
    if available.empty:
        st.success("Asta completata o nessun giocatore disponibile.")
    else:
        selected_label = st.selectbox(
            "Giocatore chiamato",
            options=available["name_norm"].tolist(),
            format_func=lambda x: f"{available.loc[available['name_norm'].eq(x), 'name'].iloc[0]} · {available.loc[available['name_norm'].eq(x), 'roles'].iloc[0]} · {available.loc[available['name_norm'].eq(x), 'team'].iloc[0]}",
            index=None,
            placeholder="Scrivi il nome del giocatore…",
        )
        if selected_label:
            adv = player_advice(analysis, selected_label)
            if adv:
                st.subheader(f"{adv['name']} · {adv['roles']} · {adv['team']}")
                a1, a2, a3, a4, a5 = st.columns(5)
                a1.metric("MAX BID", adv["max_bid"])
                a2.metric("Fair value", f"{adv['fair_value']:.1f}")
                a3.metric("Exposure", f"{adv['exposure']:.0%}")
                a4.metric("Expected FP", f"{adv['expected_fp']:.0f}")
                a5.metric("Risk", f"{adv['risk']:.0%}")

                if adv["decision"] == "BUY":
                    st.success(f"🟢 **BUY** — target importante nei portafogli ottimali. Non superare {adv['max_bid']}.")
                elif adv["decision"] == "VALUE":
                    st.info(f"🟡 **VALUE** — interessante al prezzo giusto. Stop consigliato: {adv['max_bid']}.")
                else:
                    st.warning(f"🔴 **DISCIPLINA** — bassa esposizione ottimale. Non inseguire oltre {adv['max_bid']}.")

                m1, m2, m3 = st.columns(3)
                m1.caption(f"Inflazione ruolo: {(adv['inflation']-1):+.1%}")
                m2.caption(f"Scarsità: {(adv['scarcity']-1):+.1%}")
                m3.caption(f"FM proiettata: {adv['projected_fm']:.2f}")

                st.markdown("#### Registra la vendita")
                with st.form("sale_form", clear_on_submit=True):
                    final_price = st.number_input("Prezzo finale", min_value=1, max_value=config.starting_budget, value=max(1, min(adv["max_bid"], config.starting_budget)), step=1)
                    mine = st.radio("Acquirente", ["ALTRI", "MIO"], horizontal=True)
                    submitted = st.form_submit_button("Registra acquisto", use_container_width=True)
                    if submitted:
                        if mine == "MIO" and final_price > analysis["my_budget_left"]:
                            st.error("Prezzo superiore al tuo budget residuo.")
                        else:
                            event = {
                                "name_norm": selected_label,
                                "name": adv["name"],
                                "price": int(final_price),
                                "mine": mine == "MIO",
                                "ts": datetime.now(timezone.utc).isoformat(),
                            }
                            state["events"].append(event)
                            persist_state()
                            st.rerun()

    if events:
        st.divider()
        left, right = st.columns([4, 1])
        left.subheader("Ultime vendite")
        hist = pd.DataFrame(events[-10:][::-1])
        if not hist.empty:
            left.dataframe(hist[["name", "price", "mine"]].rename(columns={"name": "Giocatore", "price": "Prezzo", "mine": "Mio"}), use_container_width=True, hide_index=True)
        if right.button("↩️ Annulla ultima"):
            state["events"] = state["events"][:-1]
            persist_state()
            st.rerun()


elif page == "Chi chiamo?":
    st.title("Chi chiamo?")
    st.caption("Il suggerimento non è un vincolo: puoi sempre chiamare liberamente chi vuoi dalla pagina Asta live.")
    targets = analysis["next_targets"].copy()
    targets["Max bid"] = targets["max_bid"].astype(int)
    targets["Exposure"] = (targets["exposure"] * 100).round(0).astype(int).astype(str) + "%"
    targets["Expected FP"] = targets["expected_fp"].round(0).astype(int)
    display = targets[["name", "team", "roles", "Max bid", "Exposure", "Expected FP"]].rename(columns={"name": "Giocatore", "team": "Squadra", "roles": "Ruoli"})
    st.markdown("### 🎯 Target consigliati")
    st.dataframe(display.head(10), use_container_width=True, hide_index=True)

    st.markdown("### 💸 Chiamate budget-drain")
    st.caption("Top costosi che il modello usa poco: utili per far emergere/spendere capitale agli altri, senza inseguirli.")
    drain = analysis["budget_drain"].copy()
    drain["Prezzo dinamico"] = drain["dynamic_price"].round(0).astype(int)
    drain["Exposure"] = (drain["exposure"] * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(drain[["name", "team", "roles", "Prezzo dinamico", "Exposure"]].rename(columns={"name":"Giocatore","team":"Squadra","roles":"Ruoli"}), use_container_width=True, hide_index=True)


elif page == "La mia rosa":
    st.title("La mia rosa")
    roster = analysis["my_roster"].copy()
    if roster.empty:
        st.info("Non hai ancora registrato acquisti tuoi.")
    else:
        ev = pd.DataFrame([e for e in events if e.get("mine")])[["name_norm", "price"]]
        roster = roster.merge(ev, on="name_norm", how="left")
        st.dataframe(roster[["name", "team", "roles", "price", "expected_fp", "risk"]].rename(columns={
            "name":"Giocatore", "team":"Squadra", "roles":"Ruoli", "price":"Prezzo", "expected_fp":"Expected FP", "risk":"Rischio"
        }), use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Speso", f"{analysis['my_spend']:.0f}")
        c2.metric("Budget residuo", f"{analysis['my_budget_left']:.0f}")
        c3.metric("Expected FP proxy", f"{roster['expected_fp'].sum():.0f}")
        st.markdown("### Esposizione per club")
        team_exp = roster.groupby("team").agg(Giocatori=("name", "count"), Expected_FP=("expected_fp", "sum")).sort_values("Giocatori", ascending=False)
        st.dataframe(team_exp, use_container_width=True)


elif page == "Mercato":
    st.title("Mercato dell'asta")
    rows = []
    for role, info in analysis["inflation"].items():
        if role == "ALL":
            continue
        rows.append({"Ruolo": role, "Osservazioni": info["n"], "Inflazione posterior": info["multiplier"] - 1})
    market_df = pd.DataFrame(rows)
    if market_df.empty:
        st.info("Servono alcune vendite prima di stimare il regime di mercato.")
    else:
        market_df["Inflazione"] = (market_df["Inflazione posterior"] * 100).round(1).astype(str) + "%"
        st.dataframe(market_df[["Ruolo", "Osservazioni", "Inflazione"]].sort_values("Osservazioni", ascending=False), use_container_width=True, hide_index=True)
    st.markdown("### Liquidità aggregata")
    spent_others = analysis["market_spend"] - analysis["my_spend"]
    avg_other_budget = (config.starting_budget * (config.participants - 1) - spent_others) / max(1, config.participants - 1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Spesa totale", f"{analysis['market_spend']:.0f}")
    c2.metric("Spesa altri", f"{spent_others:.0f}")
    c3.metric("Budget medio altri", f"{avg_other_budget:.1f}")


elif page == "Dati & setup":
    st.title("Dati & setup")
    st.markdown("### Configurazione lega")
    with st.form("config_form"):
        c1, c2, c3 = st.columns(3)
        participants = c1.number_input("Partecipanti", 4, 20, config.participants)
        starting_budget = c2.number_input("Crediti iniziali", 100, 2000, config.starting_budget, step=50)
        roster_size = c3.number_input("Giocatori per rosa", 18, 35, config.roster_size)
        c4, c5, c6 = st.columns(3)
        target_goalkeepers = c4.number_input("Portieri target", 2, 4, config.target_goalkeepers)
        max_team = c5.number_input("Max giocatori stesso club", 0, 8, config.max_players_per_real_team, help="0 = nessun limite")
        risk = c6.slider("Avversione al rischio", 0.0, 1.0, float(config.risk_aversion), 0.05)
        if players_raw is not None and "team" in players_raw:
            teams = sorted([x for x in players_raw["team"].dropna().astype(str).unique() if x and x != "nan"])
        else:
            teams = []
        overweights = st.multiselect("Club su cui vuoi fare una piccola scommessa (opzionale)", teams, default=[x for x in state.get("manual_overweights", []) if x in teams], max_selections=4)
        save_cfg = st.form_submit_button("Salva configurazione")
        if save_cfg:
            new_cfg = AuctionConfig(
                participants=int(participants), starting_budget=int(starting_budget), roster_size=int(roster_size),
                target_goalkeepers=int(target_goalkeepers), max_players_per_real_team=int(max_team),
                risk_aversion=float(risk), upside_weight=config.upside_weight, value_weight=config.value_weight,
                simulations=config.simulations, manual_team_premium=config.manual_team_premium,
            )
            state["config"] = new_cfg.to_dict()
            state["manual_overweights"] = overweights
            persist_state()
            st.success("Configurazione salvata.")
            st.rerun()

    st.divider()
    st.markdown("### Dataset")
    health = state.get("data_health")
    if players_raw is not None:
        h1, h2, h3 = st.columns(3)
        h1.metric("Giocatori", len(players_raw))
        h2.metric("Ruoli Mantra coperti", f"{(players_raw['roles'].fillna('').str.len()>0).mean():.0%}" if "roles" in players_raw else "—")
        h3.metric("Storico 25/26", f"{players_raw.get('fm_25', pd.Series(dtype=float)).notna().mean():.0%}" if "fm_25" in players_raw else "—")
        if health:
            st.caption(f"Fonte: {health.get('source')}")
            for warning in health.get("warnings", []):
                st.warning(warning)

    st.caption("Fonte primaria: **listone ufficiale Fantacalcio 2026/27 incluso nell'app**. Il web serve solo per arricchire lo storico; l'asta non dipende dal sito.")
    col_a, col_b = st.columns(2)
    if col_a.button("📦 Ricarica listone incluso + storico", use_container_width=True):
        with st.spinner("Leggo il listone ufficiale incluso e aggiorno lo storico…"):
            try:
                df, new_health = load_bundled_dataset(enrich_online=True)
                save_dataset(df, new_health)
                st.success(f"Caricati {len(df)} giocatori dal listone ufficiale incluso.")
                st.rerun()
            except Exception as exc:
                st.error(f"Aggiornamento fallito: {exc}")

    uploaded = col_b.file_uploader("Sostituisci listone: Excel/CSV ufficiale più recente", type=["xlsx", "xls", "csv"])
    if uploaded is not None:
        try:
            df, new_health = load_uploaded(uploaded, uploaded.name, enrich_online=True)
            save_dataset(df, new_health)
            st.success(f"Caricati {len(df)} giocatori dal file e aggiornato lo storico disponibile.")
            st.rerun()
        except Exception as exc:
            st.error(f"File non leggibile: {exc}")

    with st.expander("Fallback web"):
        st.caption("Usalo solo se vuoi ricostruire il listone dalle pagine pubbliche invece dell'Excel incluso.")
        if st.button("🌐 Ricostruisci dalle pagine pubbliche", use_container_width=True):
            with st.spinner("Scarico quotazioni e storico dalle pagine pubbliche…"):
                try:
                    df, new_health = load_remote_dataset()
                    save_dataset(df, new_health)
                    st.success(f"Ricostruiti {len(df)} giocatori dalle pagine pubbliche.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Fallback web fallito: {exc}")

    if players_raw is not None:
        st.markdown("### Team factor (posterior iniziale)")
        factors = compute_team_factors(players_raw).sort_values("team_factor", ascending=False)
        factors["Segnale"] = factors["team_factor"].round(3)
        st.dataframe(factors[["team", "Segnale", "team_evidence"]].rename(columns={"team":"Squadra","team_evidence":"Evidenza"}), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Gestione asta")
    if st.button("🗑️ Reset asta", type="secondary"):
        reset_auction()


elif page == "Modello":
    st.title("Come ragiona il modello")
    st.markdown(
        """
**1. Projection engine.** Usa storico recente, prime evidenze 2026/27 con forte shrinkage, FVM come prior di mercato e un piccolo team factor. Produce fantamedia proiettata, disponibilità, expected FP, rischio e upside.

**2. Market engine.** Converte l'FVM su 500 crediti e, dopo ogni vendita, aggiorna un moltiplicatore di inflazione per ruolo con shrinkage verso 1. Aggiunge scarsità e liquidità aggregata.

**3. Portfolio optimizer.** Risolve un problema di ottimizzazione intera con budget, numero giocatori, coperture Mantra e diversificazione per club. Più simulazioni perturbano le proiezioni per ottenere la **portfolio exposure** di ogni giocatore.

**4. Live advisor.** Combina fair value, fit della tua rosa ed exposure. Dopo ogni evento `giocatore + prezzo + MIO/ALTRI` il pool cambia e l'intero portafoglio viene ricalcolato.

La V1 è un **decision-support model**, non una previsione certa. In particolare, senza tracciare le nove rose avversarie non conosce la distribuzione esatta della liquidità né chi abbia ancora bisogno di uno specifico ruolo.
        """
    )
