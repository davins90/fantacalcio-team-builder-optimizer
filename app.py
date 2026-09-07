from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from engine.advisor import analyze_auction, player_advice
from engine.config import AuctionConfig
from engine.data_loader import DataHealth, load_bundled_dataset, load_remote_dataset, load_uploaded
from engine.projections import compute_team_factors, project_players
from engine.tactics import compute_roster_depth, evaluate_formations
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

@st.cache_data(show_spinner=False)
def _get_cached_projections(df: pd.DataFrame, overweights: tuple[str, ...], premium: float) -> pd.DataFrame:
    return project_players(df, list(overweights), premium)


analysis = None
if players_raw is not None and not players_raw.empty and page not in ["Dati & setup", "Modello"]:
    with st.spinner("Ricalcolo portafogli e stato dell'asta…"):
        projected_cached = _get_cached_projections(
            players_raw, tuple(state.get("manual_overweights", [])), config.manual_team_premium
        )
        analysis = analyze_auction(projected_cached, events, config, state.get("manual_overweights", []))



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
        f_col1, f_col2 = st.columns([3, 1])
        role_filter = f_col1.radio(
            "Filtro rapido reparto:",
            ["TUTTI", "🧤 Portieri", "🛡️ Difesa (D/E)", "⚙️ Centro (M/C)", "🎯 Attacco (W/T/A/Pc)"],
            horizontal=True,
            index=0,
        )
        all_teams = sorted([t for t in available["team"].dropna().unique() if t and t != "nan"])
        team_filter = f_col2.selectbox("Filtra squadra:", ["TUTTE"] + all_teams, index=0)

        filtered_available = available.copy()
        if role_filter == "🧤 Portieri":
            filtered_available = filtered_available[filtered_available["roles"].str.contains(r"\bPor\b")]
        elif role_filter == "🛡️ Difesa (D/E)":
            filtered_available = filtered_available[filtered_available["roles"].str.contains(r"\b(?:Dc|Dd|Ds|B|E)\b")]
        elif role_filter == "⚙️ Centro (M/C)":
            filtered_available = filtered_available[filtered_available["roles"].str.contains(r"\b(?:M|C)\b")]
        elif role_filter == "🎯 Attacco (W/T/A/Pc)":
            filtered_available = filtered_available[filtered_available["roles"].str.contains(r"\b(?:W|T|A|Pc)\b")]

        if team_filter != "TUTTE":
            filtered_available = filtered_available[filtered_available["team"] == team_filter]

        if filtered_available.empty:
            st.info("Nessun giocatore disponibile con i filtri selezionati.")
            selected_label = None
        else:
            selected_label = st.selectbox(
                f"Giocatore chiamato ({len(filtered_available)} disponibili)",
                options=filtered_available["name_norm"].tolist(),
                format_func=lambda x: f"{filtered_available.loc[filtered_available['name_norm'].eq(x), 'name'].iloc[0]} · {filtered_available.loc[filtered_available['name_norm'].eq(x), 'roles'].iloc[0]} · {filtered_available.loc[filtered_available['name_norm'].eq(x), 'team'].iloc[0]}",
                index=None,
                placeholder="Scrivi o seleziona il nome del giocatore…",
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

                if adv["decision"] == "ROSTER_FULL":
                    st.error("🚫 **ROSA COMPLETA** — Hai già completato tutti i posti disponibili in rosa.")
                elif adv["decision"] == "OUT_OF_BUDGET":
                    st.error("⚠️ **BUDGET INSUFFICIENTE** — Devi conservare almeno 1 credito per ogni slot ancora vuoto.")
                elif adv["decision"] == "BUY":
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
                q_cols = st.columns(3)
                if q_cols[0].button("🏷️ Prezzo = 1", use_container_width=True):
                    st.session_state["quick_val"] = 1
                if q_cols[1].button(f"🎯 Al Max Bid ({adv['max_bid']})", use_container_width=True):
                    st.session_state["quick_val"] = max(1, adv["max_bid"])
                if q_cols[2].button(f"📊 Al Fair Value ({adv['fair_value']:.0f})", use_container_width=True):
                    st.session_state["quick_val"] = max(1, int(round(adv["fair_value"])))

                current_default = st.session_state.pop("quick_val", None)
                if current_default is None:
                    current_default = max(1, min(adv["max_bid"] or 1, config.starting_budget))

                with st.form("sale_form", clear_on_submit=True):
                    final_price = st.number_input("Prezzo finale", min_value=1, max_value=config.starting_budget, value=current_default, step=1)
                    mine = st.radio("Acquirente", ["ALTRI", "MIO"], horizontal=True)
                    submitted = st.form_submit_button("Registra acquisto", use_container_width=True)

                    if submitted:
                        slots_left = config.roster_size - len(analysis["my_roster"])
                        if mine == "MIO" and slots_left <= 0:
                            st.error("La tua rosa ha già raggiunto il limite massimo di giocatori!")
                        elif mine == "MIO" and final_price > analysis["my_budget_left"]:
                            st.error(f"Prezzo ({final_price}) superiore al tuo budget residuo ({analysis['my_budget_left']:.0f}).")
                        elif mine == "MIO" and final_price > (analysis["my_budget_left"] - (slots_left - 1)):
                            max_bid_legal = max(0, int(analysis["my_budget_left"] - (slots_left - 1)))
                            st.error(
                                f"Prezzo non valido: devi tenere almeno 1 credito per ciascuno dei restanti "
                                f"{slots_left - 1} slot (massimo consentito: {max_bid_legal} crediti)."
                            )
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
        st.info("Non hai ancora registrato acquisti tuoi. Quando compri un giocatore nell'Asta live selezionando 'MIO', apparirà qui.")
    else:
        ev = pd.DataFrame([e for e in events if e.get("mine")])[["name_norm", "price"]]
        roster = roster.merge(ev, on="name_norm", how="left")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Giocatori", f"{len(roster)}/{config.roster_size}")
        c2.metric("Speso", f"{analysis['my_spend']:.0f}")
        c3.metric("Budget residuo", f"{analysis['my_budget_left']:.0f}")
        c4.metric("Expected FP totali", f"{roster['expected_fp'].sum():.0f}")

        st.markdown("### 📐 Compatibilità Moduli Ufficiali Mantra")
        fmts = evaluate_formations(roster)
        f_cols = st.columns(min(4, len(fmts)))
        for idx in range(min(4, len(fmts))):
            fmt = fmts[idx]
            with f_cols[idx]:
                icon = "🟢" if fmt["playable"] else "🟡" if fmt["score"] >= 65 else "🔴"
                st.metric(label=f"{icon} {fmt['name']}", value=f"{fmt['score']}%")
                if fmt["playable"]:
                    st.success("Titolari coperti!")
                else:
                    st.caption("Mancano: " + ", ".join(fmt["missing"][:2]))

        with st.expander("🔍 Dettaglio di tutti i moduli Mantra"):
            for fmt in fmts:
                status_icon = "🟢" if fmt["playable"] else "🟡" if fmt["score"] >= 65 else "🔴"
                st.markdown(f"**{status_icon} {fmt['name']}** — *{fmt['description']}* (Copertura: **{fmt['score']}%**)")
                if fmt["missing"]:
                    st.caption("Ruoli ancora scoperti per gli 11 titolari: " + "; ".join(fmt["missing"]))

        st.divider()
        st.markdown("### 📋 Depth Chart Ruoli Mantra")
        depth = compute_roster_depth(roster)
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1:
            st.markdown(f"**🧤 Portieri ({len(depth['Por'])}/3)**")
            for p in depth["Por"]:
                st.caption(f"• {p['name']} ({p['team']})")
            if len(depth["Por"]) < 3:
                st.warning(f"⚠️ Mancano {3 - len(depth['Por'])}")

        with d2:
            st.markdown(f"**🛡️ Centrali ({len(depth['Dc'])})**")
            for p in depth["Dc"]:
                st.caption(f"• {p['name']} ({p['team']})")
            if len(depth["Dc"]) < 4:
                st.warning(f"⚠️ Min. 4 (hai {len(depth['Dc'])})")

        with d3:
            st.markdown("**⚡ Fasce Esterne**")
            st.caption(f"**Dd ({len(depth['Dd'])}):** " + (", ".join([p["name"] for p in depth["Dd"]]) if depth["Dd"] else "⚠️ Nessuno"))
            st.caption(f"**Ds ({len(depth['Ds'])}):** " + (", ".join([p["name"] for p in depth["Ds"]]) if depth["Ds"] else "⚠️ Nessuno"))
            st.caption(f"**E ({len(depth['E'])}):** " + (", ".join([p["name"] for p in depth["E"]]) if depth["E"] else "—"))

        with d4:
            st.markdown("**⚙️ Centrocampo**")
            st.caption(f"**M ({len(depth['M'])}):** " + (", ".join([p["name"] for p in depth["M"]]) if depth["M"] else "⚠️ Nessun M"))
            st.caption(f"**C ({len(depth['C'])}):** " + (", ".join([p["name"] for p in depth["C"]]) if depth["C"] else "—"))

        with d5:
            st.markdown("**🎯 Attacco**")
            st.caption(f"**Pc ({len(depth['Pc'])}):** " + (", ".join([p["name"] for p in depth["Pc"]]) if depth["Pc"] else "🚨 Nessuna punta!"))
            st.caption(f"**T ({len(depth['T'])}):** " + (", ".join([p["name"] for p in depth["T"]]) if depth["T"] else "—"))
            st.caption(f"**A ({len(depth['A'])}):** " + (", ".join([p["name"] for p in depth["A"]]) if depth["A"] else "—"))
            st.caption(f"**W ({len(depth['W'])}):** " + (", ".join([p["name"] for p in depth["W"]]) if depth["W"] else "—"))

        st.divider()
        st.markdown("### 👥 Tutti i Giocatori Acquistati")
        st.dataframe(roster[["name", "team", "roles", "price", "expected_fp", "risk"]].rename(columns={
            "name":"Giocatore", "team":"Squadra", "roles":"Ruoli", "price":"Prezzo", "expected_fp":"Expected FP", "risk":"Rischio"
        }), use_container_width=True, hide_index=True)

        st.markdown("### 🏟️ Esposizione per club")
        team_exp = roster.groupby("team").agg(Giocatori=("name", "count"), Expected_FP=("expected_fp", "sum")).sort_values("Giocatori", ascending=False)
        st.dataframe(team_exp, use_container_width=True)

        csv_roster = roster[["name", "team", "roles", "price", "expected_fp", "risk"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Scarica la mia rosa (.csv)",
            data=csv_roster,
            file_name=f"la_mia_rosa_mantra_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


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

    st.markdown("### 💰 Liquidità aggregata")
    spent_others = analysis["market_spend"] - analysis["my_spend"]
    avg_other_budget = (config.starting_budget * (config.participants - 1) - spent_others) / max(1, config.participants - 1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Spesa totale lega", f"{analysis['market_spend']:.0f}")
    c2.metric("Spesa altri avversari", f"{spent_others:.0f}")
    c3.metric("Budget medio avversari", f"{avg_other_budget:.1f}")

    if events:
        st.divider()
        st.markdown("### 📊 Spesa di Lega per Reparto")
        ev_df = pd.DataFrame(events)
        lookup_roles = players_raw.set_index("name_norm")["roles"].to_dict() if players_raw is not None else {}
        def get_reparto(name_norm):
            r = str(lookup_roles.get(name_norm, ""))
            if "Por" in r: return "🧤 Portieri"
            if any(x in r for x in ["Pc", "A", "T", "W"]): return "🎯 Attacco/Trequarti"
            if any(x in r for x in ["M", "C"]): return "⚙️ Centrocampo"
            return "🛡️ Difesa"
        ev_df["Reparto"] = ev_df["name_norm"].map(get_reparto)
        rep_summary = ev_df.groupby("Reparto").agg(
            Acquisti=("price", "count"),
            Spesa_Totale=("price", "sum"),
            Prezzo_Medio=("price", "mean"),
            Prezzo_Max=("price", "max")
        ).reset_index()
        rep_summary["Prezzo_Medio"] = rep_summary["Prezzo_Medio"].round(1)
        rep_summary["% Spesa Lega"] = (rep_summary["Spesa_Totale"] / max(1, analysis["market_spend"]) * 100).round(1).astype(str) + "%"
        st.dataframe(rep_summary, use_container_width=True, hide_index=True)

        c_top1, c_top2 = st.columns(2)
        with c_top1:
            st.markdown("### 💎 Top 5 Acquisti Più Cari dell'Asta")
            top_sales = ev_df.sort_values("price", ascending=False).head(5)
            st.dataframe(top_sales[["name", "price", "mine"]].rename(columns={"name": "Giocatore", "price": "Prezzo", "mine": "Mio"}), use_container_width=True, hide_index=True)
        with c_top2:
            st.markdown("### 📥 Esporta Dati Asta")
            st.caption("Scarica il report di tutti i giocatori chiamati finora con acquirente e prezzo battuto.")
            st.download_button(
                "Scarica storico completo vendite (.csv)",
                data=ev_df.to_csv(index=False).encode("utf-8"),
                file_name=f"storico_aste_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )



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
