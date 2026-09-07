from __future__ import annotations

from typing import Iterable
import pandas as pd
from .utils import parse_roles, has_any_role

MANTRA_FORMATIONS = {
    "4-3-3": {
        "name": "4-3-3",
        "description": "Dd - 2 Dc - Ds | M/C - 2 C | 2 W/A - Pc",
        "def_line": 4,
        "needs": [
            ({"Por"}, 1, "Por"),
            ({"Dd"}, 1, "Dd"),
            ({"Dc"}, 2, "Dc"),
            ({"Ds"}, 1, "Ds"),
            ({"M", "C"}, 1, "M/C"),
            ({"C"}, 2, "C"),
            ({"W", "A"}, 2, "W/A"),
            ({"Pc"}, 1, "Pc"),
        ],
    },
    "4-2-3-1": {
        "name": "4-2-3-1",
        "description": "Dd - 2 Dc - Ds | 2 M/C (almeno 1 M) | 3 W/T/A | Pc",
        "def_line": 4,
        "needs": [
            ({"Por"}, 1, "Por"),
            ({"Dd"}, 1, "Dd"),
            ({"Dc"}, 2, "Dc"),
            ({"Ds"}, 1, "Ds"),
            ({"M"}, 1, "M"),
            ({"M", "C"}, 1, "M/C"),
            ({"W", "T", "A"}, 3, "W/T/A"),
            ({"Pc"}, 1, "Pc"),
        ],
    },
    "3-4-2-1": {
        "name": "3-4-2-1",
        "description": "3 Dc | 2 E - 2 M/C | 2 T/A | Pc",
        "def_line": 3,
        "needs": [
            ({"Por"}, 1, "Por"),
            ({"Dc"}, 3, "Dc"),
            ({"E"}, 2, "E"),
            ({"M", "C"}, 2, "M/C"),
            ({"T", "A"}, 2, "T/A"),
            ({"Pc"}, 1, "Pc"),
        ],
    },
    "3-5-2": {
        "name": "3-5-2",
        "description": "3 Dc | 2 E - 1 M - 2 C | 2 Pc/A (almeno 1 Pc)",
        "def_line": 3,
        "needs": [
            ({"Por"}, 1, "Por"),
            ({"Dc"}, 3, "Dc"),
            ({"E"}, 2, "E"),
            ({"M"}, 1, "M"),
            ({"C"}, 2, "C"),
            ({"Pc"}, 1, "Pc"),
            ({"A", "Pc"}, 1, "A/Pc"),
        ],
    },
    "4-3-1-2": {
        "name": "4-3-1-2",
        "description": "Dd - 2 Dc - Ds | 1 M - 2 C | 1 T | 2 Pc/A",
        "def_line": 4,
        "needs": [
            ({"Por"}, 1, "Por"),
            ({"Dd"}, 1, "Dd"),
            ({"Dc"}, 2, "Dc"),
            ({"Ds"}, 1, "Ds"),
            ({"M"}, 1, "M"),
            ({"C"}, 2, "C"),
            ({"T"}, 1, "T"),
            ({"Pc"}, 1, "Pc"),
            ({"A", "Pc"}, 1, "A/Pc"),
        ],
    },
    "3-4-3": {
        "name": "3-4-3",
        "description": "3 Dc | 2 E - 2 M/C | 2 W/A - Pc",
        "def_line": 3,
        "needs": [
            ({"Por"}, 1, "Por"),
            ({"Dc"}, 3, "Dc"),
            ({"E"}, 2, "E"),
            ({"M", "C"}, 2, "M/C"),
            ({"W", "A"}, 2, "W/A"),
            ({"Pc"}, 1, "Pc"),
        ],
    },
    "4-4-2": {
        "name": "4-4-2",
        "description": "Dd - 2 Dc - Ds | 2 E/W - 2 M/C | 2 Pc/A",
        "def_line": 4,
        "needs": [
            ({"Por"}, 1, "Por"),
            ({"Dd"}, 1, "Dd"),
            ({"Dc"}, 2, "Dc"),
            ({"Ds"}, 1, "Ds"),
            ({"E", "W"}, 2, "E/W"),
            ({"M", "C"}, 2, "M/C"),
            ({"Pc"}, 1, "Pc"),
            ({"A", "Pc"}, 1, "A/Pc"),
        ],
    },
}


def _max_assignment(player_roles: list[set[str]], slots: list[tuple[set[str], str]]) -> list[int | None]:
    """Maximum bipartite matching of players to XI slots (Kuhn's augmenting paths).

    Returns, for each player, the index of the slot they fill or None. Sizes here are
    tiny (<= 25 players, 11 slots), so the O(V*E) formulation is more than fast enough.
    """
    slot_to_player: list[int | None] = [None] * len(slots)

    def _augment(player: int, seen: list[bool]) -> bool:
        for s_idx, (slot_roles, _) in enumerate(slots):
            if seen[s_idx] or not player_roles[player].intersection(slot_roles):
                continue
            seen[s_idx] = True
            holder = slot_to_player[s_idx]
            if holder is None or _augment(holder, seen):
                slot_to_player[s_idx] = player
                return True
        return False

    for player in range(len(player_roles)):
        _augment(player, [False] * len(slots))

    assigned: list[int | None] = [None] * len(player_roles)
    for s_idx, player in enumerate(slot_to_player):
        if player is not None:
            assigned[player] = s_idx
    return assigned


def compute_roster_depth(my_roster: pd.DataFrame) -> dict[str, list[dict]]:
    """Organizes the roster into categorical Mantra role buckets."""
    buckets = {
        "Por": [],
        "Dc": [],
        "Dd": [],
        "Ds": [],
        "E": [],
        "M": [],
        "C": [],
        "W": [],
        "T": [],
        "A": [],
        "Pc": [],
    }
    if my_roster.empty:
        return buckets

    for _, row in my_roster.iterrows():
        roles = parse_roles(row.get("roles", ""))
        info = {
            "name": row.get("name", ""),
            "team": row.get("team", ""),
            "roles": row.get("roles", ""),
            "fvm": float(row.get("fvm", 0) or 0),
            "expected_fp": float(row.get("expected_fp", 0) or 0),
        }
        for r in roles:
            if r in buckets:
                buckets[r].append(info)
    return buckets


def evaluate_formations(my_roster: pd.DataFrame) -> list[dict]:
    """Calculates tactical viability percentage for each official Mantra formation."""
    results = []
    if my_roster.empty:
        for key, fmt in MANTRA_FORMATIONS.items():
            results.append({
                "key": key,
                "name": fmt["name"],
                "description": fmt["description"],
                "score": 0.0,
                "playable": False,
                "missing": [label for _, _, label in fmt["needs"]],
                "depth_ratio": 0.0,
            })
        return results

    player_roles = [set(parse_roles(r)) for r in my_roster["roles"].tolist()]

    for key, fmt in MANTRA_FORMATIONS.items():
        # Expand the requirement groups into individual XI slots, then find the
        # maximum player->slot assignment. Counting each group independently would
        # let the same player cover several slots at once (e.g. two C players
        # satisfying both "C x2" and "M/C x1" in the 4-3-3).
        slots = [(roles, label) for roles, count, label in fmt["needs"] for _ in range(count)]
        total_slots = len(slots)
        assigned = _max_assignment(player_roles, slots)
        covered = sum(1 for slot in assigned if slot is not None)

        filled_per_label: dict[str, int] = {}
        for slot_idx in assigned:
            if slot_idx is not None:
                filled_per_label[slots[slot_idx][1]] = filled_per_label.get(slots[slot_idx][1], 0) + 1
        missing = []
        for roles, count, label in fmt["needs"]:
            got = filled_per_label.get(label, 0)
            if got < count:
                missing.append(f"{label} (ha {got}/{count})")

        playable = (covered == total_slots) and (len(my_roster) >= 11)
        # Depth score: starters covered + bench bonus (up to 22 players)
        raw_score = covered / max(1, total_slots)
        bench_ratio = min(1.0, len(my_roster) / 22.0)
        overall_score = raw_score * 0.70 + (raw_score * bench_ratio) * 0.30

        results.append({
            "key": key,
            "name": fmt["name"],
            "description": fmt["description"],
            "score": round(overall_score * 100, 1),
            "playable": playable,
            "missing": missing,
            "starters_covered": covered,
        })

    return sorted(results, key=lambda x: (x["playable"], x["score"]), reverse=True)
