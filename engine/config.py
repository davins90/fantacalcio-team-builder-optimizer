from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class AuctionConfig:
    participants: int = 10
    starting_budget: int = 500
    roster_size: int = 25
    target_goalkeepers: int = 3
    max_players_per_real_team: int = 4
    risk_aversion: float = 0.35
    upside_weight: float = 0.20
    value_weight: float = 0.15
    simulations: int = 8
    bench_weight: float = 0.30
    manual_team_premium: float = 0.05

    @property
    def total_market_budget(self) -> int:
        return self.participants * self.starting_budget

    @property
    def total_roster_slots(self) -> int:
        return self.participants * self.roster_size

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "AuctionConfig":
        raw = raw or {}
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in raw.items() if k in allowed})
