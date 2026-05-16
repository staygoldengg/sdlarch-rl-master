# strategy_engine.py
"""
Strategy engine and player state prediction, weaponized for integration.
Includes: Vec2, PlayerState, landing prediction, projectile lead, strategy ranking.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import math

BH_GRAVITY = 980.0  # pixels/s²

@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def dist_to(self, other: "Vec2") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        l = self.length()
        return Vec2(self.x / l, self.y / l) if l > 0 else Vec2(0, 0)

    def sub(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def add(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def scale(self, s: float) -> "Vec2":
        return Vec2(self.x * s, self.y * s)


@dataclass
class PlayerState:
    id: str = "p1"
    pos: Vec2 = field(default_factory=Vec2)
    vel: Vec2 = field(default_factory=Vec2)
    is_airborne: bool = False
    is_invulnerable: bool = False
    is_attacking: bool = False
    current_move: Optional[str] = None
    last_move: Optional[str] = None
    buff_active: bool = False
    buff_expires_at: float = 0.0
    stocks_remaining: int = 3
    damage: float = 0.0


@dataclass
class Strategy:
    id: str
    name: str
    description: str
    priority: int
    suggested_moves: List[str]


def predict_landing(state: PlayerState, ground_y: float = 400.0,
                    gravity: float = BH_GRAVITY) -> Optional[Dict]:
    """Predict where an airborne player will land (quadratic solve)."""
    if not state.is_airborne:
        return None
    dy = ground_y - state.pos.y
    vy = state.vel.y
    # dy = vy*t + 0.5*gravity*t^2
    a = 0.5 * gravity
    b = vy
    c = -dy
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t = (-b + math.sqrt(disc)) / (2 * a)
    if t < 0:
        t = (-b - math.sqrt(disc)) / (2 * a)
    if t < 0:
        return None
    land_x = state.pos.x + state.vel.x * t
    return {"x": land_x, "y": ground_y, "time_ms": t * 1000}


def projectile_lead(shooter: PlayerState, target: PlayerState,
                    projectile_speed: float = 800.0) -> Optional[Vec2]:
    """Compute intercept point for a projectile given target velocity."""
    dx = target.pos.x - shooter.pos.x
    dy = target.pos.y - shooter.pos.y
    vx, vy = target.vel.x, target.vel.y
    speed = projectile_speed
    # Quadratic: |pos + vel*t|² = (speed*t)²
    a = vx * vx + vy * vy - speed * speed
    b = 2 * (dx * vx + dy * vy)
    c = dx * dx + dy * dy
    disc = b * b - 4 * a * c
    if abs(a) < 1e-6:
        return None
    if disc < 0:
        return None
    t = (-b - math.sqrt(disc)) / (2 * a)
    if t < 0:
        t = (-b + math.sqrt(disc)) / (2 * a)
    if t < 0:
        return None
    return Vec2(target.pos.x + vx * t, target.pos.y + vy * t)


# ── Strategy ranking ──────────────────────────────────────────────────────────
ALL_STRATEGIES: List[Strategy] = [
    Strategy("edge_guard", "Edge Guard", "Pressure opponent near the blast zone edge.", 90,
             ["DHeavy", "Dair", "Dodge", "Dash"]),
    Strategy("combo_starter", "Combo Starter", "Initiate a ground combo from a confirmed hit.", 80,
             ["NLight", "SLight", "DLight"]),
    Strategy("anti_air", "Anti-Air", "Intercept an airborne opponent.", 75,
             ["NHeavy", "Nair", "NSig"]),
    Strategy("neutral", "Neutral Game", "Hold center stage and apply pressure.", 50,
             ["Dash", "NLight", "Jump", "Dodge"]),
    Strategy("recovery_punish", "Recovery Punish", "Punish opponent's off-stage recovery.", 85,
             ["DHeavy", "Dair", "SSig"]),
    Strategy("defensive", "Defensive", "Avoid damage and reset neutral.", 40,
             ["Dodge", "Jump", "DJ", "WT"]),
    Strategy("buff_burst", "Buff Burst", "Capitalize on active buff window for damage.", 95,
             ["NSig", "SSig", "DSig", "NHeavy"]),
]


def rank_strategies(me: PlayerState, opp: PlayerState) -> List[Strategy]:
    """Return strategies sorted by situational priority."""
    dist = me.pos.dist_to(opp.pos)
    results = []
    for s in ALL_STRATEGIES:
        score = s.priority
        if s.id == "edge_guard" and abs(opp.pos.x) > 300:
            score += 20
        if s.id == "buff_burst" and me.buff_active:
            score += 30
        if s.id == "anti_air" and opp.is_airborne and dist < 200:
            score += 15
        if s.id == "combo_starter" and not opp.is_invulnerable and dist < 120:
            score += 10
        if s.id == "defensive" and me.damage > 120:
            score += 25
        if s.id == "recovery_punish" and opp.is_airborne and abs(opp.pos.y) > 200:
            score += 20
        results.append((score, s))
    results.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in results]
