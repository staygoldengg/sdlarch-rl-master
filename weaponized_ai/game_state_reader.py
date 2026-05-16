# game_state_reader.py
"""
Reads Brawlhalla game state from a screen frame.
Extracts: player damage %, stock count, approximate positions, airborne flag.

All pixel regions are calibrated for 1920x1080 full-screen Brawlhalla.
Call set_resolution(w, h) to rescale for other resolutions.

Methods:
    read_state(frame) → dict with me/opp state as flat obs vector (18 floats)
"""

import re
from typing import Optional, Tuple

# ── Screen layout constants (1920×1080 defaults) ──────────────────────────────
_W, _H = 1920, 1080

# Damage % readout boxes (bottom HUD)
# P1 damage is bottom-left, P2 is bottom-right
_DMGS = {
    "p1": (148, 930, 160, 60),   # x, y, w, h
    "p2": (1610, 930, 160, 60),
}

# Stock icon rows
_STOCKS = {
    "p1": (100, 1020, 120, 30),
    "p2": (1700, 1020, 120, 30),
}

# Stage floor Y estimate (pixels from top)
_GROUND_Y = 700


def set_resolution(w: int, h: int):
    """Rescale all regions to a different resolution."""
    global _W, _H, _DMGS, _STOCKS, _GROUND_Y
    sx, sy = w / 1920, h / 1080
    _W, _H = w, h
    _DMGS  = {
        "p1": (int(148*sx), int(930*sy), int(160*sx), int(60*sy)),
        "p2": (int(1610*sx), int(930*sy), int(160*sx), int(60*sy)),
    }
    _STOCKS = {
        "p1": (int(100*sx), int(1020*sy), int(120*sx), int(30*sy)),
        "p2": (int(1700*sx), int(1020*sy), int(120*sx), int(30*sy)),
    }
    _GROUND_Y = int(700 * sy)


# ── OCR helpers ───────────────────────────────────────────────────────────────
def _crop(frame: np.ndarray, region: Tuple[int,int,int,int]) -> np.ndarray:
    x, y, w, h = region
    return frame[y:y+h, x:x+w]


def _ocr_number(frame, region: Tuple[int,int,int,int]) -> float:
    """Extract a numeric value (damage %) from a screen region using pytesseract."""
    try:
        import cv2
        import pytesseract
        crop = _crop(frame, region)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        scaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(
            scaled,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789%"
        )
        nums = re.findall(r"\d+", text)
        return float(nums[0]) if nums else 0.0
    except Exception:
        return 0.0


def _count_stocks(frame, region: Tuple[int,int,int,int],
                  stock_color_hsv=(25, 200, 200)) -> int:
    """
    Count stock icons by looking for bright circular blobs in a HUD region.
    Falls back to 3 if detection fails.
    """
    try:
        import cv2
        import numpy as np
        crop = _crop(frame, region)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower = np.array([max(0, stock_color_hsv[0]-15), 100, 100])
        upper = np.array([min(179, stock_color_hsv[0]+15), 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Each stock is a small circle/blob
        count = sum(1 for c in contours if 50 < cv2.contourArea(c) < 2000)
        return max(0, min(3, count)) if count > 0 else 3
    except Exception:
        return 3


def _find_character_pos(frame, hue_range: Tuple[int,int]) -> Optional[Tuple[float,float]]:
    """
    Estimate a character's on-screen position by finding the largest colored blob.
    hue_range: (low_hue, high_hue) in HSV for the character's color.
    Returns (x, y) normalized to [-1, 1] relative to screen center, or None.
    """
    try:
        import cv2
        import numpy as np
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([hue_range[0], 80, 80]),
            np.array([hue_range[1], 255, 255])
        )
        # Ignore HUD area (bottom 20% of screen)
        mask[int(_H * 0.80):, :] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        # Normalize to [-1, 1] centered on screen
        nx = (cx - _W / 2) / (_W / 2)
        ny = (cy - _H / 2) / (_H / 2)
        return (nx, ny)
    except Exception:
        return None


# ── Main state reader ─────────────────────────────────────────────────────────

# Default character hue ranges (tune for your legend colors)
P1_HUE = (100, 130)   # blue-ish (player 1)
P2_HUE = (0, 15)      # red-ish  (player 2)


def read_state(frame) -> dict:
    """
    Extract full game state from a frame.
    Returns a dict with 'obs' (18-float list) plus human-readable fields.
    Includes: damage %, stocks, positions, KO flash, damage color tier, weapons.
    """
    import numpy as np

    p1_dmg    = _ocr_number(frame, _DMGS["p1"])
    p2_dmg    = _ocr_number(frame, _DMGS["p2"])

    # Enhance damage reading with color-based estimation
    p1_color = _damage_from_color(frame, _DMGS["p1"])
    p2_color = _damage_from_color(frame, _DMGS["p2"])
    # If OCR returned 0 but color says high damage, use color estimate
    if p1_dmg == 0 and p1_color["tier"] != "white":
        p1_dmg = p1_color["estimated_pct"]
    if p2_dmg == 0 and p2_color["tier"] != "white":
        p2_dmg = p2_color["estimated_pct"]

    p1_stocks = _count_stocks(frame, _STOCKS["p1"])
    p2_stocks = _count_stocks(frame, _STOCKS["p2"])

    p1_pos = _find_character_pos(frame, P1_HUE) or (0.0, 0.0)
    p2_pos = _find_character_pos(frame, P2_HUE) or (0.0, 0.0)

    # KO flash detection
    ko_info = detect_ko_flash(frame)

    # Weapon detection
    weapon_info = detect_weapons(frame)

    # Airborne: character Y is above the estimated ground line
    p1_screen_y = (p1_pos[1] + 1) / 2 * _H
    p2_screen_y = (p2_pos[1] + 1) / 2 * _H
    p1_air = float(p1_screen_y < _GROUND_Y * 0.90)
    p2_air = float(p2_screen_y < _GROUND_Y * 0.90)

    # Relative position / distance
    dx = p1_pos[0] - p2_pos[0]
    dy = p1_pos[1] - p2_pos[1]
    dist = float(np.hypot(dx, dy))

    # Weapon encoding: 1.0 = holding weapon, 0.0 = unarmed
    p1_armed = 0.0 if weapon_info["p1_weapon"] in ("none", "unknown") else 1.0

    # ── Build the 18-float obs vector ────────────────────────────────────────
    # [p1x, p1y, p1vx*, p1vy*, p1air, p1dmg/300, p1stocks/3,
    #  p2x, p2y, p2vx*, p2vy*, p2air, p2dmg/300, p2stocks/3,
    #  dist, dx, dy, p1_armed]
    obs = [
        p1_pos[0], p1_pos[1], 0.0, 0.0,
        p1_air, p1_dmg / 300.0, p1_stocks / 3.0,
        p2_pos[0], p2_pos[1], 0.0, 0.0,
        p2_air, p2_dmg / 300.0, p2_stocks / 3.0,
        dist, dx, dy, p1_armed,
    ]

    return {
        "obs": obs,
        "p1_damage": p1_dmg,
        "p2_damage": p2_dmg,
        "p1_stocks": p1_stocks,
        "p2_stocks": p2_stocks,
        "p1_pos": list(p1_pos),
        "p2_pos": list(p2_pos),
        "p1_airborne": bool(p1_air),
        "p2_airborne": bool(p2_air),
        # New fields
        "ko_flash": ko_info["ko_flash"],
        "screen_brightness": ko_info["brightness"],
        "p1_damage_tier": p1_color["tier"],
        "p2_damage_tier": p2_color["tier"],
        "p1_weapon": weapon_info["p1_weapon"],
        "p2_weapon": weapon_info["p2_weapon"],
        "stage_pickups": weapon_info["stage_pickups"],
    }



def compute_reward(prev: dict, curr: dict) -> float:
    """
    Compute reward from state transition.
      +2.0 per % of damage dealt to opponent
      -1.0 per % of damage taken
      +10.0 per stock taken from opponent  (also triggered by KO flash)
      -10.0 per stock lost
      +0.5 if p1 holds a weapon (weapon advantage)
    """
    dmg_dealt  = max(0.0, curr["p2_damage"] - prev["p2_damage"])
    dmg_taken  = max(0.0, curr["p1_damage"] - prev["p1_damage"])
    stocks_taken = max(0, prev["p2_stocks"] - curr["p2_stocks"])
    stocks_lost  = max(0, prev["p1_stocks"] - curr["p1_stocks"])

    reward = (dmg_dealt * 2.0) - (dmg_taken * 1.0) + (stocks_taken * 10.0) - (stocks_lost * 10.0)

    # KO flash bonus: reward the moment of the kill
    if curr.get("ko_flash") and stocks_taken > 0:
        reward += 5.0

    # Small bonus for holding a weapon (weapon = fighting advantage)
    if curr.get("p1_weapon") and curr["p1_weapon"] != "none":
        reward += 0.5

    return reward


# ── KO Flash Detection ────────────────────────────────────────────────────────
def detect_ko_flash(frame) -> dict:
    """
    Detect KO events by looking for:
      1. Bright white screen flash (average luminance > threshold)
      2. 'KO!' text region — a very bright band in the center of the screen

    Returns dict: {"ko_flash": bool, "brightness": float, "ko_center_bright": float}
    """
    try:
        import cv2
        import numpy as np
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        # Centre band (where KO! appears)
        h, w = gray.shape
        center = gray[h // 3: 2 * h // 3, w // 5: 4 * w // 5]
        center_bright = float(np.mean(center))

        # KO flash: very high overall brightness AND a very bright center
        ko_flash = brightness > 200 and center_bright > 210
        return {"ko_flash": ko_flash, "brightness": round(brightness, 1),
                "ko_center_bright": round(center_bright, 1)}
    except Exception:
        return {"ko_flash": False, "brightness": 0.0, "ko_center_bright": 0.0}


# ── Damage Color Detection ────────────────────────────────────────────────────
# Brawlhalla damage counter changes color with damage:
#   0–50 %  → white / light (low saturation)
#   50–100 % → yellow  (hue ~25–35)
#   100–150 % → orange  (hue ~10–25)
#   150 %+   → red/pink  (hue 0–10 or >160)
_DMG_COLOR_TIERS = {
    "white":  (0,   50),
    "yellow": (50,  100),
    "orange": (100, 150),
    "red":    (150, 999),
}

def _damage_from_color(frame, region: Tuple[int, int, int, int]) -> dict:
    """
    Estimate damage tier from the color of the HUD damage counter.
    Returns {"tier": str, "estimated_pct": float}
    """
    try:
        import cv2
        import numpy as np
        crop = _crop(frame, region)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Focus on bright pixels (the glowing number itself)
        mask = hsv[:, :, 2] > 160  # value channel
        if not np.any(mask):
            return {"tier": "white", "estimated_pct": 0.0}
        h_vals = hsv[:, :, 0][mask]
        s_vals = hsv[:, :, 1][mask]
        h_mean = float(np.mean(h_vals))
        s_mean = float(np.mean(s_vals))

        if s_mean < 40:
            tier = "white"
        elif h_mean < 12 or h_mean > 160:
            tier = "red"
        elif h_mean < 28:
            tier = "orange"
        else:
            tier = "yellow"

        mid = (_DMG_COLOR_TIERS[tier][0] + _DMG_COLOR_TIERS[tier][1]) / 2
        return {"tier": tier, "estimated_pct": mid}
    except Exception:
        return {"tier": "white", "estimated_pct": 0.0}


# ── Weapon Detection ──────────────────────────────────────────────────────────
# Brawlhalla HUD weapon icon positions (1920×1080)
_WEAPON_ICON = {
    "p1": (195, 965, 100, 45),   # x, y, w, h  — left HUD
    "p2": (1625, 965, 100, 45),  # right HUD
}

# Dominant hue ranges for each weapon family
_WEAPON_HUE = {
    "sword":     (20, 35),    # gold
    "axe":       (95, 115),   # steel-blue / gray
    "hammer":    (10, 22),    # brown-gold
    "spear":     (85, 105),   # teal
    "bow":       (15, 28),    # wood brown
    "blasters":  (170, 180),  # dark pinkish-gray
    "scythe":    (120, 145),  # purple
    "katars":    (90, 120),   # silver-blue
    "boots":     (0, 10),     # red-orange
    "orb":       (140, 160),  # purple-pink
    "gauntlets": (8, 18),     # dark brown
    "greatsword":(25, 40),    # bright gold
    "cannon":    (100, 125),  # dark blue-gray
}

# On-stage weapon pickup: glowing orb detection
_STAGE_REGION = (200, 200, 1520, 620)   # exclude HUD rows

def _identify_weapon_from_region(frame, region: Tuple[int, int, int, int]) -> str:
    """Classify a weapon icon by dominant hue in the region."""
    try:
        import cv2
        import numpy as np
        crop = _crop(frame, region)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Only consider saturated, bright pixels (the icon itself)
        mask = (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 80)
        if not np.any(mask):
            return "none"
        h_vals = hsv[:, :, 0][mask]
        h_mean = float(np.mean(h_vals))

        for weapon, (lo, hi) in _WEAPON_HUE.items():
            if lo <= h_mean <= hi:
                return weapon
        return "unknown"
    except Exception:
        return "none"


def detect_weapons(frame) -> dict:
    """
    Detect which weapons P1/P2 are currently holding (from HUD icons).
    Also detects weapon pickups on stage.
    Returns {"p1_weapon": str, "p2_weapon": str, "stage_pickups": int}
    """
    p1_w = _identify_weapon_from_region(frame, _WEAPON_ICON["p1"])
    p2_w = _identify_weapon_from_region(frame, _WEAPON_ICON["p2"])

    # Detect on-stage weapon orbs: bright glowing blobs in stage area
    stage_pickups = 0
    try:
        import cv2
        import numpy as np
        sx, sy, sw, sh = _STAGE_REGION
        stage = frame[sy:sy + sh, sx:sx + sw]
        hsv = cv2.cvtColor(stage, cv2.COLOR_BGR2HSV)
        # Weapons glow with high value + moderate saturation
        mask = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 200)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stage_pickups = sum(1 for c in contours if 300 < cv2.contourArea(c) < 5000)
    except Exception:
        pass

    return {"p1_weapon": p1_w, "p2_weapon": p2_w, "stage_pickups": stage_pickups}

