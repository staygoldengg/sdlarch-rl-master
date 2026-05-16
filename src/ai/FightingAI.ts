/**
 * FightingAI — Local pattern-recognition + NPC strategy engine for Brawlhalla.
 * No API, no internet. Pure JS Markov chains + habit analysis + physics-aware NPC.
 *
 * How it works:
 *   1. You record opponent moves as you observe them in-match.
 *   2. Bigram + trigram Markov chains build a transition model of their flowchart.
 *   3. Unigram frequency builds a habit profile.
 *   4. predict() blends tri/bi/unigram scores to forecast the next move.
 *   5. getPatterns() surfaces repeated move-chains above a confidence threshold.
 *   6. Strategy engine enumerates NPC tactics, checks preconditions per-frame, and
 *      ranks applicable strategies (position, state flags, sight radius, timing).
 *   7. Lead targeting computes projectile intercept given player velocity + distance.
 *   8. Landing prediction uses jump arc physics (pos, vel, gravity) to time attacks.
 *   9. All counter suggestions are sourced from BH meta knowledge.
 */

// ── 2-D vector helpers ───────────────────────────────────────────────────────
export interface Vec2 { x: number; y: number }
const len = (v: Vec2) => Math.sqrt(v.x * v.x + v.y * v.y)
const sub = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y })
const add = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x + b.x, y: a.y + b.y })
const scale = (v: Vec2, s: number): Vec2 => ({ x: v.x * s, y: v.y * s })
const norm = (v: Vec2): Vec2 => { const l = len(v); return l > 0 ? scale(v, 1 / l) : { x: 0, y: 0 } }

// ── Player state ─────────────────────────────────────────────────────────────
/** Full observable state of one character at a point in time. */
export interface PlayerState {
  id: string
  pos: Vec2
  vel: Vec2
  isAirborne: boolean
  isInvulnerable: boolean
  isAttacking: boolean
  currentMove: string | null
  lastMove: string | null
  lastMoveAt: number       // ms timestamp
  buffActive: boolean
  buffExpiresAt: number    // ms timestamp (0 = no buff)
  stocksRemaining: number
  damage: number           // current damage %
}

export interface LandingPrediction { pos: Vec2; timeMs: number }

// ── Physics helpers ──────────────────────────────────────────────────────────
/** Standard BH gravity approximation (pixels/s²). Tune to match real measurements. */
export const BH_GRAVITY = 980

/**
 * Predict where an airborne player will land given current pos/vel and gravity.
 * groundY is the Y coordinate of the platform/stage floor (positive Y = down).
 * Returns null if the player is already grounded or will never reach groundY.
 */
export function predictLanding(
  player: PlayerState,
  gravity = BH_GRAVITY,
  groundY = 400,
): LandingPrediction | null {
  if (!player.isAirborne) return null
  const { pos, vel } = player
  // y(t) = pos.y + vel.y*t + 0.5*gravity*t²  →  solve for y(t) = groundY
  const a = 0.5 * gravity
  const b = vel.y
  const c = pos.y - groundY
  const disc = b * b - 4 * a * c
  if (disc < 0) return null
  const t1 = (-b + Math.sqrt(disc)) / (2 * a)
  const t2 = (-b - Math.sqrt(disc)) / (2 * a)
  const t = [t1, t2].filter(t => t > 0).sort((a, b) => a - b)[0]
  if (t == null) return null
  return {
    pos: { x: pos.x + vel.x * t, y: groundY },
    timeMs: Math.round(t * 1000),
  }
}

/**
 * Compute normalized direction to lead a projectile onto a moving target.
 * Uses the quadratic intercept formula: solve |myPos + dir*projSpeed*t - targetPos - targetVel*t| = 0.
 * Returns null if the projectile can never catch the target.
 */
export function computeLeadDirection(
  myPos: Vec2,
  target: PlayerState,
  projectileSpeed: number,
): Vec2 | null {
  const dp = sub(target.pos, myPos)
  const dv = target.vel
  const a = dv.x * dv.x + dv.y * dv.y - projectileSpeed * projectileSpeed
  const b = 2 * (dp.x * dv.x + dp.y * dv.y)
  const c = dp.x * dp.x + dp.y * dp.y
  let t: number
  if (Math.abs(a) < 1e-6) {
    // Linear — projectile is fast relative to target
    if (Math.abs(b) < 1e-6) return null
    t = -c / b
  } else {
    const disc = b * b - 4 * a * c
    if (disc < 0) return null
    const t1 = (-b + Math.sqrt(disc)) / (2 * a)
    const t2 = (-b - Math.sqrt(disc)) / (2 * a)
    const positives = [t1, t2].filter(t => t > 0).sort((a, b) => a - b)
    if (!positives.length) return null
    t = positives[0]
  }
  const intercept = add(target.pos, scale(target.vel, t))
  return norm(sub(intercept, myPos))
}

/**
 * Find the nearest PlayerState within sight radius from myPos.
 * Ignores targets with the same id as the NPC.
 */
export function detectNearest(
  myPos: Vec2,
  players: PlayerState[],
  radius: number,
  myId?: string,
): PlayerState | null {
  let best: PlayerState | null = null
  let bestDist = Infinity
  for (const p of players) {
    if (myId && p.id === myId) continue
    const d = len(sub(p.pos, myPos))
    if (d <= radius && d < bestDist) { best = p; bestDist = d }
  }
  return best
}

// ── Strategy system ──────────────────────────────────────────────────────────
export interface Strategy {
  id: string
  name: string
  description: string
  /** True when this strategy is currently applicable. */
  conditions: (my: PlayerState, target: PlayerState, now: number) => boolean
  /** Higher priority strategies are checked first / preferred. */
  priority: number
  /** BH moves to execute when this strategy is active. */
  suggestedMoves?: string[]
}

/** Straight-line distance between two states. */
const dist = (a: PlayerState, b: PlayerState) => len(sub(a.pos, b.pos))

/**
 * Predefined BH NPC strategy library.
 * Each entry encodes: when to use it, what to do, and at what priority.
 */
export const BH_STRATEGIES: Strategy[] = [
  // ── Avoidance ─────────────────────────────────────────────────────────────
  {
    id: 'invuln_wait',
    name: 'Wait Out Invulnerability',
    description: 'Target is temporarily invulnerable — hold position, do not waste attacks.',
    priority: 95,
    conditions: (_my, target) => target.isInvulnerable,
    suggestedMoves: ['Dodge away', 'Platform camp'],
  },
  // ── Off-stage gimp ────────────────────────────────────────────────────────
  {
    id: 'gimp_recovery',
    name: 'Gimp Recovery',
    description: 'Target is airborne and below stage edge — intercept their recovery.',
    priority: 88,
    conditions: (_my, target) => target.isAirborne && target.pos.y > 350,
    suggestedMoves: ['Dair spike', 'Rec gimp', 'Edge-guard dodge-cancel'],
  },
  // ── Land-punish ───────────────────────────────────────────────────────────
  {
    id: 'punish_landing',
    name: 'Punish Landing',
    description: 'Target will land soon — position for a confirm on landing lag.',
    priority: 85,
    conditions: (my, target) => {
      if (!target.isAirborne) return false
      const landing = predictLanding(target)
      if (!landing) return false
      // Viable if we can reach landing zone within the time window
      const myTravelTime = (dist(my, target) / 600) * 1000 // rough px/s estimate
      return landing.timeMs > 0 && landing.timeMs < 1200 && myTravelTime < landing.timeMs
    },
    suggestedMoves: ['DLight on landing', 'GC NLight', 'Sair into landing lag'],
  },
  // ── Buff expiry punish ────────────────────────────────────────────────────
  {
    id: 'punish_buff_expiry',
    name: 'Punish Buff Expiry',
    description: 'Target buff is about to expire — attack the moment they lose the enhancement.',
    priority: 82,
    conditions: (_my, target, now) =>
      target.buffActive && target.buffExpiresAt > 0 && (target.buffExpiresAt - now) < 1200,
    suggestedMoves: ['Wait 1 s then NSig', 'GC on expiry frame', 'Jump Nair timing'],
  },
  // ── Panic jump punish ─────────────────────────────────────────────────────
  {
    id: 'anti_panic_jump',
    name: 'Anti Panic Jump',
    description: 'Target panic-jumped (fast upward velocity with no attack) — anti-air.',
    priority: 80,
    conditions: (_my, target) => target.isAirborne && target.vel.y < -300 && !target.isAttacking,
    suggestedMoves: ['GP anti-air', 'Jump → Dair follow', 'Scythe NLight read tree'],
  },
  // ── Projectile lead ───────────────────────────────────────────────────────
  {
    id: 'lead_projectile',
    name: 'Lead Projectile at Moving Target',
    description: 'Target is moving laterally fast — lead the weapon throw to intercept.',
    priority: 75,
    conditions: (_my, target) => Math.abs(target.vel.x) > 250,
    suggestedMoves: ['WT lead (computeLeadDirection)', 'Bow charged shot lead'],
  },
  // ── Close-range scramble ──────────────────────────────────────────────────
  {
    id: 'close_scramble',
    name: 'Close-Range Scramble',
    description: 'Target is within grab/katar range — force scramble or throw.',
    priority: 70,
    conditions: (my, target) => dist(my, target) < 80,
    suggestedMoves: ['Katar DLight string', 'GC string into Nair', 'Dodge-cancel GP'],
  },
  // ── Zone/poke ─────────────────────────────────────────────────────────────
  {
    id: 'mid_range_poke',
    name: 'Mid-Range Poke',
    description: 'Maintain optimal spacing and poke with long-range normals.',
    priority: 55,
    conditions: (my, target) => { const d = dist(my, target); return d >= 80 && d <= 220 },
    suggestedMoves: ['SLight max range', 'Spear NLight', 'Bow SLight'],
  },
  // ── Chase / close gap ────────────────────────────────────────────────────
  {
    id: 'close_gap',
    name: 'Close the Gap',
    description: 'Target is far away — dash or GC to close distance before they recover.',
    priority: 40,
    conditions: (my, target) => dist(my, target) > 220,
    suggestedMoves: ['GC approach', 'Dash toward', 'DJFF approach'],
  },
]

/**
 * Evaluate all strategies against the current game state.
 * Returns all applicable strategies sorted by priority (highest first).
 * Iterate only until you find the first strategy to act on each frame to avoid
 * checking too many conditions in a single frame.
 */
export function evaluateStrategies(
  my: PlayerState,
  target: PlayerState,
  strategies: Strategy[] = BH_STRATEGIES,
): Strategy[] {
  const now = Date.now()
  return strategies
    .filter(s => s.conditions(my, target, now))
    .sort((a, b) => b.priority - a.priority)
}

// ── Counter knowledge (BH-meta informed) ────────────────────────────────────
export const MOVE_COUNTERS: Record<string, string[]> = {
  NLight:   ['Chase dodge through it', 'Out-space with SLight at max range', 'X-Pivot Nair punish on whiff'],
  SLight:   ['Dodge in (phase through hitbox)', 'GC DLight punish on landing', 'Jump + Nair over the swing'],
  DLight:   ['Dodge back (bait the whiff)', 'GC NLight after the dodge', 'Stay just outside range'],
  Nair:     ['Dodge away then DLight on landing', 'GP from below to beat it', 'Out-range with SLight'],
  Sair:     ['Dodge toward (go under hitbox)', 'DLight on their landing lag', 'GC reverse Nair'],
  Dair:     ['Dodge away or diagonally up', 'Chase offstage with Rec gimp', 'Punish landing with DLight'],
  Rec:      ['Dodge toward for full punish', 'Gimp with Dair offstage', 'Platform bait → punish landing'],
  GP:       ['Dodge up or hard away', 'Jump → dodge in after GP whiffs', 'Fast aerial on the whiff window'],
  NSig:     ['Dodge hard away (huge startup)', 'Jump over + aerial punish', 'Bait with movement then punish'],
  SSig:     ['Dodge toward (past hitbox arc)', 'Punish the startup with fast combo', 'Stay mobile, avoid committing'],
  DSig:     ['Dodge back and high', 'Chase with jump aerial', 'Bait the DSig, punish on land'],
  DJFF:     ['Anti-air with GP', 'React to approach angle', 'Maintain grounded spacing'],
  GC:       ['Punish the landing platform', 'Catch with fast aerial', 'Mirror GC to match their tech'],
  CD:       ['Reverse Nair punish on the chase', 'Stay at same height level', 'Bait the CD then dodge through'],
  WT:       ['Catch the weapon throw', 'Dodge toward + immediate punish', 'Throw back to force pressure'],
  'X-Pivot':['Dodge in the pivot direction', 'Punish the hesitation window', 'Stay grounded and space patiently'],
}

// ── Frame data ────────────────────────────────────────────────────────────────
/** Startup / active / endlag in 60 Hz frames; range in pixels; base damage %. */
export interface FrameData {
  startup: number
  active:  number
  endlag:  number
  range:   number
  damage:  number
}

export const BH_FRAME_DATA: Record<string, FrameData> = {
  NLight:  { startup: 6,  active: 4,  endlag: 10, range: 90,  damage: 14 },
  SLight:  { startup: 5,  active: 3,  endlag: 12, range: 120, damage: 16 },
  DLight:  { startup: 4,  active: 3,  endlag: 10, range: 80,  damage: 13 },
  Nair:    { startup: 8,  active: 5,  endlag: 14, range: 100, damage: 17 },
  Sair:    { startup: 7,  active: 4,  endlag: 16, range: 130, damage: 18 },
  Dair:    { startup: 9,  active: 6,  endlag: 18, range: 80,  damage: 15 },
  NSig:    { startup: 18, active: 10, endlag: 30, range: 150, damage: 28 },
  SSig:    { startup: 14, active: 8,  endlag: 28, range: 200, damage: 32 },
  DSig:    { startup: 20, active: 12, endlag: 35, range: 120, damage: 26 },
  GC:      { startup: 3,  active: 2,  endlag: 8,  range: 0,   damage: 0  },
  Dodge:   { startup: 4,  active: 15, endlag: 12, range: 0,   damage: 0  },
  Jump:    { startup: 3,  active: 0,  endlag: 0,  range: 0,   damage: 0  },
  // Weapon-specific overrides
  NLight_Scythe: { startup: 6,  active: 4,  endlag: 10, range: 95,  damage: 14 },
  SLight_Chak:   { startup: 5,  active: 3,  endlag: 11, range: 115, damage: 15 },
}

/**
 * Calculate the fastest viable counter move given the enemy's current move
 * and the NPC's available moves with their frame data.
 * Falls back to 'DODGE' if no move beats the enemy startup in time.
 */
export function calculateCounter(
  enemyMoveId: string,
  distance: number,
  myMoves: Array<{ id: string; fd: FrameData }>,
): { move: string; fd: FrameData } | 'DODGE' {
  const enemyFd = BH_FRAME_DATA[enemyMoveId]
  if (!enemyFd) return 'DODGE'
  const viable = myMoves.filter(
    m => m.fd.startup < enemyFd.startup && m.fd.range >= distance,
  )
  if (!viable.length) return 'DODGE'
  const winner = viable.reduce((best, m) => m.fd.damage > best.fd.damage ? m : best)
  return { move: winner.id, fd: winner.fd }
}

// ── Scancode table (matches fight_engine.py DEFAULT_BINDS + _VK) ──────────────
/** Windows scancodes for BH-relevant keys. Mirrors fight_engine.py _VK table. */
export const SC: Record<string, number> = {
  A: 0x1E, D: 0x20, W: 0x11, S: 0x1F,   // directional
  N: 0x31, M: 0x32,                       // default light / heavy
  SPACE: 0x39,                            // jump
  E: 0x12,                                // dodge
  F: 0x21,                                // pickup
  SHIFT: 0x2A,                            // alt heavy
  J: 0x24, K: 0x25,                       // alt light/heavy
}

// ── Move → key translation ────────────────────────────────────────────────────
export interface KeyPress {
  /** Primary scancode (the attack / action key). */
  primary:    number
  /** Simultaneous modifier scancode — direction key held while attacking. 0 = none. */
  modifier:   number
  /** How long to hold the keys (ms). */
  holdMs:     number
  /** Pre-action delay — simulates decision time for training realism (ms). */
  reactionMs: number
}

/**
 * Translate a BH move name to its KeyPress descriptor.
 * @param move       Move name string (e.g. 'Sair', 'NSig')
 * @param direction  Facing direction context for side-moves
 * @param binds      Override default SC values per key name
 */
export function translateMoveToKeys(
  move: string,
  direction: 'left' | 'right' | 'neutral' = 'right',
  binds: Partial<Record<string, number>> = {},
): KeyPress {
  const b = { ...SC, ...binds } as Record<string, number>
  const dir  = direction === 'left' ? b.A : direction === 'right' ? b.D : 0
  const L    = b.N, H = b.M, J = b.SPACE, DG = b.E
  const table: Record<string, KeyPress> = {
    NLight:  { primary: L,   modifier: 0,   holdMs: 45,  reactionMs: 0 },
    SLight:  { primary: L,   modifier: dir, holdMs: 50,  reactionMs: 0 },
    DLight:  { primary: L,   modifier: b.S, holdMs: 45,  reactionMs: 0 },
    NHeavy:  { primary: H,   modifier: 0,   holdMs: 50,  reactionMs: 0 },
    SHeavy:  { primary: H,   modifier: dir, holdMs: 55,  reactionMs: 0 },
    DHeavy:  { primary: H,   modifier: b.S, holdMs: 50,  reactionMs: 0 },
    NSig:    { primary: H,   modifier: 0,   holdMs: 180, reactionMs: 0 },
    SSig:    { primary: H,   modifier: dir, holdMs: 160, reactionMs: 0 },
    DSig:    { primary: H,   modifier: b.S, holdMs: 200, reactionMs: 0 },
    Nair:    { primary: L,   modifier: J,   holdMs: 50,  reactionMs: 0 },
    Sair:    { primary: L,   modifier: dir, holdMs: 55,  reactionMs: 0 },
    Dair:    { primary: L,   modifier: b.S, holdMs: 50,  reactionMs: 0 },
    Jump:    { primary: J,   modifier: 0,   holdMs: 25,  reactionMs: 0 },
    DJ:      { primary: J,   modifier: 0,   holdMs: 25,  reactionMs: 80 },
    Dodge:   { primary: DG,  modifier: dir, holdMs: 20,  reactionMs: 0 },
    GC:      { primary: J,   modifier: dir, holdMs: 20,  reactionMs: 0 },
    Dash:    { primary: dir, modifier: 0,   holdMs: 30,  reactionMs: 0 },
    WT:      { primary: H,   modifier: dir, holdMs: 30,  reactionMs: 0 },
    Pickup:  { primary: b.F, modifier: 0,   holdMs: 20,  reactionMs: 0 },
  }
  return table[move] ?? { primary: DG, modifier: 0, holdMs: 20, reactionMs: 0 }
}

// ── Combo sequences ───────────────────────────────────────────────────────────
export interface ComboStep {
  move: string
  direction?: 'left' | 'right' | 'neutral'
  /** Milliseconds to wait after this step before executing the next. */
  delayMs: number
}

export type ComboSequence = ComboStep[]

/** BH weapon combo database. Each weapon maps to ordered confirmed sequences. */
export const BH_COMBOS: Record<string, ComboSequence[]> = {
  Scythe: [
    // NLight → Sair — 2-frame-gap true combo
    [{ move: 'NLight', direction: 'neutral', delayMs: 80 },
     { move: 'Sair',   direction: 'right',   delayMs: 0  }],
    // DLight → Nair
    [{ move: 'DLight', direction: 'neutral', delayMs: 70 },
     { move: 'Nair',   direction: 'neutral', delayMs: 0  }],
    // Dair spike → Chase Dodge
    [{ move: 'Dair',   direction: 'neutral', delayMs: 100 },
     { move: 'Dodge',  direction: 'right',   delayMs: 0   }],
  ],
  Chakram: [
    // Split mode: SLight → DLight (true damage string)
    [{ move: 'SLight', direction: 'right',   delayMs: 65 },
     { move: 'DLight', direction: 'neutral', delayMs: 0  }],
    // NLight → Sair (2-frame gap)
    [{ move: 'NLight', direction: 'neutral', delayMs: 80 },
     { move: 'Sair',   direction: 'right',   delayMs: 0  }],
    // Fused DSig kill confirm
    [{ move: 'DSig', direction: 'neutral', delayMs: 0 }],
  ],
  Sword: [
    [{ move: 'SLight', direction: 'right',   delayMs: 70 },
     { move: 'Nair',   direction: 'neutral', delayMs: 0  }],
    [{ move: 'DLight', direction: 'neutral', delayMs: 65 },
     { move: 'GC',     direction: 'right',   delayMs: 50 },
     { move: 'NLight', direction: 'neutral', delayMs: 0  }],
  ],
  Gauntlets: [
    [{ move: 'DLight', direction: 'neutral', delayMs: 60 },
     { move: 'DLight', direction: 'neutral', delayMs: 60 },
     { move: 'Nair',   direction: 'neutral', delayMs: 0  }],
  ],
  Katars: [
    [{ move: 'DLight', direction: 'neutral', delayMs: 55 },
     { move: 'DLight', direction: 'neutral', delayMs: 55 },
     { move: 'SLight', direction: 'right',   delayMs: 0  }],
  ],
  Spear: [
    [{ move: 'NLight', direction: 'neutral', delayMs: 80 },
     { move: 'Nair',   direction: 'neutral', delayMs: 0  }],
  ],
  Axe: [
    [{ move: 'NLight', direction: 'neutral', delayMs: 80 },
     { move: 'NSig',   direction: 'neutral', delayMs: 0  }],
  ],
  Bow: [
    [{ move: 'SLight', direction: 'right', delayMs: 70 },
     { move: 'SLight', direction: 'right', delayMs: 0  }],
  ],
  Hammer: [
    [{ move: 'SLight', direction: 'right',   delayMs: 80 },
     { move: 'Nair',   direction: 'neutral', delayMs: 0  }],
  ],
}

// ── Circular input queue ──────────────────────────────────────────────────────
export interface InputQueueEntry {
  keys:        KeyPress
  scheduledAt: number   // Date.now() — execute when now ≥ this value
  id:          string
}

/**
 * Fixed-capacity circular input queue.
 * Prevents dropped inputs when the AI enqueues combos faster than they execute.
 */
export class InputQueue {
  private _buf: Array<InputQueueEntry | null>
  private _head = 0
  private _tail = 0
  private _count = 0
  private readonly _cap: number

  constructor(capacity = 32) {
    this._cap = capacity
    this._buf = new Array(capacity).fill(null)
  }

  /** Add an entry. Returns false and discards if the queue is at capacity. */
  enqueue(entry: InputQueueEntry): boolean {
    if (this._count >= this._cap) return false
    this._buf[this._tail] = entry
    this._tail = (this._tail + 1) % this._cap
    this._count++
    return true
  }

  /** Dequeue the next entry whose scheduledAt ≤ Date.now(). Returns null if none ready. */
  dequeueReady(): InputQueueEntry | null {
    if (!this._count) return null
    const entry = this._buf[this._head]
    if (!entry || entry.scheduledAt > Date.now()) return null
    this._buf[this._head] = null
    this._head = (this._head + 1) % this._cap
    this._count--
    return entry
  }

  get length()   { return this._count }
  get capacity() { return this._cap   }

  clear() {
    this._buf.fill(null)
    this._head = this._tail = this._count = 0
  }
}

// ── Lady Vera flowcharts (Scythe + Chakram) ───────────────────────────────────
export type VeraWeapon    = 'Scythe' | 'Chakram'
export type ChakramMode   = 'Split' | 'Fused'
export type DetectionType = 'DODGE_BLUE' | 'SIG_YELLOW' | 'JUMP_SMOKE' | 'NONE'
export type OffstageRelPos = 'AWAY_FROM_STAGE' | 'TOWARD_STAGE' | 'BELOW_STAGE'

export interface VeraAction {
  combo:   ComboSequence
  label:   string
  urgency: 'instant' | 'normal'
}

/**
 * Scythe off-stage gimp engine.
 * Selects the optimal edge-guard sequence based on opponent's airborne behaviour.
 *
 * Decision tree:
 *   DODGE_BLUE + AWAY  → Chase Dodge Out → Reverse N-Air  (pull into killbox)
 *   DODGE_BLUE + IN    → N-Sig Active Input               (pull them back out)
 *   JUMP_SMOKE         → D-Air spike                      (out of options)
 *   SIG_YELLOW         → Spot Dodge                       (counter startup)
 *   BELOW              → D-Air → Reverse N-Light          (catch wall-hug)
 */
export function scytheOffstageEngine(
  detection: DetectionType,
  relPos:    OffstageRelPos,
  _enemyDamage: number,
): VeraAction {
  if (detection === 'SIG_YELLOW') {
    return {
      label: 'Spot Dodge (Sig counter)',
      urgency: 'instant',
      combo: [{ move: 'Dodge', direction: 'neutral', delayMs: 0 }],
    }
  }
  if (detection === 'JUMP_SMOKE') {
    return {
      label: 'D-Air Spike (no options left)',
      urgency: 'instant',
      combo: [{ move: 'Dair', direction: 'neutral', delayMs: 0 }],
    }
  }
  if (detection === 'DODGE_BLUE') {
    if (relPos === 'AWAY_FROM_STAGE') {
      return {
        label: 'Gimp: CD Out → Rev N-Air',
        urgency: 'normal',
        combo: [
          { move: 'Dodge',  direction: 'left',    delayMs: 80 },
          { move: 'Nair',   direction: 'neutral', delayMs: 0  },
        ],
      }
    }
    if (relPos === 'TOWARD_STAGE') {
      return {
        label: 'Scythe N-Sig Active (pull back out)',
        urgency: 'normal',
        combo: [{ move: 'NSig', direction: 'neutral', delayMs: 0 }],
      }
    }
  }
  // Default: D-Air ground-pound trap → Reverse N-Light for wall-hug
  return {
    label: 'D-Air → Rev N-Light (wall-hug trap)',
    urgency: 'normal',
    combo: [
      { move: 'Dair',   direction: 'neutral', delayMs: 100 },
      { move: 'NLight', direction: 'left',    delayMs: 0   },
    ],
  }
}

/**
 * Chakram mode controller.
 * Fused mode = kill confirms at ≥ 100 damage (D-Sig, massive hitbox).
 * Split mode = damage building (S-Light → D-Light true string).
 */
export function chakramController(
  mode:        ChakramMode,
  enemyDamage: number,
  distance:    number,
): VeraAction {
  if (mode === 'Fused' && enemyDamage >= 100) {
    return {
      label: 'Fused D-Sig kill confirm',
      urgency: 'normal',
      combo: [{ move: 'DSig', direction: 'neutral', delayMs: 0 }],
    }
  }
  if (mode === 'Split' && enemyDamage < 100) {
    return {
      label: 'Split: S-Light → D-Light (true string)',
      urgency: 'normal',
      combo: [
        { move: 'SLight', direction: 'right',   delayMs: 65 },
        { move: 'DLight', direction: 'neutral', delayMs: 0  },
      ],
    }
  }
  if (distance > 200) {
    return {
      label: 'Chakram D-Sig (zoning, at range)',
      urgency: 'normal',
      combo: [{ move: 'DSig', direction: 'neutral', delayMs: 0 }],
    }
  }
  return {
    label: 'Chakram N-Light → S-Air (2-frame gap)',
    urgency: 'normal',
    combo: BH_COMBOS.Chakram[1],
  }
}

/**
 * Top-level Lady Vera weapon controller.
 * Routes to Scythe or Chakram engine based on current weapon and game state.
 * @param weapon       'Scythe' | 'Chakram'
 * @param detection    Current visual detection event
 * @param target       Tracked opponent PlayerState
 * @param myPos        NPC position
 * @param chakramMode  'Split' (damage) or 'Fused' (kill confirm)
 */
export function veraController(
  weapon:      VeraWeapon,
  detection:   DetectionType,
  target:      PlayerState,
  myPos:       Vec2,
  chakramMode: ChakramMode = 'Split',
): VeraAction {
  const distance = len(sub(target.pos, myPos))
  if (weapon === 'Scythe') {
    const offstage = target.isAirborne && target.pos.y > 350
    if (offstage) {
      const relPos: OffstageRelPos =
        target.vel.x < -80  ? 'AWAY_FROM_STAGE' :
        target.vel.x >  80  ? 'TOWARD_STAGE' : 'BELOW_STAGE'
      return scytheOffstageEngine(detection, relPos, target.damage)
    }
    // On-stage: S-Sig Active Input for kill confirm at orange/red health
    if (target.damage >= 100) {
      return {
        label: 'Scythe S-Sig Active (hold dir, kill confirm)',
        urgency: 'normal',
        combo: [{ move: 'SSig', direction: 'right', delayMs: 0 }],
      }
    }
    // On-stage damage building
    return {
      label: 'Scythe N-Light → S-Air',
      urgency: 'normal',
      combo: BH_COMBOS.Scythe[0],
    }
  }
  return chakramController(chakramMode, target.damage, distance)
}

// ── Types ────────────────────────────────────────────────────────────────────
export interface Prediction {
  move: string
  confidence: number   // 0–100
  counter: string[]
}

export interface HabitEntry {
  move: string
  count: number
  pct: number
}

export interface PatternEntry {
  from: string
  to: string
  count: number
  pct: number   // % of time "from" is followed by "to"
}

// ── Engine ────────────────────────────────────────────────────────────────────
type TransitionMap = Record<string, Record<string, number>>

export class FightingAI {
  private unigrams: Record<string, number> = {}
  private bigrams: TransitionMap = {}
  private trigrams: TransitionMap = {}
  private history: string[] = []
  private totalMoves = 0

  // ── State tracking ──────────────────────────────────────────────────────
  private _targetState: PlayerState | null = null
  private _myState: PlayerState | null = null
  private _sightRadius = 350  // px — NPC sight area radius

  // ── Combo queue + Lady Vera config ─────────────────────────────────────
  private _inputQueue = new InputQueue(32)
  private _vera: { weapon: VeraWeapon; chakramMode: ChakramMode } | null = null

  /**
   * Update the tracked state of the observed player character.
   * Call this every frame from your capture/game loop.
   */
  trackTargetState(state: PlayerState) {
    // If the state reflects a move, auto-record it for Markov chains
    if (state.currentMove && state.currentMove !== this._targetState?.currentMove) {
      this.record(state.currentMove)
    }
    this._targetState = state
  }

  /** Update the NPC's own state (position, velocity, flags). */
  trackMyState(state: PlayerState) {
    this._myState = state
  }

  /** Set the sight radius for nearest-target detection (pixels). */
  setSightRadius(radius: number) { this._sightRadius = radius }

  /**
   * Return the best applicable NPC strategy given current tracked states.
   * Returns null if no target is in sight or no strategy applies.
   */
  getBestStrategy(allPlayers?: PlayerState[]): Strategy | null {
    const my = this._myState
    if (!my) return null

    // Sight detection: pick nearest player within radius (ignores self)
    let target = this._targetState
    if (allPlayers && allPlayers.length) {
      target = detectNearest(my.pos, allPlayers, this._sightRadius, my.id) ?? target
    }
    if (!target) return null

    const applicable = evaluateStrategies(my, target)
    return applicable[0] ?? null
  }

  /**
   * Compute the projectile lead direction toward the currently tracked target.
   * @param myPos       NPC shooter position
   * @param projSpeed   Projectile speed in px/s (e.g. 600 for bow, 500 for WT)
   */
  getLeadDirection(myPos: Vec2, projSpeed: number): Vec2 | null {
    if (!this._targetState) return null
    return computeLeadDirection(myPos, this._targetState, projSpeed)
  }

  /**
   * Predict where the tracked target will land.
   * Useful for timing attacks to hit on landing lag.
   */
  predictTargetLanding(gravity?: number, groundY?: number): LandingPrediction | null {
    if (!this._targetState) return null
    return predictLanding(this._targetState, gravity, groundY)
  }

  /**
   * Return all applicable strategies sorted by priority.
   * Use this to pick a strategy, or iterate the list and execute the first
   * one whose conditions are still true when you act (lazy evaluation).
   */
  getAllApplicableStrategies(allPlayers?: PlayerState[]): Strategy[] {
    const my = this._myState
    if (!my) return []
    let target = this._targetState
    if (allPlayers && allPlayers.length) {
      target = detectNearest(my.pos, allPlayers, this._sightRadius, my.id) ?? target
    }
    if (!target) return []
    return evaluateStrategies(my, target)
  }

  // Record one observed opponent move
  record(move: string) {
    this.totalMoves++
    this.unigrams[move] = (this.unigrams[move] ?? 0) + 1

    const h = this.history

    // Bigram: previous → current
    if (h.length >= 1) {
      const key = h[h.length - 1]
      if (!this.bigrams[key]) this.bigrams[key] = {}
      this.bigrams[key][move] = (this.bigrams[key][move] ?? 0) + 1
    }

    // Trigram: prev2|prev1 → current
    if (h.length >= 2) {
      const key = `${h[h.length - 2]}|${h[h.length - 1]}`
      if (!this.trigrams[key]) this.trigrams[key] = {}
      this.trigrams[key][move] = (this.trigrams[key][move] ?? 0) + 1
    }

    this.history.push(move)
    if (this.history.length > 100) this.history.shift()
  }

  // Predict top-3 next moves, blending tri > bi > unigram
  predict(): Prediction[] {
    const h = this.history
    const scores: Record<string, number> = {}

    // Trigram contribution (most specific — 60% weight)
    if (h.length >= 2) {
      const key = `${h[h.length - 2]}|${h[h.length - 1]}`
      const tg = this.trigrams[key]
      if (tg) {
        const total = Object.values(tg).reduce((a, b) => a + b, 0)
        for (const [m, c] of Object.entries(tg)) {
          scores[m] = (scores[m] ?? 0) + (c / total) * 0.6
        }
      }
    }

    // Bigram contribution (30% weight)
    if (h.length >= 1) {
      const bg = this.bigrams[h[h.length - 1]]
      if (bg) {
        const total = Object.values(bg).reduce((a, b) => a + b, 0)
        for (const [m, c] of Object.entries(bg)) {
          scores[m] = (scores[m] ?? 0) + (c / total) * 0.3
        }
      }
    }

    // Unigram fallback (10% weight — baseline frequency)
    if (this.totalMoves > 0) {
      for (const [m, c] of Object.entries(this.unigrams)) {
        scores[m] = (scores[m] ?? 0) + (c / this.totalMoves) * 0.1
      }
    }

    return Object.entries(scores)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([move, conf]) => ({
        move,
        confidence: Math.min(99, Math.round(conf * 100)),
        counter: MOVE_COUNTERS[move] ?? ['Dodge and punish on whiff', 'Stay patient and space'],
      }))
  }

  // Move frequency profile (for habit bar chart)
  getHabits(): HabitEntry[] {
    if (this.totalMoves === 0) return []
    return Object.entries(this.unigrams)
      .sort((a, b) => b[1] - a[1])
      .map(([move, count]) => ({
        move,
        count,
        pct: Math.round((count / this.totalMoves) * 100),
      }))
  }

  // Detected repeating move chains (bigram frequency ≥ 3 occurrences)
  getPatterns(): PatternEntry[] {
    const results: PatternEntry[] = []
    for (const [from, targets] of Object.entries(this.bigrams)) {
      const total = Object.values(targets).reduce((a, b) => a + b, 0)
      for (const [to, count] of Object.entries(targets)) {
        if (count >= 3) {
          results.push({ from, to, count, pct: Math.round((count / total) * 100) })
        }
      }
    }
    return results.sort((a, b) => b.pct - a.pct).slice(0, 6)
  }

  // Last N moves recorded (reversed: newest first)
  getHistory(n = 12): string[] {
    return [...this.history].slice(-n).reverse()
  }

  getTotal() { return this.totalMoves }

  // ── Combo queue API ─────────────────────────────────────────────────────
  /**
   * Configure Lady Vera weapon mode for specialized flowcharts.
   * Must be called before getVeraAction() returns meaningful results.
   */
  setVeraConfig(weapon: VeraWeapon, chakramMode: ChakramMode = 'Split') {
    this._vera = { weapon, chakramMode }
  }

  /**
   * Translate a combo sequence into KeyPress entries and push them all onto
   * the input queue, spaced by each step's delayMs.
   * Returns false if the queue was too full to accept the entire combo.
   */
  queueCombo(
    seq: ComboSequence,
    direction: 'left' | 'right' = 'right',
  ): boolean {
    let t = Date.now()
    for (const step of seq) {
      const keys = translateMoveToKeys(step.move, step.direction ?? direction)
      if (!this._inputQueue.enqueue({ keys, scheduledAt: t + keys.reactionMs, id: `${step.move}-${t}` })) {
        return false
      }
      t += step.delayMs + keys.holdMs
    }
    return true
  }

  /**
   * Dequeue and return the next input that is ready to fire right now.
   * Call this every game frame (≈16 ms) from your input pump loop.
   */
  nextPendingInput(): InputQueueEntry | null {
    return this._inputQueue.dequeueReady()
  }

  /** Return the first combo sequence for the given weapon name, or null. */
  getBestCombo(weapon: string): ComboSequence | null {
    return BH_COMBOS[weapon]?.[0] ?? null
  }

  /**
   * Evaluate the Lady Vera flowchart for the current tracked state.
   * Requires setVeraConfig() to be called first.
   * @param detection Visual detection event from the capture layer
   */
  getVeraAction(detection: DetectionType = 'NONE'): VeraAction | null {
    if (!this._vera || !this._targetState || !this._myState) return null
    return veraController(
      this._vera.weapon,
      detection,
      this._targetState,
      this._myState.pos,
      this._vera.chakramMode,
    )
  }

  /** Number of inputs currently queued. */
  get queueLength() { return this._inputQueue.length }

  /** Clear all pending queued inputs. */
  clearQueue() { this._inputQueue.clear() }

  reset() {
    this.unigrams = {}
    this.bigrams = {}
    this.trigrams = {}
    this.history = []
    this.totalMoves = 0
  }
}
