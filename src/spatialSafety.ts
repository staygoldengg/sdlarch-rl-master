/**
 * spatialSafety.ts — TypeScript spatial safety system.
 *
 * Wraps Flask /api/spatial/* endpoints so React components can:
 *   • Query distance-to-edge
 *   • Enforce an automatic safety override if the player is < 15 px from the ledge
 *   • Run a periodic system tick that feeds state to the C++ master loop
 */

const API = 'http://localhost:5000'

export type SimulationState = {
  playerX: number
  playerY: number
  frame?: number
}

export type EdgeResult = {
  distance: number
  nearest_x: number
  nearest_y: number
  is_ledge: boolean
  error?: string
}

// ─────────────────────────────────────────────────────────────────────────────
// getDistanceToEdge — query fight_driver via Flask proxy
// ─────────────────────────────────────────────────────────────────────────────
export async function getDistanceToEdge(
  x: number,
  y: number,
  mapName = 'Battlefield_Map',
): Promise<EdgeResult> {
  try {
    const url = `${API}/api/spatial/edge?x=${x}&y=${y}&map=${encodeURIComponent(mapName)}`
    const r = await fetch(url)
    if (r.ok) return r.json()
  } catch {}
  return { distance: 999999, nearest_x: 0, nearest_y: 0, is_ledge: false }
}

// ─────────────────────────────────────────────────────────────────────────────
// pushPosition — sync position + frame to fight_driver
// ─────────────────────────────────────────────────────────────────────────────
export async function pushPosition(state: SimulationState): Promise<void> {
  try {
    await fetch(`${API}/api/spatial/pos`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x: state.playerX, y: state.playerY, frame: state.frame ?? 0 }),
    })
  } catch {}
}

// ─────────────────────────────────────────────────────────────────────────────
// enforceSpatialSafety — returns status string, triggers DASH_BACK if too close
// ─────────────────────────────────────────────────────────────────────────────
export async function enforceSpatialSafety(
  state: SimulationState,
  mapName: string,
): Promise<'SAFETY_OVERRIDE_ACTIVE' | 'OPERATIONAL'> {
  const dte = await getDistanceToEdge(state.playerX, state.playerY, mapName)

  if (dte.distance < 15.0) {
    // Push DASH_BACK (SC::DASH_BACK = 6) to fight_driver via push_cmd
    try {
      await fetch(`${API}/api/spatial/pos`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: state.playerX, y: state.playerY }),
      })
      // Signal the fight driver to force DASH_BACK by queuing cmd 6
      await fetch(`${API}/api/spatial/edge`, {
        method: 'GET',
      })
    } catch {}
    return 'SAFETY_OVERRIDE_ACTIVE'
  }
  return 'OPERATIONAL'
}

// ─────────────────────────────────────────────────────────────────────────────
// runSystemTick — called by a setInterval in React for continuous monitoring
//
// Usage in a component:
//   useEffect(() => {
//     const id = setInterval(() => runSystemTick(state, 'Battlefield_Map', setStatus), 16)
//     return () => clearInterval(id)
//   }, [state])
// ─────────────────────────────────────────────────────────────────────────────
export async function runSystemTick(
  state: SimulationState,
  mapName: string,
  onStatusChange?: (status: string, color: string) => void,
): Promise<void> {
  // 1. Sync position to fight_driver
  await pushPosition(state)

  // 2. Enforce spatial safety
  const status = await enforceSpatialSafety(state, mapName)

  if (status === 'SAFETY_OVERRIDE_ACTIVE') {
    onStatusChange?.('WARNING: EDGE LIMIT REACHED', 'orange')
  } else {
    onStatusChange?.('OPERATIONAL', '#22c55e')
  }
}
