/**
 * SimulationTab.tsx — Lady Vera simulation control panel.
 *
 * Sends toggle state to /api/active_sim/vera (SHM update, no immediate input)
 * and fires one-shot actions via /api/fight/vera.
 *
 * Latency path: button click → fetch (LAN loopback ~0.3ms) → Flask handler
 *               → shm_write_sim_state (ctypes memmove, ~1µs) → C++ reads next poll.
 */
import { useState, useCallback, useEffect } from 'react'

const API = 'http://localhost:5000'

// ── Types matching the Flask API ─────────────────────────────────────────────
type VeraWeapon    = 'Scythe' | 'Chakram'
type ChakramMode   = 'Split' | 'Fused'
type DetectionType = 'DODGE_BLUE' | 'SIG_YELLOW' | 'JUMP_SMOKE' | 'NONE'

interface SimState {
  strategy_id:  string
  weapon_id:    string
  chakram_mode: ChakramMode
  sim_active:   boolean
  urgent:       boolean
}

// ── Helpers ───────────────────────────────────────────────────────────────────
async function postJson(path: string, body: object): Promise<{ ok: boolean; [k: string]: unknown }> {
  const r = await fetch(`${API}${path}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  return r.json()
}

async function getJson(path: string): Promise<unknown> {
  const r = await fetch(`${API}${path}`)
  return r.json()
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function SimulationTab() {
  const [weapon,      setWeapon     ] = useState<VeraWeapon>('Scythe')
  const [chakramMode, setChakramMode] = useState<ChakramMode>('Split')
  const [detection,   setDetection  ] = useState<DetectionType>('NONE')
  const [simActive,   setSimActive  ] = useState(false)
  const [status,      setStatus     ] = useState<string>('Idle')
  const [shmState,    setShmState   ] = useState<SimState | null>(null)
  const [log,         setLog        ] = useState<string[]>([])

  const addLog = useCallback((msg: string) => {
    setLog(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev].slice(0, 50))
  }, [])

  // ── Push SHM state whenever any toggle changes ────────────────────────────
  const pushSimState = useCallback(async (
    overrides: Partial<{ weapon: VeraWeapon; mode: ChakramMode; active: boolean }>
  ) => {
    const w = overrides.weapon ?? weapon
    const m = overrides.mode   ?? chakramMode
    const a = overrides.active ?? simActive

    const res = await postJson('/api/active_sim/vera', {
      current_strategy: w === 'Scythe' ? 'scyO' : 'chkS',
      weapon_id:        w,
      chakram_mode:     m,
      sim_active:       a,
    })

    if (res.ok) {
      setShmState(res.shm as SimState)
      addLog(`SHM updated → weapon=${w} mode=${m} active=${a}`)
    } else {
      addLog(`SHM write failed: ${res.error ?? 'unknown'}`)
    }
  }, [weapon, chakramMode, simActive, addLog])

  // ── Toggle handlers ───────────────────────────────────────────────────────
  const handleWeaponChange = useCallback((w: VeraWeapon) => {
    setWeapon(w)
    pushSimState({ weapon: w })
  }, [pushSimState])

  const handleModeChange = useCallback((m: ChakramMode) => {
    setChakramMode(m)
    pushSimState({ mode: m })
  }, [pushSimState])

  const handleSimActiveToggle = useCallback(() => {
    const next = !simActive
    setSimActive(next)
    setStatus(next ? 'Armed' : 'Idle')
    pushSimState({ active: next })
  }, [simActive, pushSimState])

  // ── Fire one-shot Vera action ─────────────────────────────────────────────
  const fireVeraAction = useCallback(async () => {
    setStatus('Executing…')
    addLog(`Firing Vera: weapon=${weapon} detection=${detection} mode=${chakramMode}`)
    const res = await postJson('/api/fight/vera', {
      weapon,
      detection,
      chakram_mode: chakramMode,
    })
    if (res.ok) {
      setStatus('Done')
      addLog(`✓ Vera action executed (${weapon} / ${detection})`)
    } else {
      setStatus('Error')
      addLog(`✗ Error: ${res.error ?? 'unknown'}`)
    }
  }, [weapon, detection, chakramMode, addLog])

  // ── Poll SHM state for diagnostics ───────────────────────────────────────
  useEffect(() => {
    const id = setInterval(async () => {
      const s = await getJson('/api/active_sim/state') as SimState
      setShmState(s)
    }, 1000)
    return () => clearInterval(id)
  }, [])

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ padding: '1rem', fontFamily: 'monospace', maxWidth: 560 }}>
      <h2 style={{ margin: '0 0 1rem' }}>Simulation — Lady Vera</h2>

      {/* Status chip */}
      <div style={{ marginBottom: '1rem' }}>
        <span style={{
          padding: '2px 10px', borderRadius: 4, fontSize: 12, fontWeight: 700,
          background: simActive ? '#1a7a3c' : '#555', color: '#fff',
        }}>
          {status}
        </span>
      </div>

      {/* Weapon selector */}
      <section style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', marginBottom: 4 }}>Weapon</label>
        <div style={{ display: 'flex', gap: 8 }}>
          {(['Scythe', 'Chakram'] as VeraWeapon[]).map(w => (
            <button key={w}
              onClick={() => handleWeaponChange(w)}
              style={{
                padding: '6px 18px', borderRadius: 4, cursor: 'pointer', fontSize: 13,
                background: weapon === w ? '#2563eb' : '#333',
                color: '#fff', border: 'none',
              }}
            >{w}</button>
          ))}
        </div>
      </section>

      {/* Chakram mode toggle — only shown when Chakram is selected */}
      {weapon === 'Chakram' && (
        <section style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', marginBottom: 4 }}>Chakram Mode</label>
          <div style={{ display: 'flex', gap: 8 }}>
            {(['Split', 'Fused'] as ChakramMode[]).map(m => (
              <button key={m}
                onClick={() => handleModeChange(m)}
                style={{
                  padding: '6px 18px', borderRadius: 4, cursor: 'pointer', fontSize: 13,
                  background: chakramMode === m ? '#7c3aed' : '#333',
                  color: '#fff', border: 'none',
                }}
              >{m}</button>
            ))}
          </div>
          <small style={{ color: '#888' }}>
            Split = damage strings · Fused = kill confirms (≥100%)
          </small>
        </section>
      )}

      {/* Detection event override */}
      <section style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', marginBottom: 4 }}>Detection Override</label>
        <select
          value={detection}
          onChange={e => setDetection(e.target.value as DetectionType)}
          style={{ padding: '4px 8px', background: '#222', color: '#fff', border: '1px solid #444' }}
        >
          <option value="NONE">NONE (auto)</option>
          <option value="DODGE_BLUE">DODGE_BLUE</option>
          <option value="SIG_YELLOW">SIG_YELLOW</option>
          <option value="JUMP_SMOKE">JUMP_SMOKE</option>
        </select>
      </section>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 10, marginBottom: '1.5rem' }}>
        <button
          onClick={handleSimActiveToggle}
          style={{
            padding: '8px 20px', borderRadius: 4, cursor: 'pointer',
            background: simActive ? '#dc2626' : '#16a34a',
            color: '#fff', border: 'none', fontWeight: 700,
          }}
        >
          {simActive ? 'Deactivate' : 'Activate'}
        </button>

        <button
          onClick={fireVeraAction}
          disabled={!simActive}
          style={{
            padding: '8px 20px', borderRadius: 4, cursor: simActive ? 'pointer' : 'not-allowed',
            background: simActive ? '#ea580c' : '#444',
            color: '#fff', border: 'none', fontWeight: 700,
          }}
        >
          Fire Vera Action
        </button>
      </div>

      {/* SHM diagnostics */}
      {shmState && (
        <section style={{ marginBottom: '1rem', padding: '8px', background: '#111', borderRadius: 4 }}>
          <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>Shared Memory State</div>
          <table style={{ fontSize: 11, borderCollapse: 'collapse', width: '100%' }}>
            {Object.entries(shmState).map(([k, v]) => (
              <tr key={k}>
                <td style={{ color: '#aaa', paddingRight: 12 }}>{k}</td>
                <td style={{ color: '#fff' }}>{String(v)}</td>
              </tr>
            ))}
          </table>
        </section>
      )}

      {/* Activity log */}
      <section>
        <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>Log</div>
        <div style={{
          height: 130, overflowY: 'auto', background: '#0a0a0a',
          padding: '6px 8px', borderRadius: 4, fontSize: 11, color: '#6ee7b7',
        }}>
          {log.length === 0 ? <span style={{ color: '#444' }}>—</span>
            : log.map((l, i) => <div key={i}>{l}</div>)}
        </div>
      </section>
    </div>
  )
}
