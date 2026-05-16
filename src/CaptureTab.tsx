import { useState, useEffect, useRef } from 'react'
import LatencyMonitor from './LatencyMonitor'

const API = 'http://localhost:5000'

// ── useLiveIndex ──────────────────────────────────────────────────────────────
// Sends a live screenshot to /api/translate/index and returns the current
// weapon index. Automatically cancels stale in-flight requests.
function useLiveIndex(triggerMs = 3000): number {
  const [currentIndex, setCurrentIndex] = useState(100)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    let alive = true

    async function fetchIndex() {
      // Cancel previous request so we never process a stale response
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller
      try {
        const res = await fetch(`${API}/api/translate/index`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
          signal: controller.signal,
        })
        const data = await res.json()
        if (alive && !controller.signal.aborted) {
          setCurrentIndex(data.weaponID ?? 100)
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name !== 'AbortError') console.warn('[useLiveIndex]', err)
      }
    }

    fetchIndex()
    const id = setInterval(fetchIndex, triggerMs)
    return () => {
      alive = false
      clearInterval(id)
      abortRef.current?.abort()
    }
  }, [triggerMs])

  return currentIndex
}

type Preset = {
  name: string
  game: string
  legend: string
  weapon: string
  tier: string
  frames: number
  size_mb: number
  created: string
}

const LEGENDS = [
  '', 'Ada', 'Arcadia', 'Artemis', 'Asuri', 'Azoth', 'Barraza', 'Bodvar', 'Brynn',
  'Caspian', 'Cassidy', 'Cross', 'Diana', 'Ember', 'Fait', 'Gnash', 'Hattori',
  'Isaiah', 'Jaeyun', 'Jhala', 'Jiro', 'Katarina', 'Kaya', 'Koji', 'Kor',
  'Lady Vera', 'Lin Fei', 'Loki', 'Lucien', 'Magyar', 'Mako', 'Mirage', 'Mordex',
  'Munin', 'Nix', 'Onyx', 'Orion', 'Petra', 'Priya', 'Queen Nai', 'Ragnir',
  'Rayman', 'Reno', 'Rupture', 'Scarlet', 'Sentinel', 'Sidra', 'Sir Roland',
  'TarTar', 'Teros', 'Thatch', 'Thor', 'Ulgrim', 'Val', 'Vector', 'Volkov',
  'Wu Shang', 'Xull', 'Yumiko', 'Zariel',
]

const WEAPONS = [
  '', 'Gauntlets', 'Katars', 'Bow', 'Scythe', 'Blasters', 'Orb', 'Axe',
  'Sword', 'Spear', 'Hammer', 'Lance', 'Cannon', 'Greatsword', 'Battle Boots', 'Chakram',
]

const TIER_COLORS: Record<string, string> = {
  SS: '#fca5a5', A: '#fdba74', B: '#86efac', C: '#93c5fd', D: '#9ca3af', '?': '#64748b',
}

export default function CaptureTab() {
  const [presets, setPresets] = useState<Preset[]>([])
  const [recording, setRecording] = useState(false)
  const weaponIndex = useLiveIndex(3000)    // auto-polls OCR every 3 s
  const [frameCount, setFrameCount] = useState(0)
  const [presetName, setPresetName] = useState('brawlhalla_default')
  const [legend, setLegend] = useState('')
  const [weapon, setWeapon] = useState('')
  const [monitorIdx, setMonitorIdx] = useState(0)
  const [maxFrames, setMaxFrames] = useState(10000)
  const [loadedPreset, setLoadedPreset] = useState<string | null>(null)
  const [loadedCtx, setLoadedCtx] = useState<Record<string, unknown> | null>(null)
  // Tauri-safe delete confirmation (replaces window.confirm which is blocked)
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)

  useEffect(() => {
    loadPresets()
    const id = setInterval(pollStatus, 1500)
    return () => clearInterval(id)
  }, [])

  async function loadPresets() {
    try {
      const r = await fetch(`${API}/api/presets`)
      if (r.ok) setPresets(await r.json())
    } catch {}
  }

  async function pollStatus() {
    try {
      const r = await fetch(`${API}/api/status`)
      if (r.ok) {
        const d = await r.json()
        setRecording(d.capture === 'recording')
        setFrameCount(d.frames)
        if (d.preset) setLoadedPreset(d.preset)
      }
    } catch {}
  }

  async function startCapture() {
    const saved_binds = localStorage.getItem('bh_fight_binds')
    const binds = saved_binds ? JSON.parse(saved_binds) : {}
    await fetch(`${API}/api/capture/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        preset_name: presetName,
        legend,
        weapon,
        monitor_idx: monitorIdx,
        max_frames: maxFrames,
        binds,
      }),
    })
  }

  async function stopCapture() {
    await fetch(`${API}/api/capture/stop`, { method: 'POST' })
    setTimeout(loadPresets, 2500)
  }

  async function loadPreset(name: string) {
    const r = await fetch(`${API}/api/presets/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    })
    if (r.ok) {
      const d = await r.json()
      setLoadedPreset(name)
      setLoadedCtx(d.context || null)
    }
  }

  async function deletePreset(name: string) {
    if (pendingDelete !== name) { setPendingDelete(name); return }
    setPendingDelete(null)
    await fetch(`${API}/api/presets/${encodeURIComponent(name)}`, { method: 'DELETE' })
    loadPresets()
    if (loadedPreset === name) { setLoadedPreset(null); setLoadedCtx(null) }
  }

  const captureProgress = maxFrames > 0 ? Math.min(100, (frameCount / maxFrames) * 100) : 0

  return (
    <div className="backend-panel">
      <div className="bp-header">
        <h2>📷 Capture & Presets</h2>
        <span className={`bp-conn ${recording ? 'bp-conn-rec' : 'bp-conn-off'}`}>
          {recording ? '⏺ Recording' : '○ Idle'}
        </span>
      </div>
      <p className="bp-desc">Record screen frames for imitation learning. Saved as numpy presets.</p>

      {/* Record Controls */}
      <div className="bp-section">
        <div className="bp-section-title">New Recording</div>
        <div className="bp-form-grid">
          <label className="bp-field">
            <span>Preset Name</span>
            <input value={presetName} onChange={e => setPresetName(e.target.value)}
              className="bp-input" placeholder="e.g. scythe_offstage_01" />
          </label>
          <label className="bp-field">
            <span>Legend</span>
            <select value={legend} onChange={e => setLegend(e.target.value)} className="legend-select">
              {LEGENDS.map(l => <option key={l} value={l}>{l || '— Any —'}</option>)}
            </select>
          </label>
          <label className="bp-field">
            <span>Weapon</span>
            <select value={weapon} onChange={e => setWeapon(e.target.value)} className="legend-select">
              {WEAPONS.map(w => <option key={w} value={w}>{w || '— Any —'}</option>)}
            </select>
          </label>
          <label className="bp-field">
            <span>Monitor</span>
            <select value={monitorIdx} onChange={e => setMonitorIdx(+e.target.value)} className="legend-select">
              <option value={0}>Monitor 0 (Primary)</option>
              <option value={1}>Monitor 1</option>
              <option value={2}>Monitor 2</option>
            </select>
          </label>
          <label className="bp-field">
            <span>Max Frames (0 = unlimited)</span>
            <input type="number" value={maxFrames} min={0} step={1000}
              onChange={e => setMaxFrames(+e.target.value)} className="bp-input" />
          </label>
        </div>
        <div className="bp-row" style={{ marginTop: 10 }}>
          <button className="btn btn-run" onClick={startCapture} disabled={recording}>⏺ Start Recording</button>
          <button className="btn btn-delete" onClick={stopCapture} disabled={!recording}>⏹ Stop & Save</button>
        </div>
        {recording && (
          <div style={{ marginTop: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>
              <span>{frameCount.toLocaleString()} frames captured</span>
              <span>{captureProgress.toFixed(1)}%</span>
            </div>
            <div style={{ height: 6, background: '#1e2130', borderRadius: 3 }}>
              <div style={{ height: '100%', width: `${captureProgress}%`, background: '#ef4444', borderRadius: 3, transition: 'width .3s' }} />
            </div>
          </div>
        )}
      </div>

      {/* Loaded preset context */}
      {loadedPreset && loadedCtx && (
        <div className="bp-section">
          <div className="bp-section-title">📂 Loaded: {loadedPreset}</div>
          <div className="bp-ctx-grid">
            {Object.entries(loadedCtx).map(([k, v]) => (
              <div key={k} className="bp-ctx-row">
                <span className="bp-ctx-key">{k}</span>
                <span className="bp-ctx-val">{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Weapon index from live OCR */}
      <div className="bp-section">
        <div className="bp-section-title">🎯 Live Weapon Index (OCR)</div>
        <div className="bp-row" style={{ alignItems: 'center', gap: 12 }}>
          <span className="bp-chip bp-chip-blue" style={{ fontSize: 13, padding: '3px 14px' }}>
            ID: {weaponIndex}
          </span>
          <span style={{ color: '#94a3b8', fontSize: 12 }}>
            {weaponIndex === 100 ? 'Unarmed' :
             weaponIndex === 101 ? 'Scythe' :
             weaponIndex === 102 ? 'Spear' :
             weaponIndex === 103 ? 'Chakram' :
             weaponIndex === 104 ? 'Gauntlets' :
             weaponIndex === 105 ? 'Katars' :
             weaponIndex === 106 ? 'Bow' :
             weaponIndex === 107 ? 'Blasters' :
             weaponIndex === 108 ? 'Orb' :
             weaponIndex === 109 ? 'Axe' :
             weaponIndex === 110 ? 'Sword' :
             weaponIndex === 111 ? 'Hammer' :
             weaponIndex === 112 ? 'Lance' :
             weaponIndex === 113 ? 'Cannon' :
             weaponIndex === 114 ? 'Greatsword' :
             weaponIndex === 115 ? 'Battle Boots' : 'Unknown'}
          </span>
          <span style={{ color: '#475569', fontSize: 11 }}>(polls every 3 s via OCR)</span>
        </div>
      </div>

      {/* Latency monitor */}
      <LatencyMonitor />

      {/* Presets list */}
      <div className="bp-section" style={{ flexGrow: 1 }}>
        <div className="bp-section-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
          Saved Presets
          <button className="btn btn-cancel" style={{ padding: '3px 10px', fontSize: 11 }} onClick={loadPresets}>↻ Refresh</button>
        </div>
        {presets.length === 0 ? (
          <p style={{ color: '#64748b', fontSize: 13 }}>No presets saved yet. Record some gameplay first.</p>
        ) : (
          <table className="bp-table">
            <thead>
              <tr>
                <th>Name</th><th>Legend</th><th>Weapon</th><th>Tier</th>
                <th>Frames</th><th>Size</th><th>Created</th><th></th>
              </tr>
            </thead>
            <tbody>
              {presets.map(p => (
                <tr key={p.name} className={loadedPreset === p.name ? 'bp-row-active' : ''}>
                  <td style={{ fontWeight: 600, color: '#e2e8f0' }}>{p.name}</td>
                  <td style={{ color: '#94a3b8' }}>{p.legend || '—'}</td>
                  <td style={{ color: '#94a3b8' }}>{p.weapon || '—'}</td>
                  <td>
                    <span className="bp-tier-badge" style={{ color: TIER_COLORS[p.tier] ?? '#64748b' }}>
                      {p.tier}
                    </span>
                  </td>
                  <td>{p.frames.toLocaleString()}</td>
                  <td>{p.size_mb} MB</td>
                  <td style={{ color: '#64748b', fontSize: 11 }}>{p.created}</td>
                  <td>
                    <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                      <button className="btn btn-edit" style={{ padding: '3px 8px', fontSize: 11 }}
                        onClick={() => loadPreset(p.name)}>Load</button>
                      {pendingDelete === p.name ? (
                        <>
                          <button className="btn btn-delete" style={{ padding: '3px 8px', fontSize: 11 }}
                            onClick={() => deletePreset(p.name)}>Confirm</button>
                          <button className="btn btn-cancel" style={{ padding: '3px 6px', fontSize: 11 }}
                            onClick={() => setPendingDelete(null)}>✕</button>
                        </>
                      ) : (
                        <button className="btn btn-delete" style={{ padding: '3px 8px', fontSize: 11 }}
                          onClick={() => deletePreset(p.name)}>🗑</button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
