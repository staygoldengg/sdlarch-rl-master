import { useState } from 'react'

const API = 'http://localhost:5000'

type Vec2 = { x: number; y: number }

type StrategyResult = {
  strategy: {
    id: string; name: string; description: string
    priority: number; suggested_moves: string[]
  } | null
  lead_direction: Vec2 | null
  target_landing: { x: number; y: number; time_ms: number } | null
}

type AllStrategy = {
  id: string; name: string; description: string
  priority: number; suggested_moves: string[]
}

const PRIORITY_COLOR = (p: number) =>
  p >= 85 ? '#ef4444' : p >= 70 ? '#f59e0b' : p >= 50 ? '#22c55e' : '#64748b'

function defaultState(id: string) {
  return {
    player_id: id,
    pos: { x: 0, y: 0 },
    vel: { x: 0, y: 0 },
    is_airborne: false,
    is_invulnerable: false,
    is_attacking: false,
    current_move: '',
    last_move: '',
    buff_active: false,
    buff_expires_at: 0,
    stocks_remaining: 3,
    damage: 0,
  }
}

type PSForm = ReturnType<typeof defaultState>

function StateForm({
  label, state, onChange,
}: { label: string; state: PSForm; onChange: (s: PSForm) => void }) {
  function set<K extends keyof PSForm>(key: K, val: PSForm[K]) {
    onChange({ ...state, [key]: val })
  }
  return (
    <div className="bp-state-form">
      <div className="bp-section-title">{label}</div>
      <div className="bp-form-grid2">
        <label className="bp-field">
          <span>Pos X</span>
          <input type="number" value={state.pos.x} className="bp-input"
            onChange={e => set('pos', { ...state.pos, x: +e.target.value })} />
        </label>
        <label className="bp-field">
          <span>Pos Y</span>
          <input type="number" value={state.pos.y} className="bp-input"
            onChange={e => set('pos', { ...state.pos, y: +e.target.value })} />
        </label>
        <label className="bp-field">
          <span>Vel X</span>
          <input type="number" value={state.vel.x} className="bp-input"
            onChange={e => set('vel', { ...state.vel, x: +e.target.value })} />
        </label>
        <label className="bp-field">
          <span>Vel Y</span>
          <input type="number" value={state.vel.y} className="bp-input"
            onChange={e => set('vel', { ...state.vel, y: +e.target.value })} />
        </label>
        <label className="bp-field">
          <span>Damage %</span>
          <input type="number" value={state.damage} min={0} max={300} className="bp-input"
            onChange={e => set('damage', +e.target.value)} />
        </label>
        <label className="bp-field">
          <span>Stocks</span>
          <input type="number" value={state.stocks_remaining} min={0} max={5} className="bp-input"
            onChange={e => set('stocks_remaining', +e.target.value)} />
        </label>
      </div>
      <div className="bp-row" style={{ gap: 16, marginTop: 6, flexWrap: 'wrap' }}>
        {(['is_airborne', 'is_invulnerable', 'is_attacking', 'buff_active'] as const).map(k => (
          <label key={k} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: '#94a3b8', cursor: 'pointer' }}>
            <input type="checkbox" checked={state[k] as boolean}
              onChange={e => set(k, e.target.checked as PSForm[typeof k])} />
            {k.replace('is_', '').replace('_', ' ')}
          </label>
        ))}
      </div>
    </div>
  )
}

export default function StrategyTab() {
  const [myState, setMyState] = useState<PSForm>(defaultState('player'))
  const [targetState, setTargetState] = useState<PSForm>(defaultState('enemy'))
  const [projSpeed, setProjSpeed] = useState(600)
  const [result, setResult] = useState<StrategyResult | null>(null)
  const [allStrategies, setAllStrategies] = useState<AllStrategy[]>([])
  const [loading, setLoading] = useState(false)
  const [allLoading, setAllLoading] = useState(false)
  const [coachTip, setCoachTip] = useState('')

  async function pushState() {
    await fetch(`${API}/api/npc/state`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ my: myState, target: targetState }),
    })
  }

  async function getBestStrategy() {
    setLoading(true)
    await pushState()
    const r = await fetch(`${API}/api/npc/strategy?proj_speed=${projSpeed}`)
    if (r.ok) setResult(await r.json())
    setLoading(false)
  }

  async function getAllStrategies() {
    setAllLoading(true)
    await pushState()
    const r = await fetch(`${API}/api/npc/strategies/all`)
    if (r.ok) setAllStrategies(await r.json())
    setAllLoading(false)
  }

  async function getCoachTip() {
    const state = [myState.pos.x, myState.pos.y, myState.vel.x, myState.vel.y, myState.damage]
    const r = await fetch(`${API}/api/learning/tip`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: Array(128).fill(0).map((_, i) => state[i] ?? 0) }),
    })
    if (r.ok) { const d = await r.json(); setCoachTip(d.tip) }
  }

  async function predictLanding() {
    const r = await fetch(`${API}/api/npc/predict-landing`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...targetState }),
    })
    if (r.ok) {
      const d = await r.json()
      if (d.landing) {
        setResult(prev => ({
          ...prev!,
          target_landing: d.landing,
          strategy: prev?.strategy ?? null,
          lead_direction: prev?.lead_direction ?? null,
        }))
      }
    }
  }

  return (
    <div className="backend-panel">
      <div className="bp-header">
        <h2>🧬 NPC Strategy Engine</h2>
      </div>
      <p className="bp-desc">
        Push live player states to the physics engine. Get the highest-priority strategy,
        projectile lead direction, and landing prediction.
      </p>

      <div className="bp-strategy-layout">
        {/* Left: state inputs */}
        <div className="bp-strategy-left">
          <StateForm label="My State" state={myState} onChange={setMyState} />
          <StateForm label="Target State" state={targetState} onChange={setTargetState} />
          <div className="bp-field" style={{ marginTop: 8 }}>
            <span style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: '.4px' }}>Projectile Speed (px/s)</span>
            <input type="number" value={projSpeed} min={100} max={2000}
              onChange={e => setProjSpeed(+e.target.value)} className="bp-input" />
          </div>
          <div className="bp-row" style={{ marginTop: 10, flexWrap: 'wrap' }}>
            <button className="btn btn-edit" onClick={getBestStrategy} disabled={loading}>
              {loading ? '…' : '🎯 Best Strategy'}
            </button>
            <button className="btn btn-save" onClick={getAllStrategies} disabled={allLoading}>
              {allLoading ? '…' : '📋 All Applicable'}
            </button>
            <button className="btn btn-cancel" onClick={getCoachTip}>💬 Coaching Tip</button>
            <button className="btn btn-cancel" onClick={predictLanding} disabled={!targetState.is_airborne}>
              📍 Predict Landing
            </button>
          </div>
        </div>

        {/* Right: results */}
        <div className="bp-strategy-right">
          {coachTip && (
            <div className="bp-result-card bp-result-tip">
              <div className="bp-result-label">💬 Coaching Tip</div>
              <p>{coachTip}</p>
            </div>
          )}

          {result?.strategy && (
            <div className="bp-result-card">
              <div className="bp-result-label">🎯 Best Strategy</div>
              <div className="bp-strat-name">{result.strategy.name}</div>
              <div className="bp-strat-desc">{result.strategy.description}</div>
              <div style={{ marginTop: 6, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <span className="bp-chip" style={{ color: PRIORITY_COLOR(result.strategy.priority) }}>
                  Priority {result.strategy.priority}
                </span>
                <span className="bp-chip">{result.strategy.id}</span>
              </div>
              <div style={{ marginTop: 8 }}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Suggested Moves</div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {result.strategy.suggested_moves.map(m => (
                    <span key={m} className="bp-chip bp-chip-blue">{m}</span>
                  ))}
                </div>
              </div>
            </div>
          )}

          {result?.lead_direction && (
            <div className="bp-result-card">
              <div className="bp-result-label">➡️ Lead Direction (projectile intercept)</div>
              <div style={{ fontFamily: 'monospace', fontSize: 13 }}>
                x: {result.lead_direction.x.toFixed(4)}, y: {result.lead_direction.y.toFixed(4)}
              </div>
            </div>
          )}

          {result?.target_landing && (
            <div className="bp-result-card">
              <div className="bp-result-label">📍 Predicted Landing</div>
              <div style={{ fontFamily: 'monospace', fontSize: 13 }}>
                x: {result.target_landing.x}, y: {result.target_landing.y}
              </div>
              <div style={{ fontSize: 12, color: '#f59e0b', marginTop: 4 }}>
                ⏱ {result.target_landing.time_ms} ms until landing
              </div>
            </div>
          )}

          {allStrategies.length > 0 && (
            <div className="bp-result-card">
              <div className="bp-result-label">📋 All Applicable Strategies ({allStrategies.length})</div>
              <div className="bp-all-strats">
                {allStrategies.map(s => (
                  <div key={s.id} className="bp-strat-row">
                    <span className="bp-strat-row-name">{s.name}</span>
                    <span className="bp-chip" style={{ color: PRIORITY_COLOR(s.priority), fontSize: 11 }}>
                      {s.priority}
                    </span>
                    <span style={{ fontSize: 11, color: '#64748b' }}>{s.description}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {!result && allStrategies.length === 0 && !coachTip && (
            <div style={{ color: '#64748b', fontSize: 13, marginTop: 16 }}>
              Set player states and click a button above to query the strategy engine.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
