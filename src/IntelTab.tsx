import { useState, useEffect } from 'react'

const API = 'http://localhost:5000'

type OmniBlock = Record<string, unknown>
type Rec = {
  id: string; legend?: string; weapon?: string; tier?: string
  confidence?: string; score?: number; notes?: string
  [k: string]: unknown
}
type Legend = {
  name: string; w1: string; w2: string
  w1_dsig?: { force: number; startup: number; rec: number }
  w1_nsig?: { force: number; startup: number; rec: number }
  [k: string]: unknown
}
type WeaponInfo = {
  name: string; tier: string; description: string
  combos: { name: string; notation: string }[]
  legends: string[]
}

const TIER_COLORS: Record<string, string> = {
  SS: '#fca5a5', A: '#fdba74', B: '#86efac', C: '#93c5fd', D: '#9ca3af',
}

export default function IntelTab() {
  const [intel, setIntel] = useState<OmniBlock | null>(null)
  const [recs, setRecs] = useState<Rec[]>([])
  const [legends, setLegends] = useState<Legend[]>([])
  const [weapons, setWeapons] = useState<WeaponInfo[]>([])
  const [tierList, setTierList] = useState<Record<string, string[]>>({})
  const [selectedLegend, setSelectedLegend] = useState<Legend | null>(null)
  const [selectedWeapon, setSelectedWeapon] = useState<WeaponInfo | null>(null)
  const [activePanel, setActivePanel] = useState<'intel' | 'recs' | 'legends' | 'weapons' | 'tiers'>('intel')
  const [ctx, setCtx] = useState({ legend: '', weapon: '', playstyle: '' })
  const [filterLegend, setFilterLegend] = useState('')
  const [filterWeapon, setFilterWeapon] = useState('')

  useEffect(() => {
    fetchIntel()
    fetchRecs()
    fetchLegends()
    fetchWeapons()
    fetchTiers()
  }, [])

  async function fetchIntel() {
    try { const r = await fetch(`${API}/api/coscore/intel`); if (r.ok) setIntel(await r.json()) } catch {}
  }
  async function fetchRecs() {
    try { const r = await fetch(`${API}/api/coscore/recommend`); if (r.ok) setRecs(await r.json()) } catch {}
  }
  async function fetchLegends() {
    try { const r = await fetch(`${API}/api/bh/legends`); if (r.ok) setLegends(await r.json()) } catch {}
  }
  async function fetchWeapons() {
    try { const r = await fetch(`${API}/api/bh/weapons`); if (r.ok) setWeapons(await r.json()) } catch {}
  }
  async function fetchTiers() {
    try { const r = await fetch(`${API}/api/bh/tier-list`); if (r.ok) setTierList(await r.json()) } catch {}
  }

  async function updateContext() {
    await fetch(`${API}/api/coscore/context`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(ctx),
    })
    fetchIntel()
  }

  async function selectLegend(name: string) {
    const r = await fetch(`${API}/api/bh/legend/${encodeURIComponent(name)}`)
    if (r.ok) setSelectedLegend(await r.json())
  }

  async function selectWeapon(name: string) {
    const r = await fetch(`${API}/api/bh/weapon/${encodeURIComponent(name)}`)
    if (r.ok) setSelectedWeapon(await r.json())
  }

  const filteredLegends = legends.filter(l =>
    !filterLegend || l.name?.toString().toLowerCase().includes(filterLegend.toLowerCase())
  )
  const filteredWeapons = weapons.filter(w =>
    !filterWeapon || w.name?.toString().toLowerCase().includes(filterWeapon.toLowerCase())
  )

  return (
    <div className="backend-panel">
      <div className="bp-header">
        <h2>🧠 Intelligence Hub</h2>
      </div>
      <p className="bp-desc">CoScore intel, pick recommendations, legend stats, weapon data, and tier list.</p>

      {/* Sub-nav */}
      <div className="bp-subnav">
        {(['intel', 'recs', 'legends', 'weapons', 'tiers'] as const).map(p => (
          <button key={p}
            className={`bp-subnav-btn ${activePanel === p ? 'bp-subnav-active' : ''}`}
            onClick={() => setActivePanel(p)}>
            {p === 'intel' ? '📊 Intel' :
             p === 'recs'  ? '🏆 Picks' :
             p === 'legends' ? '🧙 Legends' :
             p === 'weapons' ? '⚔️ Weapons' : '📈 Tier List'}
          </button>
        ))}
      </div>

      {/* ── Intel panel ── */}
      {activePanel === 'intel' && (
        <div>
          <div className="bp-section">
            <div className="bp-section-title">Session Context</div>
            <div className="bp-form-grid">
              <label className="bp-field">
                <span>Legend</span>
                <input value={ctx.legend} onChange={e => setCtx(c => ({ ...c, legend: e.target.value }))}
                  className="bp-input" placeholder="e.g. Lady Vera" />
              </label>
              <label className="bp-field">
                <span>Weapon</span>
                <input value={ctx.weapon} onChange={e => setCtx(c => ({ ...c, weapon: e.target.value }))}
                  className="bp-input" placeholder="e.g. Scythe" />
              </label>
              <label className="bp-field">
                <span>Playstyle</span>
                <input value={ctx.playstyle} onChange={e => setCtx(c => ({ ...c, playstyle: e.target.value }))}
                  className="bp-input" placeholder="e.g. aggressive_off" />
              </label>
            </div>
            <button className="btn btn-edit" style={{ marginTop: 8 }} onClick={updateContext}>
              Push Context
            </button>
          </div>
          <div className="bp-section">
            <div className="bp-section-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
              CoScore Omniblock Snapshot
              <button className="btn btn-cancel" style={{ padding: '3px 10px', fontSize: 11 }} onClick={fetchIntel}>↻ Refresh</button>
            </div>
            {intel ? (
              <div className="bp-intel-grid">
                {Object.entries(intel).map(([k, v]) => (
                  <div key={k} className="bp-intel-row">
                    <span className="bp-ctx-key">{k}</span>
                    <span className="bp-ctx-val">
                      {typeof v === 'object' ? JSON.stringify(v).slice(0, 80) : String(v)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: '#64748b', fontSize: 13 }}>No intel data — ensure Flask server is running.</p>
            )}
          </div>
        </div>
      )}

      {/* ── Picks panel ── */}
      {activePanel === 'recs' && (
        <div className="bp-section" style={{ flexGrow: 1 }}>
          <div className="bp-section-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
            Pick Recommendations
            <button className="btn btn-cancel" style={{ padding: '3px 10px', fontSize: 11 }} onClick={fetchRecs}>↻ Refresh</button>
          </div>
          {recs.length === 0 ? (
            <p style={{ color: '#64748b', fontSize: 13 }}>No recommendations in the library.</p>
          ) : (
            <table className="bp-table">
              <thead>
                <tr><th>Legend</th><th>Weapon</th><th>Tier</th><th>Confidence</th><th>Score</th><th>Notes</th></tr>
              </thead>
              <tbody>
                {recs.map(r => (
                  <tr key={r.id}>
                    <td style={{ fontWeight: 600, color: '#e2e8f0' }}>{r.legend ?? '—'}</td>
                    <td>{r.weapon ?? '—'}</td>
                    <td>
                      <span className="bp-tier-badge" style={{ color: TIER_COLORS[r.tier ?? ''] ?? '#64748b' }}>
                        {r.tier ?? '?'}
                      </span>
                    </td>
                    <td>{r.confidence ?? '—'}</td>
                    <td>{r.score != null ? r.score.toFixed(2) : '—'}</td>
                    <td style={{ color: '#64748b', fontSize: 11 }}>{r.notes ?? ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* ── Legends panel ── */}
      {activePanel === 'legends' && (
        <div>
          <input className="glossary-search" placeholder="🔍 Filter legends…"
            value={filterLegend} onChange={e => setFilterLegend(e.target.value)} />
          <div className="bp-legend-grid">
            {filteredLegends.map((l: Legend) => (
              <div key={String(l.name)} className={`bp-legend-card ${selectedLegend?.name === l.name ? 'bp-legend-active' : ''}`}
                onClick={() => selectLegend(String(l.name))}>
                <div className="bp-legend-name">{String(l.name)}</div>
                <div className="bp-legend-weapons">{String(l.w1)} / {String(l.w2)}</div>
              </div>
            ))}
          </div>
          {selectedLegend && (
            <div className="bp-section" style={{ marginTop: 12 }}>
              <div className="bp-section-title">{String(selectedLegend.name)} — Signature Data</div>
              <table className="bp-table">
                <thead>
                  <tr><th>Weapon</th><th>Direction</th><th>Force</th><th>Startup</th><th>Recovery</th></tr>
                </thead>
                <tbody>
                  {(['w1', 'w2'] as const).flatMap(wk =>
                    (['dsig', 'nsig', 'ssig'] as const).map(dk => {
                      const key = `${wk}_${dk}` as keyof Legend
                      const sig = selectedLegend[key] as { force: number; startup: number; rec: number } | undefined
                      if (!sig) return null
                      return (
                        <tr key={`${wk}-${dk}`}>
                          <td>{String(selectedLegend[wk])}</td>
                          <td style={{ color: '#94a3b8' }}>{dk.toUpperCase()}</td>
                          <td style={{ color: '#22c55e', fontWeight: 600 }}>{sig.force}</td>
                          <td style={{ color: sig.startup <= 15 ? '#22c55e' : sig.startup >= 25 ? '#ef4444' : '#f59e0b' }}>{sig.startup}</td>
                          <td style={{ color: '#94a3b8' }}>{sig.rec}</td>
                        </tr>
                      )
                    }).filter(Boolean)
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ── Weapons panel ── */}
      {activePanel === 'weapons' && (
        <div>
          <input className="glossary-search" placeholder="🔍 Filter weapons…"
            value={filterWeapon} onChange={e => setFilterWeapon(e.target.value)} />
          <div className="bp-weapon-grid">
            {filteredWeapons.map(w => (
              <div key={w.name}
                className={`bp-weapon-card ${selectedWeapon?.name === w.name ? 'bp-legend-active' : ''}`}
                onClick={() => selectWeapon(w.name)}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="bp-legend-name">{w.name}</span>
                  <span className="bp-tier-badge" style={{ color: TIER_COLORS[w.tier] ?? '#64748b' }}>{w.tier}</span>
                </div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{w.legends?.slice(0, 4).join(', ')}{(w.legends?.length ?? 0) > 4 ? '…' : ''}</div>
              </div>
            ))}
          </div>
          {selectedWeapon && (
            <div className="bp-section" style={{ marginTop: 12 }}>
              <div className="bp-section-title">{selectedWeapon.name} — {selectedWeapon.description}</div>
              {selectedWeapon.combos?.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '.4px' }}>Top Combos</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {selectedWeapon.combos.map((c, i) => (
                      <div key={i} className="bp-combo-row">
                        <span className="bp-chip bp-chip-blue">{c.name}</span>
                        <span style={{ fontSize: 12, color: '#94a3b8', fontFamily: 'monospace' }}>{c.notation}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Tier list panel ── */}
      {activePanel === 'tiers' && (
        <div className="bp-section">
          <div className="bp-section-title">Weapon Tier List</div>
          {Object.entries(tierList).map(([tier, ws]) => (
            <div key={tier} className="bp-tier-row">
              <span className="bp-tier-label" style={{ color: TIER_COLORS[tier] ?? '#64748b' }}>{tier}</span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {ws.map(w => (
                  <span key={w} className="bp-chip" style={{ borderColor: TIER_COLORS[tier] ?? '#333' }}>{w}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
