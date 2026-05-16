import { useState, useEffect, useRef } from 'react'
import {
  checkHealth, policyInfer, executeMacro, listMacros, rlTrain, rlSave, rlStats,
  startLoop, stopLoop, getLoopStatus, getOBSStatus, launchOBS, installOBS, ensureOBS, calibrateOBS,
  ingestVideo, getVideoStatus, pretrainFromVideo, resetVideo, getVideoKnowledge,
  getBrainInfo, saveBrain, reloadBrain, clearBrain,
  replayScan, replayIngest, replayIngestAll,
  getMemoryInfo, getMemoryState, rescanMemory,
  getBTRStats, saveBTR, btrPretrain,
  type InferResult, type LoopStatus, type OBSStatus, type AgentStats, type VideoStatus,
  type BrainInfo, type ReplayScanResult, type ReplayFileMeta, type MemoryInfo, type BTRStats,
} from './WeaponizedAPI'

const API = 'http://localhost:5000'

// ── useMetaUpdates ────────────────────────────────────────────────────────────
// Polls /api/meta/latest every 60 s and reflects current weapon trends +
// tournament context into the Fight tab UI.
type MetaState = {
  weapon_trends: Record<string, { priority: number; trend: number }>
  tournament: { multiplier: number; strategy: string; match_type: string; viewer_count?: number }
  viewer_count: number
  last_updated: string | null
}

function useMetaUpdates(intervalMs = 60_000): MetaState {
  const [meta, setMeta] = useState<MetaState>({
    weapon_trends: {},
    tournament: { multiplier: 1.0, strategy: 'STANDARD', match_type: 'Unknown' },
    viewer_count: 0,
    last_updated: null,
  })

  useEffect(() => {
    async function fetchMeta() {
      try {
        const res = await fetch(`${API}/api/meta/latest`)
        if (res.ok) setMeta(await res.json())
      } catch {}
    }
    fetchMeta()
    const id = setInterval(fetchMeta, intervalMs)
    return () => clearInterval(id)
  }, [intervalMs])

  return meta
}

type FightStatus = {
  status: string
  last_strategy: string | null
  move_count: number
  tick_ms: number
  binds: Record<string, string>
  queue_size: number
}

const DEFAULT_BINDS: Record<string, string> = {
  light: 'N', heavy: 'M', jump: 'SPACE',
  left: 'A', right: 'D', up: 'W', down: 'S',
  dodge: 'E', pickup: 'F',
}

const MOVE_LIST = [
  'NLight', 'SLight', 'DLight', 'NHeavy', 'SHeavy', 'DHeavy',
  'Nair', 'Sair', 'Dair', 'Jump', 'DJ', 'Dodge', 'GC', 'Dash', 'WT', 'Pickup',
  'NSig', 'SSig', 'DSig', 'GP', 'PlatDrop',
]

export default function FightTab() {
  const [status, setStatus] = useState<FightStatus | null>(null)
  const [binds, setBinds] = useState<Record<string, string>>(() => {
    try {
      const saved = localStorage.getItem('bh_fight_binds')
      return saved ? JSON.parse(saved) : DEFAULT_BINDS
    } catch {
      return DEFAULT_BINDS
    }
  })
  const [tickMs, setTickMs] = useState(150)
  const [selectedMove, setSelectedMove] = useState('NLight')
  const [log, setLog] = useState<string[]>([])
  const [connected, setConnected] = useState(false)
  const [hitResult, setHitResult] = useState<{ result: string; accuracy: number } | null>(null)
  const esRef = useRef<EventSource | null>(null)
  const meta = useMetaUpdates(60_000)

  // ── Weaponized AI state ────────────────────────────────────────────────────
  const [aiOnline, setAiOnline] = useState(false)
  const [macros, setMacros] = useState<string[]>([])
  const [selectedMacro, setSelectedMacro] = useState('nlight')
  const [inferResult, setInferResult] = useState<InferResult | null>(null)
  const [trainResult, setTrainResult] = useState<{ loss_policy: number; loss_value: number } | null>(null)
  const DEMO_OBS = Array(18).fill(0).map((_, i) => parseFloat((Math.random() * 2 - 1).toFixed(3)))

  // ── OBS Training Loop state ────────────────────────────────────────────────
  const [loopStatus, setLoopStatus] = useState<LoopStatus | null>(null)
  const [captureMode, setCaptureMode] = useState<'mss' | 'obs'>('mss')
  const loopPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── OBS Manager state ──────────────────────────────────────────────────────
  const [obsStatus, setObsStatus] = useState<OBSStatus | null>(null)
  const obsInstallPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [calibrateResult, setCalibrateResult] = useState<string | null>(null)
  const [agentStats, setAgentStats] = useState<AgentStats | null>(null)
  // ── Brain Storage state ────────────────────────────────────────────
  const [brainInfo, setBrainInfo]         = useState<BrainInfo | null>(null)
  const [brainSaving, setBrainSaving]     = useState(false)
  const [brainMsg, setBrainMsg]           = useState<string | null>(null)
  const [showRegistry, setShowRegistry]   = useState(false)
  // ── Replay Engine state ────────────────────────────────────────────
  const [replayScanData, setReplayScanData]         = useState<ReplayScanResult | null>(null)
  const [replayScanning, setReplayScanning]         = useState(false)
  const [replayMsg, setReplayMsg]                   = useState<string | null>(null)
  const [selectedReplay, setSelectedReplay]         = useState<string | null>(null)
  const [replayIngesting, setReplayIngesting]       = useState(false)
  const [replayIngestingAll, setReplayIngestingAll] = useState(false)
  // ── Live Memory state ──────────────────────────────────────────────
  const [memInfo, setMemInfo]     = useState<MemoryInfo | null>(null)
  const [memState, setMemState]   = useState<{p1: {damage: number; stocks: number}; p2: {damage: number; stocks: number}; source: string} | null>(null)
  const [memPolling, setMemPolling] = useState(false)
  const memPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  // ── BTR Agent state ────────────────────────────────────────────────
  const [btrStats, setBtrStats]       = useState<BTRStats | null>(null)
  const [btrPretraining, setBtrPretraining] = useState(false)
  const [btrMsg, setBtrMsg]           = useState<string | null>(null)
  // ── YouTube Learning state ─────────────────────────────────────────────────
  const [videoUrl, setVideoUrl]           = useState('')
  const [videoTranscribe, setVideoTranscribe] = useState(true)
  const [videoStatus, setVideoStatus]     = useState<VideoStatus | null>(null)
  const [videoLogOpen, setVideoLogOpen]   = useState(false)
  const [videoKnowledge, setVideoKnowledge] = useState<{term: string; count: number}[]>([])
  const videoPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    localStorage.setItem('bh_fight_binds', JSON.stringify(binds))
  }, [binds])

  useEffect(() => {
    pollStatus()
    const id = setInterval(pollStatus, 2000)
    const es = new EventSource(`${API}/api/log-stream`)
    esRef.current = es
    es.onopen = () => setConnected(true)
    es.onerror = () => setConnected(false)
    es.onmessage = (e) => {
      if (memPollRef.current) clearInterval(memPollRef.current)
      try {
        const msg = JSON.parse(e.data) as string
        setLog(l => [msg, ...l].slice(0, 60))
      } catch {}
    }
    // Check weaponized AI server
    checkHealth().then(() => {
      setAiOnline(true)
      listMacros().then(r => { setMacros(r.macros); setSelectedMacro(r.macros[0] ?? 'nlight') })
      getOBSStatus().then(setObsStatus).catch(() => {})
      rlStats().then(setAgentStats).catch(() => {})
    }).catch(() => setAiOnline(false))
    return () => {
      clearInterval(id); es.close()
      if (loopPollRef.current) clearInterval(loopPollRef.current)
      if (obsInstallPollRef.current) clearInterval(obsInstallPollRef.current)
    }
  }, [])

  async function pollStatus() {
    try {
      const r = await fetch(`${API}/api/fight/status`)
      if (r.ok) { setStatus(await r.json()); setConnected(true) }
    } catch { setConnected(false) }
  }

  async function post(path: string, body?: object) {
    try {
      const r = await fetch(`${API}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined,
      })
      if (r.ok) { await pollStatus() }
      return r
    } catch { return null }
  }

  async function handleStartLoop() {
    await startLoop({ capture_mode: captureMode }).catch(console.error)
    loopPollRef.current = setInterval(async () => {
      try {
        const s = await getLoopStatus()
        setLoopStatus(s)
        // Update agent stats from loop status data
        setAgentStats({
          total_steps: s.total_updates ?? 0,
          total_updates: s.total_updates ?? 0,
          episode_count: s.episode_count ?? 0,
          mean_ep_reward: s.mean_ep_reward ?? 0,
          best_ep_reward: s.best_ep_reward ?? 0,
        })
        if (!s.running && loopPollRef.current) {
          clearInterval(loopPollRef.current)
          loopPollRef.current = null
        }
      } catch {}
    }, 1000)
  }

  async function handleStopLoop() {
    await stopLoop().catch(console.error)
    if (loopPollRef.current) { clearInterval(loopPollRef.current); loopPollRef.current = null }
    getLoopStatus().then(setLoopStatus).catch(() => {})
  }

  async function handleInstallOBS() {
    await installOBS().catch(console.error)
    // Poll until install finishes
    obsInstallPollRef.current = setInterval(async () => {
      try {
        const s = await getOBSStatus()
        setObsStatus(s)
        if (s.install_done || s.install_error || (!s.downloading && !s.installing)) {
          if (obsInstallPollRef.current) { clearInterval(obsInstallPollRef.current); obsInstallPollRef.current = null }
        }
      } catch {}
    }, 1200)
  }

  async function handleLaunchOBS() {
    const r = await launchOBS().catch(() => null)
    if (r) setTimeout(() => getOBSStatus().then(setObsStatus).catch(() => {}), 2000)
  }

  async function handleEnsureOBS() {
    const s = await ensureOBS().catch(() => null)
    if (s) setObsStatus(s)
    if (s && (s.downloading || s.installing)) {
      obsInstallPollRef.current = setInterval(async () => {
        try {
          const ns = await getOBSStatus()
          setObsStatus(ns)
          if (!ns.downloading && !ns.installing) {
            if (obsInstallPollRef.current) { clearInterval(obsInstallPollRef.current); obsInstallPollRef.current = null }
          }
        } catch {}
      }, 1200)
    }
  }

  async function handleCalibrate() {
    setCalibrateResult('Detecting Brawlhalla window…')
    try {
      const r = await calibrateOBS()
      const w = r.window
      setCalibrateResult(`✓ ${w.width}×${w.height} at (${w.left},${w.top})`)
    } catch (e: unknown) {
      setCalibrateResult(`⚠ ${e instanceof Error ? e.message : 'Not found — is Brawlhalla running?'}`)
    }
  }

  function _startVideoPoll() {
    if (videoPollRef.current) return
    videoPollRef.current = setInterval(async () => {
      try {
        const s = await getVideoStatus()
        setVideoStatus(s)
        if (s.state === 'done') {
          clearInterval(videoPollRef.current!)
          videoPollRef.current = null
          // Fetch full knowledge base when done
          getVideoKnowledge().then(r => setVideoKnowledge(r.terms)).catch(() => {})
        } else if (s.state === 'error' || s.state === 'idle') {
          clearInterval(videoPollRef.current!)
          videoPollRef.current = null
        }
      } catch {}
    }, 1200)
  }

  async function handleIngestVideo() {
    if (!videoUrl.trim()) return
    try {
      await ingestVideo(videoUrl.trim(), 8000, videoTranscribe)
      const s = await getVideoStatus()
      setVideoStatus(s)
      _startVideoPoll()
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : String(e))
    }
  }

  async function handlePretrain() {
    try {
      await pretrainFromVideo()
      const s = await getVideoStatus()
      setVideoStatus(s)
      _startVideoPoll()
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : String(e))
    }
  }

  async function handleVideoReset() {
    await resetVideo().catch(() => {})
    setVideoStatus(null)
    setVideoKnowledge([])
  }

  async function armRefAndFire() {    // 1. Snapshot the before-swing reference frame
    await fetch(`${API}/api/fight/arm-ref`, { method: 'POST' })
    // 2. Fire the selected move
    await post('/api/fight/exec-move', { move: selectedMove })
    // 3. Wait ~80 ms (5 frames) then check hit-confirm
    setTimeout(async () => {
      try {
        const r = await fetch(`${API}/api/fight/hit-confirm`, { method: 'POST' })
        if (r.ok) setHitResult(await r.json())
      } catch {}
    }, 80)
  }

  const statusColor =
    status?.status === 'fighting' ? '#22c55e' :
    status?.status === 'armed'    ? '#f59e0b' :
    status?.status === 'stopping' ? '#ef4444' : '#64748b'

  return (
    <div className="backend-panel">
      <div className="bp-header">
        <h2>⚔️ Fight Engine</h2>
        <span className={`bp-conn ${connected ? 'bp-conn-on' : 'bp-conn-off'}`}>
          {connected ? '● Backend' : '○ Offline'}
        </span>
      </div>
      <p className="bp-desc">Autonomous training fight loop — connects to Flask at localhost:5000</p>

      {/* Status strip */}
      <div className="bp-status-row">
        <span className="bp-dot" style={{ background: statusColor }} />
        <span className="bp-status-label">{status?.status ?? '—'}</span>
        {status?.last_strategy && (
          <span className="bp-chip bp-chip-blue">{status.last_strategy}</span>
        )}
        <span className="bp-chip">⚡ {status?.move_count ?? 0} moves</span>
        {(status?.queue_size ?? 0) > 0 && (
          <span className="bp-chip bp-chip-yellow">Queue: {status?.queue_size}</span>
        )}
      </div>

      {/* Controls */}
      {/* ── Weaponized AI Agent panel ── */}
      <div className="bp-section" style={{ borderLeft: '3px solid #6366f1', paddingLeft: 12 }}>
        <div className="bp-section-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>🤖 AI Agent (port 8000)</span>
          <span className={`bp-conn ${aiOnline ? 'bp-conn-on' : 'bp-conn-off'}`}>
            {aiOnline ? '● Online' : '○ Offline'}
          </span>
        </div>
        <div className="bp-row" style={{ flexWrap: 'wrap', gap: 8 }}>
          <button className="btn btn-edit" disabled={!aiOnline}
            onClick={() => policyInfer(DEMO_OBS).then(setInferResult).catch(console.error)}>
            🧠 RL Infer
          </button>
          <button className="btn" style={{ background: '#1e293b', color: '#94a3b8' }}
            disabled={!aiOnline}
            onClick={() => rlTrain().then(setTrainResult).catch(console.error)}>
            ⚙ Train Step
          </button>
          <button className="btn" style={{ background: '#1e293b', color: '#94a3b8' }}
            disabled={!aiOnline}
            onClick={() => rlSave().catch(console.error)}>
            💾 Save Model
          </button>
          <select value={selectedMacro} onChange={e => setSelectedMacro(e.target.value)}
            className="legend-select" style={{ width: 130 }}>
            {macros.map(m => <option key={m}>{m}</option>)}
          </select>
          <button className="btn btn-run" disabled={!aiOnline}
            onClick={() => executeMacro(selectedMacro).catch(console.error)}>
            ▶ Execute Macro
          </button>
        </div>
        {inferResult && (
          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 6 }}>
            Action: <b style={{ color: '#4a9eff' }}>{inferResult.action}</b>
            &nbsp;| Value: <b>{inferResult.value.toFixed(3)}</b>
            &nbsp;| LogProb: <b>{inferResult.log_prob.toFixed(3)}</b>
          </div>
        )}
        {trainResult && (
          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
            Train → Policy loss: <b style={{ color: '#f59e0b' }}>{trainResult.loss_policy.toFixed(4)}</b>
            &nbsp;| Value loss: <b style={{ color: '#f59e0b' }}>{trainResult.loss_value.toFixed(4)}</b>
          </div>
        )}
      </div>

      {/* ── OBS Setup Panel ── */}
      <div className="bp-section" style={{ borderLeft: '3px solid #0ea5e9', paddingLeft: 12 }}>
        <div className="bp-section-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>📡 OBS Studio</span>
          <span className={`bp-conn ${obsStatus?.running ? 'bp-conn-on' : obsStatus?.installed ? 'bp-conn-off' : 'bp-conn-off'}`}>
            {obsStatus?.running ? '● Running' : obsStatus?.installed ? '○ Installed (not running)' : '○ Not Installed'}
          </span>
        </div>
        <div className="bp-row" style={{ flexWrap: 'wrap', gap: 8 }}>
          <button className="btn btn-edit" disabled={!aiOnline}
            onClick={() => getOBSStatus().then(setObsStatus).catch(console.error)}>
            🔄 Check Status
          </button>
          {obsStatus?.installed && !obsStatus?.running && (
            <button className="btn btn-run" disabled={!aiOnline} onClick={handleLaunchOBS}>
              ▶ Launch OBS
            </button>
          )}
          {!obsStatus?.installed && (
            <button className="btn" style={{ background: '#0ea5e9', color: '#fff' }}
              disabled={!aiOnline || obsStatus?.downloading || obsStatus?.installing}
              onClick={handleInstallOBS}>
              ⬇ Install OBS
            </button>
          )}
          <button className="btn" style={{ background: '#1e293b', color: '#94a3b8' }}
            disabled={!aiOnline} onClick={handleEnsureOBS}>
            ⚡ Auto-Setup OBS
          </button>
        </div>
        {obsStatus && (obsStatus.downloading || obsStatus.installing) && (
          <div style={{ marginTop: 6 }}>
            <div style={{ fontSize: 11, color: '#0ea5e9' }}>{obsStatus.install_message}</div>
            {obsStatus.downloading && (
              <div style={{ marginTop: 4, background: '#0f172a', borderRadius: 4, height: 8, width: '100%' }}>
                <div style={{ background: '#0ea5e9', height: '100%', borderRadius: 4,
                  width: `${obsStatus.download_pct}%`, transition: 'width 0.3s' }} />
              </div>
            )}
          </div>
        )}
        {obsStatus?.install_error && (
          <div style={{ fontSize: 11, color: '#ef4444', marginTop: 4 }}>⚠ {obsStatus.install_error}</div>
        )}
        {obsStatus?.install_done && !obsStatus?.install_error && (
          <div style={{ fontSize: 11, color: '#22c55e', marginTop: 4 }}>✓ {obsStatus.install_message}</div>
        )}
        {obsStatus?.running && (
          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
            {obsStatus.install_path && <span>Path: <span style={{ color: '#64748b' }}>{obsStatus.install_path}</span></span>}
            {obsStatus.camera_index !== null && <span> | Camera index: <b style={{ color: '#0ea5e9' }}>{obsStatus.camera_index}</b></span>}
          </div>
        )}
      </div>

      {/* ── OBS / Screen Capture Training Loop ── */}
      <div className="bp-section" style={{ borderLeft: '3px solid #10b981', paddingLeft: 12 }}>
        <div className="bp-section-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>📸 OBS Training Loop</span>
          <span className={`bp-conn ${loopStatus?.running ? 'bp-conn-on' : 'bp-conn-off'}`}>
            {loopStatus?.running ? `● Running (step ${loopStatus.step_count})` : '○ Idle'}
          </span>
        </div>
        <div className="bp-row" style={{ flexWrap: 'wrap', gap: 8, marginBottom: 6 }}>
          <label style={{ fontSize: 12, color: '#94a3b8' }}>Capture:</label>
          <select value={captureMode} onChange={e => setCaptureMode(e.target.value as 'mss' | 'obs')}
            className="legend-select" style={{ width: 110 }}>
            <option value="mss">mss (direct)</option>
            <option value="obs">OBS Camera</option>
          </select>
          <button className="btn" style={{ background: '#1e3a5f', color: '#60a5fa', border: '1px solid #3b82f6' }}
            title="Auto-detect Brawlhalla window and set capture region"
            onClick={handleCalibrate}>
            🎯 Calibrate
          </button>
          {calibrateResult && (
            <span style={{ fontSize: 11, color: calibrateResult.startsWith('✓') ? '#22c55e' : '#f59e0b' }}>
              {calibrateResult}
            </span>
          )}
          <button className="btn btn-run" disabled={!aiOnline || loopStatus?.running}
            onClick={handleStartLoop}>
            ▶ Start Loop
          </button>
          <button className="btn btn-delete" disabled={!loopStatus?.running}
            onClick={handleStopLoop}>
            ⏹ Stop Loop
          </button>
        </div>
        {loopStatus && (
          <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.8 }}>
            <span>Reward: <b style={{ color: loopStatus.last_reward >= 0 ? '#22c55e' : '#ef4444' }}>
              {loopStatus.last_reward.toFixed(2)}</b></span>
            &nbsp;| Action: <b style={{ color: '#a78bfa' }}>{loopStatus.last_action || '—'}</b>
            &nbsp;| P‑Loss: <b style={{ color: '#f59e0b' }}>{loopStatus.loss_policy.toFixed(4)}</b>
            &nbsp;| V‑Loss: <b style={{ color: '#f59e0b' }}>{loopStatus.loss_value.toFixed(4)}</b>
            &nbsp;| Entropy: <b style={{ color: '#818cf8' }}>{loopStatus.entropy?.toFixed(3) ?? '—'}</b>
            {loopStatus.last_ko_flash && <span style={{ color: '#ef4444', marginLeft: 8 }}>💥 KO FLASH</span>}
            <br />
            P1&nbsp;
            <b style={{ color: loopStatus.last_p1_damage_tier === 'red' ? '#ef4444' :
              loopStatus.last_p1_damage_tier === 'orange' ? '#f97316' :
              loopStatus.last_p1_damage_tier === 'yellow' ? '#eab308' : '#e2e8f0' }}>
              {loopStatus.last_p1_damage}%
            </b>
            &nbsp;({loopStatus.last_p1_stocks}🔵)
            {loopStatus.last_p1_weapon && loopStatus.last_p1_weapon !== 'none' &&
              <span style={{ color: '#0ea5e9' }}>&nbsp;⚔ {loopStatus.last_p1_weapon}</span>}
            &nbsp;vs P2&nbsp;
            <b style={{ color: loopStatus.last_p2_damage_tier === 'red' ? '#ef4444' :
              loopStatus.last_p2_damage_tier === 'orange' ? '#f97316' :
              loopStatus.last_p2_damage_tier === 'yellow' ? '#eab308' : '#e2e8f0' }}>
              {loopStatus.last_p2_damage}%
            </b>
            &nbsp;({loopStatus.last_p2_stocks}🔴)
            {loopStatus.last_p2_weapon && loopStatus.last_p2_weapon !== 'none' &&
              <span style={{ color: '#f97316' }}>&nbsp;⚔ {loopStatus.last_p2_weapon}</span>}
            {(loopStatus.last_stage_pickups ?? 0) > 0 &&
              <span style={{ color: '#22c55e', marginLeft: 8 }}>🟢 {loopStatus.last_stage_pickups} pickup(s) on stage</span>}
            {loopStatus.errors.length > 0 && (
              <div style={{ color: '#ef4444', marginTop: 4 }}>
                ⚠ {loopStatus.errors[loopStatus.errors.length - 1]}
              </div>
            )}
          </div>
        )}
        {/* ── RL Stats mini-panel ─────────────────────────────────────────── */}
        {agentStats !== null && (
          <div style={{
            marginTop: 10, padding: '8px 10px', background: '#0f172a',
            borderRadius: 6, border: '1px solid #1e3a5f', fontSize: 11,
            display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 4, textAlign: 'center',
          }}>
            {[
              ['Updates', agentStats.total_updates],
              ['Episodes', agentStats.episode_count],
              ['Mean Rew.', agentStats.mean_ep_reward?.toFixed(1) ?? '—'],
              ['Best Rew.', agentStats.best_ep_reward?.toFixed(1) ?? '—'],
              ['Steps', agentStats.total_steps ?? '—'],
            ].map(([label, val]) => (
              <div key={label as string}>
                <div style={{ color: '#475569', marginBottom: 2 }}>{label}</div>
                <div style={{ color: '#a5b4fc', fontWeight: 700 }}>{val}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── YouTube Learning Panel ───────────────────────────────────────────── */}
      <div className="bp-section" style={{ borderLeft: '3px solid #a855f7', paddingLeft: 12 }}>
        <div className="bp-section-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>🎥 YouTube Learning</span>
          {videoStatus && (
            <span style={{
              fontSize: 11, padding: '2px 8px', borderRadius: 10,
              background: videoStatus.state === 'error' ? '#7f1d1d' :
                          videoStatus.state === 'done'  ? '#14532d' : '#312e81',
              color: videoStatus.state === 'error' ? '#fca5a5' :
                     videoStatus.state === 'done'  ? '#86efac' : '#c4b5fd',
            }}>
              {videoStatus.state.toUpperCase()}
            </span>
          )}
        </div>

        {/* URL input row */}
        <div className="bp-row" style={{ flexWrap: 'wrap', gap: 8, marginBottom: 6 }}>
          <input
            type="text"
            value={videoUrl}
            onChange={e => setVideoUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=…"
            style={{
              flex: 1, minWidth: 200, padding: '4px 8px', background: '#0f172a',
              border: '1px solid #334155', borderRadius: 4, color: '#e2e8f0', fontSize: 12,
            }}
            onKeyDown={e => { if (e.key === 'Enter') handleIngestVideo() }}
          />
          <label style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: '#94a3b8', cursor: 'pointer' }}
            title="Use Whisper speech recognition to extract Brawlhalla terminology from commentary">
            <input type="checkbox" checked={videoTranscribe} onChange={e => setVideoTranscribe(e.target.checked)} />
            🎙 Whisper
          </label>
          <button
            className="btn"
            style={{ background: '#581c87', color: '#e9d5ff', border: '1px solid #7c3aed' }}
            disabled={!aiOnline || !videoUrl.trim() || (videoStatus?.state === 'downloading' || videoStatus?.state === 'transcribing' || videoStatus?.state === 'extracting')}
            onClick={handleIngestVideo}
          >
            ⬇ Ingest
          </button>
        </div>

        {/* Progress bar */}
        {videoStatus && videoStatus.state !== 'idle' && (
          <div style={{ marginBottom: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#94a3b8', marginBottom: 3 }}>
              <span>
                {videoStatus.state === 'downloading'  ? `Downloading${videoStatus.video_title ? ` "${videoStatus.video_title}"` : '…'}` :
                 videoStatus.state === 'transcribing' ? `🎙 Transcribing audio with Whisper…` :
                 videoStatus.state === 'extracting'   ? `Extracting frames (${videoStatus.frames_done}/${videoStatus.frames_total})` :
                 videoStatus.state === 'pretraining'  ? `Behavioural Cloning epoch ${videoStatus.bc_epochs_done}` :
                 videoStatus.state === 'done'         ? `✓ Done — ${videoStatus.transitions_extracted} transitions from "${videoStatus.video_title}"` :
                 videoStatus.state === 'error'        ? `⚠ ${videoStatus.error}` : ''}
              </span>
              <span>{Math.round(videoStatus.progress * 100)}%</span>
            </div>
            <div style={{ height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: 3,
                width: `${videoStatus.progress * 100}%`,
                background: videoStatus.state === 'error' ? '#ef4444' :
                            videoStatus.state === 'pretraining' ? '#a855f7' : '#7c3aed',
                transition: 'width 0.4s ease',
              }} />
            </div>
          </div>
        )}

        {videoStatus && videoStatus.corpus_size > 0 && (
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 11, color: '#94a3b8', marginBottom: 8 }}>
            <span>Corpus: <b style={{ color: '#c084fc' }}>{videoStatus.corpus_size.toLocaleString()}</b> transitions</span>
            {videoStatus.transcribed && (
              <span>🎙 <b style={{ color: '#4ade80' }}>{videoStatus.knowledge_terms}</b> terms extracted</span>
            )}
            {videoStatus.bc_epochs_done > 0 && <>
              <span>BC Epochs: <b style={{ color: '#818cf8' }}>{videoStatus.bc_epochs_done}</b></span>
              <span>Loss: <b style={{ color: '#fb923c' }}>{videoStatus.bc_loss_initial.toFixed(4)}</b>
                &nbsp;→&nbsp;<b style={{ color: '#4ade80' }}>{videoStatus.bc_loss_last.toFixed(4)}</b></span>
            </>}
          </div>
        )}

        {/* Knowledge tag cloud — shown once terms are extracted */}
        {videoKnowledge.length > 0 && (
          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 10, color: '#475569', marginBottom: 4 }}>📚 Brawlhalla knowledge extracted from commentary:</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
              {videoKnowledge.slice(0, 30).map(({ term, count }) => {
                const maxCount = videoKnowledge[0]?.count || 1
                const intensity = Math.max(0.3, count / maxCount)
                return (
                  <span key={term} style={{
                    padding: '2px 8px', borderRadius: 10, fontSize: 10,
                    background: `rgba(139,92,246,${intensity * 0.4})`,
                    border: `1px solid rgba(139,92,246,${intensity * 0.8})`,
                    color: `rgba(233,213,255,${0.5 + intensity * 0.5})`,
                    fontWeight: count === maxCount ? 700 : 400,
                  }}>
                    {term} <span style={{ opacity: 0.6 }}>×{count}</span>
                  </span>
                )
              })}
            </div>
          </div>
        )}

        {/* Action buttons */}
        <div className="bp-row" style={{ gap: 8, flexWrap: 'wrap' }}>
          <button
            className="btn"
            style={{ background: '#3b0764', color: '#d8b4fe', border: '1px solid #7c3aed', fontSize: 11 }}
            disabled={!aiOnline || (videoStatus?.corpus_size ?? 0) < 128 || videoStatus?.state === 'pretraining'}
            onClick={handlePretrain}
            title="Run Behavioural Cloning on extracted video transitions to pre-train the policy"
          >
            🧠 Pre-train from corpus
          </button>
          <button
            className="btn"
            style={{ background: '#1e293b', color: '#64748b', border: '1px solid #334155', fontSize: 11 }}
            disabled={!videoStatus || videoStatus.state === 'downloading' || videoStatus.state === 'extracting'}
            onClick={handleVideoReset}
          >
            🗑 Clear corpus
          </button>
          <button
            className="btn"
            style={{ background: '#1e293b', color: '#64748b', border: '1px solid #334155', fontSize: 11 }}
            onClick={() => setVideoLogOpen(o => !o)}
          >
            {videoLogOpen ? '▲ Hide log' : '▼ Log'}
          </button>
        </div>

        {/* Collapsible log */}
        {videoLogOpen && videoStatus && videoStatus.log.length > 0 && (
          <div style={{
            marginTop: 8, maxHeight: 140, overflowY: 'auto', padding: '6px 8px',
            background: '#020617', borderRadius: 4, border: '1px solid #1e293b',
            fontFamily: 'monospace', fontSize: 10, color: '#64748b', lineHeight: 1.6,
          }}>
            {videoStatus.log.map((line, i) => (
              <div key={i} style={{ color: line.includes('ERROR') ? '#f87171' : line.includes('Done') ? '#4ade80' : '#64748b' }}>
                {line}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Brain Storage Panel ──────────────────────────────────────────── */}
      <div className="bp-section" style={{ borderColor: '#1d4ed8' }}>
        <div className="bp-section-title" style={{ color: '#93c5fd' }}>
          🧠 Brain Storage
          <button
            style={{ marginLeft: 'auto', background: 'none', border: 'none', color: '#475569',
                     fontSize: 10, cursor: 'pointer', padding: '0 4px' }}
            onClick={() => getBrainInfo().then(setBrainInfo).catch(() => {})}
            title="Refresh brain stats"
          >↻</button>
        </div>

        {/* Stats row — lazy load on first expand */}
        {!brainInfo && (
          <div style={{ fontSize: 11, color: '#475569' }}>
            <button className="btn"
              style={{ background: '#1e3a5f', color: '#93c5fd', border: '1px solid #1d4ed8', fontSize: 11 }}
              onClick={() => getBrainInfo().then(setBrainInfo).catch(() => {})}
            >📦 Load brain stats</button>
          </div>
        )}

        {brainInfo && (
          <>
            {/* Metric tiles */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6, marginBottom: 8 }}>
              {([
                ['Terms', brainInfo.knowledge_terms, '#818cf8'],
                ['Corpus', brainInfo.corpus_size.toLocaleString(), '#c084fc'],
                ['Videos', brainInfo.videos_ingested, '#34d399'],
                ['Model', brainInfo.model_exists ? `${brainInfo.model_mb} MB` : 'none', brainInfo.model_exists ? '#fbbf24' : '#475569'],
              ] as [string, string|number, string][]).map(([label, val, col]) => (
                <div key={label} style={{ background: '#0f172a', borderRadius: 6, padding: '6px 8px',
                  border: '1px solid #1e293b', textAlign: 'center' }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: col }}>{val}</div>
                  <div style={{ fontSize: 9, color: '#475569', marginTop: 1 }}>{label}</div>
                </div>
              ))}
            </div>

            {/* Corpus file size */}
            {brainInfo.corpus_bytes > 0 && (
              <div style={{ fontSize: 10, color: '#475569', marginBottom: 6 }}>
                💾 corpus.npz: <b style={{ color: '#94a3b8' }}>{brainInfo.corpus_mb} MB</b>
                &ensp;|  model.pt: <b style={{ color: '#94a3b8' }}>{brainInfo.model_mb} MB</b>
              </div>
            )}

            {/* Top terms preview */}
            {brainInfo.top_terms.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 10, color: '#475569', marginBottom: 4 }}>Top persistent terms:</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {brainInfo.top_terms.slice(0, 15).map(([term, count]) => (
                    <span key={term} style={{
                      padding: '2px 7px', borderRadius: 10, fontSize: 10,
                      background: '#1e3a5f', color: '#93c5fd',
                      border: '1px solid #1d4ed8',
                    }}>{term} <span style={{ opacity: 0.5 }}>×{count}</span></span>
                  ))}
                </div>
              </div>
            )}

            {/* Video registry toggle */}
            {brainInfo.videos_ingested > 0 && (
              <div style={{ marginBottom: 8 }}>
                <button style={{ background: 'none', border: 'none', color: '#64748b',
                  fontSize: 10, cursor: 'pointer', padding: 0 }}
                  onClick={() => setShowRegistry(r => !r)}
                >{showRegistry ? '▼' : '►'} {brainInfo.videos_ingested} ingested video{brainInfo.videos_ingested !== 1 ? 's' : ''}</button>
                {showRegistry && (
                  <div style={{ marginTop: 4, display: 'flex', flexDirection: 'column', gap: 3 }}>
                    {brainInfo.registry.map(v => (
                      <div key={v.url} style={{ fontSize: 9, color: '#475569', paddingLeft: 10,
                        borderLeft: '2px solid #1e3a5f' }}>
                        <span style={{ color: '#94a3b8' }}>{v.title || v.url}</span>
                        &ensp;•&ensp;{v.transitions.toLocaleString()} transitions
                        &ensp;•&ensp;ingested {v.times_ingested}×
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Feedback message */}
            {brainMsg && (
              <div style={{ fontSize: 10, color: '#4ade80', marginBottom: 6 }}>{brainMsg}</div>
            )}

            {/* Action buttons */}
            <div className="bp-row" style={{ gap: 6, flexWrap: 'wrap' }}>
              <button className="btn"
                style={{ background: '#1e3a5f', color: '#93c5fd', border: '1px solid #1d4ed8', fontSize: 11 }}
                disabled={!aiOnline || brainSaving}
                onClick={async () => {
                  setBrainSaving(true)
                  try {
                    const r = await saveBrain()
                    setBrainMsg(`✓ Saved — ${r.corpus_written.toLocaleString()} transitions, ${r.knowledge_terms} terms`)
                    getBrainInfo().then(setBrainInfo).catch(() => {})
                  } catch (e) { setBrainMsg(`⚠ Save failed`) }
                  finally { setBrainSaving(false) }
                }}
              >{brainSaving ? '⏳ Saving…' : '💾 Save brain'}</button>

              <button className="btn"
                style={{ background: '#1e293b', color: '#64748b', border: '1px solid #334155', fontSize: 11 }}
                disabled={!aiOnline}
                onClick={() => reloadBrain()
                  .then(r => { setBrainMsg(`↻ Reloaded — ${r.knowledge_terms} terms, ${r.corpus_size.toLocaleString()} transitions`)
                               getBrainInfo().then(setBrainInfo).catch(() => {}) })
                  .catch(() => setBrainMsg('⚠ Reload failed'))}
              >↻ Reload from disk</button>

              <button className="btn"
                style={{ background: '#450a0a', color: '#fca5a5', border: '1px solid #7f1d1d', fontSize: 11 }}
                disabled={!aiOnline}
                onClick={() => {
                  if (!confirm('Permanently wipe all brain data (knowledge, corpus, registry)?\nThe model.pt is NOT deleted.')) return
                  clearBrain()
                    .then(() => { setBrainMsg('🗑 Brain cleared')
                                  setBrainInfo(b => b ? {...b, knowledge_terms:0, corpus_size:0, videos_ingested:0, registry:[], top_terms:[]} : b) })
                    .catch(() => setBrainMsg('⚠ Clear failed'))
                }}
              >🗑 Clear brain</button>
            </div>
          </>
        )}
      </div>

      <div className="bp-section">
        <div className="bp-section-title">Controls</div>
        <div className="bp-row">
          <button className="btn btn-edit"
            onClick={() => post('/api/fight/activate')}
            disabled={status?.status === 'fighting'}>
            🔋 Arm
          </button>
          <button className="btn btn-run"
            onClick={() => post('/api/fight/start', { binds, tick_ms: tickMs })}
            disabled={status?.status === 'fighting'}>
            ▶ Start Loop
          </button>
          <button className="btn btn-delete"
            onClick={() => post('/api/fight/stop')}
            disabled={status?.status !== 'fighting'}>
            ⏹ Stop
          </button>
        </div>
      </div>

      {/* Tick rate */}
      <div className="bp-section">
        <div className="bp-section-title">Tick Rate — {tickMs} ms per strategy eval</div>
        <div className="bp-row" style={{ alignItems: 'center', gap: 12 }}>
          <input type="range" min={50} max={500} step={10} value={tickMs}
            onChange={e => setTickMs(+e.target.value)}
            style={{ width: 220, accentColor: '#4a9eff' }} />
          <span style={{ color: '#94a3b8', fontSize: 12 }}>
            {tickMs < 100 ? '🔥 Aggressive' : tickMs < 250 ? '⚡ Normal' : '🛡 Conservative'}
          </span>
        </div>
      </div>

      {/* Single move fire + Hit-Confirm */}
      <div className="bp-section">
        <div className="bp-section-title">Fire Single Move</div>
        <div className="bp-row">
          <select value={selectedMove} onChange={e => setSelectedMove(e.target.value)}
            className="legend-select" style={{ width: 160 }}>
            {MOVE_LIST.map(m => <option key={m}>{m}</option>)}
          </select>
          <button className="btn btn-edit" onClick={armRefAndFire}>
            Fire + Confirm ⚡
          </button>
          <button className="btn" style={{ background: '#334155', color: '#94a3b8' }}
            onClick={() => post('/api/fight/exec-move', { move: selectedMove })}>
            Fire Only
          </button>
        </div>

        {/* Hit-confirm result badge */}
        {hitResult && (
          <div className="bp-row" style={{ marginTop: 8, gap: 10 }}>
            <span className={`bp-chip ${hitResult.result === 'CONFIRMED' ? 'bp-chip-green' : 'bp-chip-red'}`}
              style={{ fontSize: 13, padding: '3px 12px' }}>
              {hitResult.result === 'CONFIRMED' ? '💥 HIT CONFIRMED' : '❌ WHIFF'}
            </span>
            <span className="bp-chip">{hitResult.accuracy}% accuracy</span>
            <button style={{ background: 'none', border: 'none', color: '#475569',
              cursor: 'pointer', fontSize: 11 }} onClick={() => setHitResult(null)}>
              ✕ clear
            </button>
          </div>
        )}
      </div>

      {/* Meta Intelligence panel */}
      <div className="bp-section">
        <div className="bp-section-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>📡 Live Meta Intelligence</span>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            {meta.tournament.strategy !== 'STANDARD' && (
              <span className="bp-chip bp-chip-yellow" style={{ fontSize: 11 }}>
                🏆 {meta.tournament.match_type}
              </span>
            )}
            <span className="bp-chip" style={{ fontSize: 10 }}>
              ×{meta.tournament.multiplier.toFixed(1)} {meta.tournament.strategy}
            </span>
            {meta.viewer_count > 0 && (
              <span className="bp-chip bp-chip-blue" style={{ fontSize: 10 }}>
                {meta.viewer_count.toLocaleString()} viewers
              </span>
            )}
            <button className="btn btn-edit" style={{ padding: '2px 8px', fontSize: 11 }}
              onClick={() => fetch(`${API}/api/meta/refresh`, { method: 'POST' })}>
              ↻ Refresh
            </button>
          </div>
        </div>

        {/* Weapon trend grid */}
        {Object.keys(meta.weapon_trends).length > 0 ? (
          <div className="meta-weapon-grid">
            {Object.entries(meta.weapon_trends)
              .sort((a, b) => b[1].trend - a[1].trend)
              .map(([weapon, data]) => {
                const pct = Math.round(data.trend * 100)
                const isPos = pct > 0
                const isNeg = pct < 0
                return (
                  <div key={weapon} className="meta-weapon-row">
                    <span className="meta-weapon-name">{weapon}</span>
                    <div className="meta-bar-track">
                      <div
                        className="meta-bar-fill"
                        style={{
                          width: `${Math.min(100, Math.abs(pct) * 10)}%`,
                          background: isPos ? '#22c55e' : isNeg ? '#ef4444' : '#334155',
                        }}
                      />
                    </div>
                    <span className={`meta-trend-val ${isPos ? 'pos' : isNeg ? 'neg' : ''}`}>
                      {pct > 0 ? `+${pct}` : pct}%
                    </span>
                    <span className="meta-priority">{data.priority.toFixed(2)}</span>
                  </div>
                )
              })}
          </div>
        ) : (
          <p style={{ color: '#475569', fontSize: 12, marginTop: 8 }}>
            No meta data yet — configure BH API key via /api/meta/config and click Refresh.
          </p>
        )}
        {meta.last_updated && (
          <p style={{ color: '#334155', fontSize: 10, marginTop: 6 }}>
            Last updated: {meta.last_updated}
          </p>
        )}
      </div>

      {/* ── Replay Engine Panel ──────────────────────────────────────────── */}
      <div className="bp-section" style={{ borderColor: '#15803d' }}>
        <div className="bp-section-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>🎬 Replay Digestion Engine</span>
          <div style={{ display: 'flex', gap: 6 }}>
            {replayScanData && (
              <span className="bp-chip bp-chip-green" style={{ fontSize: 11 }}>
                {replayScanData.count} files
              </span>
            )}
          </div>
        </div>
        <p style={{ color: '#475569', fontSize: 11, margin: '0 0 8px' }}>
          Parses <code style={{ color: '#86efac' }}>.replay</code> files from your Brawlhalla AppData folder
          and converts them into BC training transitions for the brain.
        </p>

        {/* Scan row */}
        <div className="bp-row" style={{ gap: 8, marginBottom: 8 }}>
          <button className="btn btn-edit"
            disabled={replayScanning}
            onClick={async () => {
              setReplayScanning(true); setReplayMsg(null)
              try {
                const d = await replayScan()
                setReplayScanData(d)
                setReplayMsg(`Found ${d.count} replay files in ${d.replay_dir}`)
                if (d.replays.length > 0) setSelectedReplay(d.replays[0].path)
              } catch { setReplayMsg('⚠ Scan failed — is the server running?') }
              finally { setReplayScanning(false) }
            }}>
            {replayScanning ? '⏳ Scanning…' : '🔍 Scan Replays'}
          </button>
          <button className="btn btn-run"
            disabled={replayIngestingAll || !replayScanData}
            onClick={async () => {
              setReplayIngestingAll(true); setReplayMsg(null)
              try {
                const r = await replayIngestAll()
                setReplayMsg(`✓ Ingested ${r.files_ingested}/${r.files_found} files → ${r.total_transitions.toLocaleString()} transitions`)
              } catch { setReplayMsg('⚠ Ingest all failed') }
              finally { setReplayIngestingAll(false) }
            }}>
            {replayIngestingAll ? '⏳ Ingesting…' : '⚡ Ingest All New'}
          </button>
        </div>

        {/* File list */}
        {replayScanData && replayScanData.replays.length > 0 && (
          <>
            <select
              style={{ width: '100%', background: '#0f172a', color: '#94a3b8',
                border: '1px solid #1e293b', borderRadius: 4, padding: '4px 8px',
                fontSize: 11, marginBottom: 8 }}
              value={selectedReplay ?? ''}
              onChange={e => setSelectedReplay(e.target.value)}>
              {replayScanData.replays.map(r => (
                <option key={r.path} value={r.path}>
                  {r.stage || r.name} — v{r.game_version} ({r.size_kb}KB
                  {r.parse_ok ? `, ~${Math.round(r.frame_count/60)}s` : ', raw'})
                </option>
              ))}
            </select>
            <button className="btn btn-edit"
              disabled={replayIngesting || !selectedReplay}
              style={{ marginBottom: 8 }}
              onClick={async () => {
                if (!selectedReplay) return
                setReplayIngesting(true); setReplayMsg(null)
                try {
                  const r = await replayIngest(selectedReplay)
                  setReplayMsg(`✓ ${r.title} → ${r.transitions.toLocaleString()} transitions`)
                } catch (e: any) { setReplayMsg(`⚠ ${e.message}`) }
                finally { setReplayIngesting(false) }
              }}>
              {replayIngesting ? '⏳ Parsing…' : '▶ Ingest Selected'}
            </button>
          </>
        )}

        {replayMsg && (
          <div style={{ fontSize: 11, color: '#4ade80', marginTop: 4 }}>{replayMsg}</div>
        )}
      </div>

      {/* ── Live Memory Reader Panel ──────────────────────────────────────── */}
      <div className="bp-section" style={{ borderColor: '#b45309' }}>
        <div className="bp-section-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>🧬 Live Memory Reader</span>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span className={`bp-chip ${memInfo?.attached ? 'bp-chip-green' : 'bp-chip-red'}`}
              style={{ fontSize: 11 }}>
              {memInfo?.attached ? `PID ${memInfo.pid}` : 'Not attached'}
            </span>
            <span className="bp-chip bp-chip-blue" style={{ fontSize: 10 }}>
              {memInfo?.addresses_found.length ?? 0} addrs
            </span>
          </div>
        </div>
        <p style={{ color: '#475569', fontSize: 11, margin: '0 0 8px' }}>
          Reads damage %, stocks, and positions directly from <code style={{ color: '#fcd34d' }}>Brawlhalla.exe</code> memory
          — faster and more accurate than OCR.
        </p>

        {/* Control row */}
        <div className="bp-row" style={{ gap: 8, marginBottom: 8 }}>
          <button className="btn btn-edit"
            onClick={() => getMemoryInfo().then(setMemInfo).catch(() => setMemInfo(null))}>
            🔗 Attach
          </button>
          <button className="btn"
            style={{ background: memPolling ? '#7f1d1d' : '#1e3a5f', color: '#94a3b8' }}
            onClick={() => {
              if (memPolling) {
                if (memPollRef.current) clearInterval(memPollRef.current)
                setMemPolling(false)
              } else {
                setMemPolling(true)
                memPollRef.current = setInterval(async () => {
                  try {
                    const s = await getMemoryState()
                    setMemState({ p1: s.p1, p2: s.p2, source: s.source })
                  } catch { setMemPolling(false); if (memPollRef.current) clearInterval(memPollRef.current) }
                }, 100)
              }
            }}>
            {memPolling ? '⏹ Stop Poll' : '▶ Poll @10Hz'}
          </button>
          <button className="btn" style={{ background: '#1a1a2e', color: '#94a3b8' }}
            onClick={() => rescanMemory().then(setMemInfo).catch(() => {})}>
            ↺ Rescan Addrs
          </button>
        </div>

        {/* Live state display */}
        {memState && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 6 }}>
            {(['p1', 'p2'] as const).map(p => {
              const player = memState[p]
              const dmgPct = Math.min(100, (player.damage / 300) * 100)
              return (
                <div key={p} style={{ background: '#0f172a', borderRadius: 6, padding: '6px 10px',
                  border: `1px solid ${p === 'p1' ? '#1d4ed8' : '#991b1b'}` }}>
                  <div style={{ fontSize: 10, color: '#64748b', marginBottom: 3 }}>
                    {p.toUpperCase()} <span style={{ color: '#475569' }}>({memState.source})</span>
                  </div>
                  <div style={{ fontSize: 18, fontWeight: 700,
                    color: player.damage > 150 ? '#ef4444' : player.damage > 80 ? '#f59e0b' : '#4ade80' }}>
                    {player.damage.toFixed(0)}%
                  </div>
                  <div style={{ height: 3, background: '#1e293b', borderRadius: 2, marginTop: 3, marginBottom: 3 }}>
                    <div style={{ height: '100%', width: `${dmgPct}%`, borderRadius: 2,
                      background: player.damage > 150 ? '#ef4444' : player.damage > 80 ? '#f59e0b' : '#22c55e' }} />
                  </div>
                  <div style={{ display: 'flex', gap: 4 }}>
                    {Array.from({ length: player.stocks }).map((_, i) => (
                      <span key={i} style={{ fontSize: 12 }}>💙</span>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {memInfo && memInfo.addresses_found.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
            {memInfo.addresses_found.map(a => (
              <span key={a} className="bp-chip" style={{ fontSize: 10 }}>{a}</span>
            ))}
          </div>
        )}
      </div>

      {/* ── BTR Agent Panel ────────────────────────────────────────────────── */}
      <div className="bp-section" style={{ borderColor: '#6d28d9' }}>
        <div className="bp-section-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>🌈 BTR Agent <span style={{ fontSize: 11, color: '#7c3aed' }}>Beyond The Rainbow</span></span>
          <button className="btn btn-edit" style={{ padding: '2px 8px', fontSize: 11 }}
            onClick={() => getBTRStats().then(setBtrStats).catch(() => {})}>
            ↻ Refresh
          </button>
        </div>
        <p style={{ color: '#475569', fontSize: 11, margin: '0 0 8px' }}>
          Noisy Dueling DQN with Prioritized Replay + n-step returns (ported from Wii-RL/ICML 2025).
          Pre-train on replay corpus then run alongside PPO.
        </p>

        {btrStats ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6, marginBottom: 8 }}>
            {[
              { label: 'Steps',   val: btrStats.total_steps.toLocaleString() },
              { label: 'Updates', val: btrStats.total_updates.toLocaleString() },
              { label: 'Replay',  val: btrStats.replay_size.toLocaleString() },
              { label: 'Episodes',val: btrStats.episode_count },
              { label: 'Mean Rew',val: btrStats.mean_ep_reward.toFixed(3) },
              { label: 'Best Rew',val: btrStats.best_ep_reward.toFixed(3) },
            ].map(({ label, val }) => (
              <div key={label} style={{ background: '#0f172a', borderRadius: 5, padding: '4px 8px',
                textAlign: 'center', border: '1px solid #2e1065' }}>
                <div style={{ fontSize: 9, color: '#6d28d9', textTransform: 'uppercase' }}>{label}</div>
                <div style={{ fontSize: 14, color: '#c4b5fd', fontWeight: 600 }}>{val}</div>
              </div>
            ))}
          </div>
        ) : (
          <button className="btn btn-edit" style={{ marginBottom: 8 }}
            onClick={() => getBTRStats().then(setBtrStats).catch(() => setBtrStats(null))}>
            Load BTR Stats
          </button>
        )}

        <div className="bp-row" style={{ gap: 8 }}>
          <button className="btn btn-run"
            disabled={btrPretraining}
            onClick={async () => {
              setBtrPretraining(true); setBtrMsg(null)
              try {
                const r = await btrPretrain()
                setBtrMsg(`✓ BC pre-train done — loss ${r.bc_loss.toFixed(4)}, ${r.samples.toLocaleString()} samples`)
                getBTRStats().then(setBtrStats).catch(() => {})
              } catch (e: any) { setBtrMsg(`⚠ ${e.message}`) }
              finally { setBtrPretraining(false) }
            }}>
            {btrPretraining ? '⏳ Training…' : '🎓 Pre-train on Corpus'}
          </button>
          <button className="btn btn-edit"
            onClick={() => saveBTR().then(() => setBtrMsg('💾 BTR model saved')).catch(() => setBtrMsg('⚠ Save failed'))}>
            💾 Save BTR
          </button>
        </div>
        {btrMsg && <div style={{ fontSize: 11, color: '#c4b5fd', marginTop: 6 }}>{btrMsg}</div>}
      </div>

      {/* Keybinds */}
      <div className="bp-section">
        <div className="bp-section-title">Keybinds (must match in-game settings)</div>
        <div className="bp-binds-grid">
          {Object.entries(binds).map(([k, v]) => (
            <label key={k} className="bp-bind-row">
              <span className="bp-bind-key">{k}</span>
              <input
                value={v}
                onChange={e => setBinds(b => ({ ...b, [k]: e.target.value.toUpperCase() }))}
                className="bp-bind-input"
                maxLength={8}
              />
            </label>
          ))}
        </div>
      </div>

      {/* Live log */}
      <div className="bp-section" style={{ flexGrow: 1 }}>
        <div className="bp-section-title">📋 Live Backend Log (SSE)</div>
        <div className="log-body bp-log">
          {log.length === 0
            ? <span className="log-empty">{connected ? 'Waiting for events…' : 'Backend offline — start capture_server.py'}</span>
            : log.map((l, i) => <div key={i} className="log-line">{l}</div>)
          }
        </div>
      </div>
    </div>
  )
}
