/**
 * BackendBridge — sidebar widget that manages the Flask capture server.
 * Polls server_health() every 2 s via Tauri invoke.
 * Can launch / stop capture_server.py directly from the Tauri app.
 */
import { useState, useEffect, useCallback } from 'react'
import { invoke } from '@tauri-apps/api/core'

const DEFAULT_SERVER_DIR =
  'C:\\Users\\carli\\OneDrive\\Documents\\My Mods\\sdlarch-rl-master'

type Health = 'online' | 'offline' | 'starting' | 'stopping'

export default function BackendBridge() {
  const [health, setHealth]     = useState<Health>('offline')
  const [serverDir, setServerDir] = useState<string>(
    () => localStorage.getItem('bh_server_dir') ?? DEFAULT_SERVER_DIR
  )
  const [editingPath, setEditingPath] = useState(false)
  const [draftPath, setDraftPath]     = useState(serverDir)
  const [log, setLog]                 = useState('')

  const checkHealth = useCallback(async () => {
    try {
      const ok = await invoke<boolean>('server_health')
      setHealth(h => (h === 'starting' || h === 'stopping') ? h : ok ? 'online' : 'offline')
    } catch {
      setHealth('offline')
    }
  }, [])


  // Auto-launch backend on mount
  useEffect(() => {
    launch();
    checkHealth();
    const id = setInterval(checkHealth, 2000);
    return () => clearInterval(id);
  }, [checkHealth]);

  // When status transitions starting→online or stopping→offline, settle
  useEffect(() => {
    if (health === 'starting') {
      const t = setTimeout(async () => {
        const ok = await invoke<boolean>('server_health').catch(() => false)
        setHealth(ok ? 'online' : 'offline')
        setLog(ok ? 'Server is online ✓' : 'Server did not respond — check Python path')
        if (ok) {
          // Auto-push stored API credentials to Flask so meta monitoring starts
          const cfg = {
            bh_api_key:           localStorage.getItem('bh_api_key')           ?? '',
            twitch_client_id:     localStorage.getItem('twitch_client_id')     ?? '',
            twitch_client_secret: localStorage.getItem('twitch_client_secret') ?? '',
          }
          const hasAny = Object.values(cfg).some(v => v.length > 0)
          if (hasAny) {
            fetch('http://localhost:5000/api/meta/config', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(cfg),
            }).catch(() => {})
          }
        }
      }, 3500)
      return () => clearTimeout(t)
    }
    if (health === 'stopping') {
      const t = setTimeout(() => setHealth('offline'), 2000)
      return () => clearTimeout(t)
    }
  }, [health])

  async function launch() {
    setHealth('starting')
    setLog('Launching capture_server.py…')
    try {
      const result = await invoke<string>('start_capture_server', { serverDir })
      setLog(result === 'already_running' ? 'Already running (external)' : 'Server process spawned')
      if (result === 'already_running') setHealth('online')
    } catch (e) {
      setLog(`Error: ${String(e)}`)
      setHealth('offline')
    }
  }

  async function stop() {
    setHealth('stopping')
    setLog('Stopping server…')
    try {
      const result = await invoke<string>('stop_capture_server')
      setLog(result === 'not_managed' ? 'Not managed by app (use Task Manager to stop external process)' : 'Server stopped')
    } catch (e) {
      setLog(`Error: ${String(e)}`)
    }
  }

  function savePath() {
    setServerDir(draftPath)
    localStorage.setItem('bh_server_dir', draftPath)
    setEditingPath(false)
  }

  const DOT_COLOR: Record<Health, string> = {
    online:   '#22c55e',
    offline:  '#ef4444',
    starting: '#f59e0b',
    stopping: '#f59e0b',
  }
  const LABEL: Record<Health, string> = {
    online:   'Online',
    offline:  'Offline',
    starting: 'Starting…',
    stopping: 'Stopping…',
  }

  // Show a big loading/error UI until backend is ready
  if (health === 'starting' || health === 'offline') {
    return (
      <div className="bridge-widget bridge-widget-loading">
        <div style={{ textAlign: 'center', margin: '2em 0' }}>
          <div style={{ fontSize: 32, marginBottom: 16 }}>⏳</div>
          <div style={{ fontSize: 20, fontWeight: 600 }}>
            {health === 'starting' ? 'Starting backend…' : 'Waiting for backend…'}
          </div>
          {log && <div style={{ color: '#f87171', marginTop: 12 }}>{log}</div>}
        </div>
      </div>
    );
  }

  // Normal UI when backend is online
  return (
    <div className="bridge-widget">
      {/* Status row */}
      <div className="bridge-status-row">
        <span className="bridge-dot" style={{ background: DOT_COLOR[health] }} />
        <span className="bridge-label">{LABEL[health]}</span>
        <span className="bridge-port">:5000</span>
        <div className="bridge-actions">
          {(health === 'online') && (
            <button className="bridge-btn bridge-btn-stop" onClick={stop}>Stop</button>
          )}
          <button
            className="bridge-btn bridge-btn-path"
            title="Configure server path"
            onClick={() => { setDraftPath(serverDir); setEditingPath(e => !e) }}>
            ⚙
          </button>
        </div>
      </div>

      {/* Path editor */}
      {editingPath && (
        <div className="bridge-path-editor">
          <input
            className="bridge-path-input"
            value={draftPath}
            onChange={e => setDraftPath(e.target.value)}
            placeholder="Path to sdlarch-rl-master folder"
            spellCheck={false}
          />
          <button className="bridge-btn bridge-btn-save" onClick={savePath}>✓</button>
        </div>
      )}

      {/* Mini log */}
      {log && <div className="bridge-log">{log}</div>}
    </div>
  )
}
