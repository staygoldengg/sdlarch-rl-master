/**
 * LatencyMonitor — System Heartbeat component.
 * Polls /api/latency/benchmark on demand and /api/spatial/telemetry every 2 s.
 * Shows per-stage ms + a STALE warning when total > 100 ms.
 */
import { useState, useEffect, useCallback } from 'react'

const API = 'http://localhost:5000'

type LatencyMetrics = {
  capture_ms?: number
  ocr_ms?: number
  logic_ms?: number
  total_ms?: number
  error?: string
}

type Telemetry = {
  x?: number
  y?: number
  frame?: number
  lookahead?: number
  weapon?: number
  error?: string
}

export default function LatencyMonitor() {
  const [metrics, setMetrics]     = useState<LatencyMetrics>({})
  const [telemetry, setTelemetry] = useState<Telemetry>({})
  const [running, setRunning]     = useState(false)

  const fetchTelemetry = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/spatial/telemetry`)
      if (r.ok) setTelemetry(await r.json())
    } catch {}
  }, [])

  useEffect(() => {
    fetchTelemetry()
    const id = setInterval(fetchTelemetry, 2000)
    return () => clearInterval(id)
  }, [fetchTelemetry])

  async function runBenchmark() {
    setRunning(true)
    try {
      const r = await fetch(`${API}/api/latency/benchmark`)
      if (r.ok) {
        const data: LatencyMetrics = await r.json()
        setMetrics(data)
        // Sync result to fight_driver if total_ms is available
        if (data.total_ms) {
          await fetch(`${API}/api/latency/sync`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ total_ms: data.total_ms }),
          })
        }
      }
    } catch {}
    setRunning(false)
  }

  const total   = metrics.total_ms ?? 0
  const isStale = total > 100

  return (
    <div className="bp-section">
      <div className="bp-section-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span>💓 System Heartbeat</span>
        <button
          className="btn btn-edit"
          style={{ padding: '3px 10px', fontSize: 11 }}
          onClick={runBenchmark}
          disabled={running}>
          {running ? 'Measuring…' : '⏱ Benchmark'}
        </button>
      </div>

      {/* Latency rows */}
      {metrics.error ? (
        <p style={{ color: '#ef4444', fontSize: 12 }}>{metrics.error}</p>
      ) : (
        <div className="latency-grid">
          <LatRow label="Capture" value={metrics.capture_ms} />
          <LatRow label="OCR"     value={metrics.ocr_ms} />
          <LatRow label="Logic"   value={metrics.logic_ms} />
          <div className="lat-row lat-total">
            <span>Total</span>
            <span style={{ color: isStale ? '#ef4444' : '#22c55e' }}>
              {total > 0 ? `${total.toFixed(1)} ms` : '—'}
            </span>
          </div>
          {isStale && (
            <div className="lat-stale">⚠ CAUTION: DATA IS STALE ({total.toFixed(0)} ms &gt; 100 ms)</div>
          )}
        </div>
      )}

      {/* Spatial telemetry */}
      {!telemetry.error && (
        <div className="telemetry-row">
          <span className="tel-item">X: <b>{telemetry.x?.toFixed(1) ?? '—'}</b></span>
          <span className="tel-item">Y: <b>{telemetry.y?.toFixed(1) ?? '—'}</b></span>
          <span className="tel-item">Frame: <b>{telemetry.frame ?? '—'}</b></span>
          <span className="tel-item">Lookahead: <b>{telemetry.lookahead ?? '—'}</b></span>
          <span className="tel-item">Weapon: <b>{telemetry.weapon ?? '—'}</b></span>
        </div>
      )}
    </div>
  )
}

function LatRow({ label, value }: { label: string; value?: number }) {
  return (
    <div className="lat-row">
      <span>{label}</span>
      <span>{value != null ? `${value.toFixed(1)} ms` : '—'}</span>
    </div>
  )
}
