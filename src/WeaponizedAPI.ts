/**
 * WeaponizedAPI.ts
 * TypeScript client for the Python weaponized_ai FastAPI server (port 8000).
 * All endpoints mirror api_server.py.
 */

const BASE = 'http://127.0.0.1:8000'

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`${path} → ${r.status}: ${await r.text()}`)
  return r.json()
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`)
  if (!r.ok) throw new Error(`${path} → ${r.status}`)
  return r.json()
}

// ── Health ────────────────────────────────────────────────────────────────────
export function checkHealth(): Promise<{ status: string; obs_dim: number; act_dim: number }> {
  return get('/health')
}

// ── RL ────────────────────────────────────────────────────────────────────────
export interface InferResult {
  action: number
  log_prob: number
  logits: number[]
  value: number
}

export function policyInfer(obs: number[]): Promise<InferResult> {
  return post('/policy/infer', { obs })
}

export function rlStore(
  obs: number[], action: number, reward: number,
  log_prob: number, done = false
): Promise<{ buffered: number }> {
  return post('/rl/store', { obs, action, reward, log_prob, done })
}

export function rlTrain(): Promise<{ loss_policy: number; loss_value: number }> {
  return post('/rl/train', {})
}

export function rlSave(): Promise<{ saved: boolean }> {
  return post('/rl/save', {})
}

export function rlLoad(): Promise<{ loaded: boolean }> {
  return post('/rl/load', {})
}

export function rlBufferSize(): Promise<{ size: number }> {
  return get('/rl/buffer_size')
}

export interface AgentStats {
  total_steps: number
  total_updates: number
  episode_count: number
  mean_ep_reward: number
  best_ep_reward: number
}

export function rlStats(): Promise<AgentStats> {
  return get('/rl/stats')
}

// ── Input ─────────────────────────────────────────────────────────────────────
export function tapKey(key: string, hold_s = 0.016): Promise<{ status: string; key: string }> {
  return post('/input/tap', { key, hold_s })
}

export function tapVk(vk: number, hold_s = 0.016): Promise<{ status: string; vk: number }> {
  return post('/input/tap', { vk, hold_s })
}

export function executeMacro(name: string): Promise<{ status: string; macro: string }> {
  return post('/input/macro', { name })
}

export function listMacros(): Promise<{ macros: string[] }> {
  return get('/input/macros')
}

// ── Strategy ──────────────────────────────────────────────────────────────────
export interface PlayerStatePayload {
  player_id?: string
  pos: { x: number; y: number }
  vel: { x: number; y: number }
  is_airborne?: boolean
  is_invulnerable?: boolean
  is_attacking?: boolean
  current_move?: string
  last_move?: string
  buff_active?: boolean
  buff_expires_at?: number
  stocks_remaining?: number
  damage?: number
}

export interface StrategyResult {
  strategy: {
    id: string; name: string; description: string
    priority: number; suggested_moves: string[]
  } | null
  all_ranked: { id: string; name: string; priority: number; suggested_moves: string[] }[]
  lead_direction: { x: number; y: number } | null
  target_landing: { x: number; y: number; time_ms: number } | null
}

export function rankStrategies(me: PlayerStatePayload, opp: PlayerStatePayload): Promise<StrategyResult> {
  return post('/strategy/rank', { me, opp })
}

export function getAllStrategies(): Promise<{
  id: string; name: string; description: string; priority: number; suggested_moves: string[]
}[]> {
  return get('/strategy/all')
}

export function vecDist(x1: number, y1: number, x2: number, y2: number): Promise<{ distance: number }> {
  return post('/vec/dist', { x1, y1, x2, y2 })
}

export function projectileLead(
  shooter: PlayerStatePayload,
  target: PlayerStatePayload,
  projectile_speed = 800
): Promise<{ lead: { x: number; y: number } | null }> {
  return post('/strategy/lead', { shooter, target, projectile_speed })
}

// ── Training Loop ─────────────────────────────────────────────────────────────
export interface LoopStatus {
  running: boolean
  step_count: number
  last_reward: number
  last_action: string
  loss_policy: number
  loss_value: number
  entropy: number
  // Episode stats
  episode_count: number
  current_ep_reward: number
  best_ep_reward: number
  mean_ep_reward: number
  total_updates: number
  // Damage / stocks
  last_p1_damage: number
  last_p2_damage: number
  last_p1_stocks: number
  last_p2_stocks: number
  // Enhanced perception fields
  last_ko_flash?: boolean
  last_p1_damage_tier?: string
  last_p2_damage_tier?: string
  last_p1_weapon?: string
  last_p2_weapon?: string
  last_stage_pickups?: number
  errors: string[]
}

export interface LoopStartOptions {
  capture_mode?: 'mss' | 'obs'
  obs_camera_index?: number
  tick_s?: number
  region_left?: number
  region_top?: number
  region_width?: number
  region_height?: number
}

export function startLoop(opts: LoopStartOptions = {}): Promise<{ status: string }> {
  return post('/loop/start', opts)
}

export function stopLoop(): Promise<{ status: string }> {
  return post('/loop/stop', {})
}

export function getLoopStatus(): Promise<LoopStatus> {
  return get('/loop/status')
}

// ── OBS Management ────────────────────────────────────────────────────────────
export interface OBSStatus {
  installed: boolean
  install_path: string | null
  running: boolean
  obs_pid: number | null
  camera_index: number | null
  downloading: boolean
  download_pct: number
  installing: boolean
  install_done: boolean
  install_error: string | null
  install_message: string
}

export function getOBSStatus(): Promise<OBSStatus> {
  return get('/obs/status')
}

export function launchOBS(): Promise<{ success: boolean; message: string }> {
  return post('/obs/launch', {})
}

export function installOBS(): Promise<{ status: string; message: string }> {
  return post('/obs/install', {})
}

export function ensureOBS(): Promise<OBSStatus> {
  return post('/obs/ensure', {})
}

export function getOBSCameraIndex(): Promise<{ camera_index: number }> {
  return get('/obs/camera-index')
}

export function setOBSRegion(left: number, top: number, width: number, height: number) {
  return post('/obs/set-region', { left, top, width, height })
}

export function calibrateOBS(): Promise<{ status: string; window: { left: number; top: number; width: number; height: number } }> {
  return post('/obs/calibrate', {})
}

// ── YouTube Video Learning ────────────────────────────────────────────────────

export interface VideoStatus {
  state: 'idle' | 'downloading' | 'transcribing' | 'extracting' | 'pretraining' | 'done' | 'error'
  progress: number          // 0..1
  current_url: string
  video_title: string
  frames_total: number
  frames_done: number
  transitions_extracted: number
  corpus_size: number
  bc_epochs_done: number
  bc_loss_last: number
  bc_loss_initial: number
  transcribed: boolean
  knowledge_terms: number
  top_terms: [string, number][]   // top-10 [term, count] pairs
  error: string
  log: string[]
}

export function ingestVideo(url: string, maxFrames = 8000, transcribe = true): Promise<{ status: string; url: string }> {
  return post('/video/ingest', { url, max_frames: maxFrames, transcribe })
}

export function getVideoStatus(): Promise<VideoStatus> {
  return get('/video/status')
}

export function pretrainFromVideo(nEpochs = 20, batchSize = 128, lr = 3e-4): Promise<{ status: string; corpus_size: number }> {
  return post('/video/pretrain', { n_epochs: nEpochs, batch_size: batchSize, lr })
}

export function resetVideo(): Promise<{ status: string }> {
  return post('/video/reset', {})
}

export function getVideoKnowledge(): Promise<{ terms: { term: string; count: number }[] }> {
  return get('/video/knowledge')
}

// ── Brain Storage ─────────────────────────────────────────────────────────────

export interface IngestedVideo {
  url:             string
  title:           string
  frames:          number
  transitions:     number
  times_ingested:  number
  first_ingested:  string
  last_ingested:   string
}

export interface BrainInfo {
  knowledge_terms: number
  top_terms:       [string, number][]
  corpus_size:     number
  corpus_bytes:    number
  corpus_mb:       number
  videos_ingested: number
  registry:        IngestedVideo[]
  model_exists:    boolean
  model_bytes:     number
  model_mb:        number
  brain_dir:       string
}

export function getBrainInfo(): Promise<BrainInfo> {
  return get('/brain/info')
}

export function saveBrain(): Promise<{ status: string; corpus_written: number; knowledge_terms: number }> {
  return post('/brain/save', {})
}

export function reloadBrain(): Promise<{ status: string; knowledge_terms: number; corpus_size: number }> {
  return post('/brain/reload', {})
}

export function clearBrain(): Promise<{ status: string }> {
  return post('/brain/clear', { confirm: true })
}

// ── Replay Engine ─────────────────────────────────────────────────────────────
export interface ReplayFileMeta {
  path:         string
  name:         string
  size_kb:      number
  parse_ok:     boolean
  frame_count:  number
  stage:        string
  game_version: string
  level_id:     number
  game_mode:    number
  characters:   string[]
  error?:       string
}

export interface ReplayScanResult {
  replay_dir: string
  count:      number
  replays:    ReplayFileMeta[]
}

export interface ReplayIngestResult {
  status:       string
  path:         string
  title:        string
  transitions:  number
  frame_count:  number
  stage?:       string
  game_version?: string
  detail?:      string
}

export interface ReplayIngestAllResult {
  files_found:       number
  files_ingested:    number
  total_transitions: number
  results:           Array<{path: string; title?: string; transitions: number; status: string}>
}

export function replayScan(): Promise<ReplayScanResult> {
  return get('/replay/scan')
}

export function replayIngest(path: string, maxTransitions = 20000): Promise<ReplayIngestResult> {
  return post('/replay/ingest', { path, max_transitions: maxTransitions })
}

export function replayIngestAll(): Promise<ReplayIngestAllResult> {
  return post('/replay/ingest_all', {})
}

// ── Live Memory Reader ────────────────────────────────────────────────────────
export interface MemoryInfo {
  attached:         boolean
  pid:              number
  addresses_found:  string[]
  scan_done:        boolean
}

export interface LiveState {
  obs:    number[]
  p1:     { damage: number; stocks: number; x: number; y: number; airborne: boolean }
  p2:     { damage: number; stocks: number; x: number; y: number }
  source: string
}

export function getMemoryInfo(): Promise<MemoryInfo> {
  return get('/memory/info')
}

export function getMemoryState(): Promise<LiveState> {
  return get('/memory/state')
}

export function rescanMemory(): Promise<MemoryInfo> {
  return post('/memory/rescan', {})
}

// ── BTR Agent ─────────────────────────────────────────────────────────────────
export interface BTRStats {
  agent:           string
  total_steps:     number
  total_updates:   number
  episode_count:   number
  replay_size:     number
  mean_ep_reward:  number
  best_ep_reward:  number
}

export function getBTRStats(): Promise<BTRStats> {
  return get('/btr/stats')
}

export function btrAction(obs: number[]): Promise<{ action: number }> {
  return post('/btr/action', { obs })
}

export function saveBTR(): Promise<{ saved: boolean }> {
  return post('/btr/save', {})
}

export function btrPretrain(): Promise<{ status: string; bc_loss: number; samples: number }> {
  return post('/btr/pretrain', {})
}

