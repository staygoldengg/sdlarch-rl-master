import { useState, useRef, useCallback } from 'react'
import './App.css'
import { FightingAI } from './ai/FightingAI'
import type { Prediction, HabitEntry, PatternEntry } from './ai/FightingAI'
import { getCurrentWindow } from '@tauri-apps/api/window'
import { LogicalSize } from '@tauri-apps/api/dpi'
import { invoke } from '@tauri-apps/api/core'
import SimulationTab from './SimulationTab'
import FightTab from './FightTab'
import CaptureTab from './CaptureTab'
import StrategyTab from './StrategyTab'
import IntelTab from './IntelTab'
import BackendBridge from './BackendBridge'

type Macro = {
  id: number
  name: string
  legend: string
  weapon: string
  steps: string
  aiPrompt: string
  running: boolean
}

const LEGENDS = [
  'Any Legend', 'Ada', 'Arcadia', 'Artemis', 'Asuri', 'Azoth',
  'Barraza', 'Bodvar', 'Brynn', 'Caspian', 'Cassidy', 'Cross',
  'Diana', 'Ember', 'Fait', 'Gnash', 'Hattori', 'Isaiah',
  'Jaeyun', 'Jhala', 'Jiro', 'Katarina', 'Kaya', 'Koji',
  'Kor', 'Lin Fei', 'Lucien', 'Magyar', 'Mako', 'Mirage',
  'Mordex', 'Nix', 'Onyx', 'Orion', 'Petra', 'Queen Nai',
  'Ragnir', 'Rayman', 'Reno', 'Scarlet', 'Sentinel', 'Sidra',
  'Sir Roland', 'TarTar', 'Teros', 'Thatch', 'Thor', 'Ulgrim',
  'Val', 'Vector', 'Volkov', 'Wu Shang', 'Xull', 'Yumiko', 'Zariel',
]

const WEAPONS = [
  'Any Weapon', 'Gauntlets', 'Katars', 'Bow', 'Scythe',
  'Blasters', 'Orb', 'Axe', 'Sword', 'Spear',
  'Hammer', 'Lance', 'Cannon', 'Greatsword', 'Boots',
]

type WeaponTemplate = { name: string; notation: string; steps: string }
const WEAPON_TEMPLATES: Record<string, WeaponTemplate[]> = {
  Gauntlets: [
    { name: 'DLight → Dair → Rec', notation: 'DLight > Dair > Rec', steps: 'Press S+N\nWait 60ms\nPress Space+S+N\nWait 80ms\nPress M' },
    { name: 'DLight → Nair → Rec', notation: 'DLight > Nair > Rec (0–110%)', steps: 'Press S+N\nWait 60ms\nPress Space+N\nWait 80ms\nPress M' },
    { name: 'DLight → Sair → GC SLight → Rec', notation: 'DLight > Sair > GC SLight > Rec (110%+)', steps: 'Press S+N\nWait 60ms\nPress Space+D+N\nWait 80ms\nPress D+N\nWait 50ms\nPress M' },
  ],
  Katars: [
    { name: 'Dair → DLight → Rec', notation: 'Dair > DLight > Rec', steps: 'Press Space+S+N\nWait 80ms\nPress S+N\nWait 60ms\nPress M' },
    { name: 'SLight → DLight → Dair', notation: 'SLight > DLight > Dair', steps: 'Press D+N\nWait 60ms\nPress S+N\nWait 60ms\nPress Space+S+N' },
    { name: 'DLight → Dair → DLight → Rec → Rec', notation: 'DLight > Dair > DLight > Rec > Rec', steps: 'Press S+N\nWait 60ms\nPress Space+S+N\nWait 80ms\nPress S+N\nWait 60ms\nPress M\nWait 60ms\nPress M' },
    { name: 'Nair → DLight → Nair (loop)', notation: 'Nair > DLight > Nair (low risk loop)', steps: 'Press Space+N\nWait 80ms\nPress S+N\nWait 60ms\nPress Space+N' },
    { name: 'SLight → Dair → NLight → Rec', notation: 'SLight > Dair > NLight > Rec', steps: 'Press D+N\nWait 60ms\nPress Space+S+N\nWait 80ms\nPress N\nWait 60ms\nPress M' },
  ],
  Bow: [
    { name: 'DLight → Sair', notation: 'DLight > Sair', steps: 'Press S+N\nWait 70ms\nPress Space+D+N' },
    { name: 'SLight → DLight → DAir', notation: 'SLight > DLight > DAir', steps: 'Press D+N\nWait 60ms\nPress S+N\nWait 60ms\nPress Space+S+N' },
    { name: 'DLight → NLight → Rec', notation: 'DLight > NLight > Rec/Nair', steps: 'Press S+N\nWait 70ms\nPress N\nWait 70ms\nPress M' },
  ],
  Scythe: [
    { name: 'NLight → Nair → Dair → GP', notation: 'NLight > Nair > Dair > GP', steps: 'Press N\nWait 60ms\nPress Space+N\nWait 80ms\nPress Space+S+N\nWait 80ms\nPress S+N' },
    { name: 'Sair → Dair → Rec', notation: 'Sair > Dair > Rec', steps: 'Press Space+D+N\nWait 80ms\nPress Space+S+N\nWait 80ms\nPress M' },
    { name: 'SLight → Nair → Dair → GP', notation: 'SLight > Nair > Dair > GP', steps: 'Press D+N\nWait 60ms\nPress Space+N\nWait 80ms\nPress Space+S+N\nWait 80ms\nPress S+N' },
    { name: 'DLight → Nair → GP (B-Dair)', notation: 'DLight > Nair > GP (offstage B-Dair)', steps: 'Press S+N\nWait 60ms\nPress Space+N\nWait 80ms\nPress S+N' },
    { name: 'A-SLight → SLight → Nair', notation: 'A-SLight > SLight > Nair (grounded)', steps: 'Press D\nWait 10ms\nPress N\nWait 60ms\nPress D+N\nWait 60ms\nPress Space+N' },
  ],
  Blasters: [
    { name: 'DJFF Sair → NLight', notation: 'DJFF Sair > NLight', steps: 'Press Space\nWait 30ms\nPress Space\nWait 30ms\nPress S\nWait 30ms\nPress D+N\nWait 80ms\nPress N' },
    { name: 'NLight → X-Pivot Nair', notation: 'NLight > X-Pivot Nair', steps: 'Press N\nWait 60ms\nPress A\nWait 20ms\nPress Space+N' },
    { name: 'Sair → GC DLight', notation: 'Sair > GC DLight (gravity cancel)', steps: 'Press Space+D+N\nWait 60ms\nPress Space\nWait 20ms\nPress S+N' },
    { name: 'Dair → Sair → CD DLight', notation: 'Dair > Sair > CD DLight', steps: 'Press Space+S+N\nWait 80ms\nPress Space+D+N\nWait 70ms\nPress S+N' },
    { name: 'Dash Jump Sair → Nair', notation: 'Dash Jump SLight > Nair', steps: 'Press D\nWait 20ms\nPress Space\nWait 30ms\nPress D+N\nWait 70ms\nPress Space+N' },
  ],
  Orb: [
    { name: 'SLight → Sair', notation: 'SLight > Sair', steps: 'Press D+N\nWait 60ms\nPress Space+D+N' },
    { name: 'Dair → SLight → DLight → Nair → Rec', notation: 'Dair > SLight > DLight > Nair > Rec', steps: 'Press Space+S+N\nWait 80ms\nPress D+N\nWait 60ms\nPress S+N\nWait 60ms\nPress Space+N\nWait 80ms\nPress M' },
    { name: 'SLight → NLight', notation: 'SLight > NLight (low %)', steps: 'Press D+N\nWait 60ms\nPress N' },
  ],
  Axe: [
    { name: 'SLight → Nair → Dair', notation: 'SLight > Nair > Dair', steps: 'Press D+N\nWait 70ms\nPress Space+N\nWait 80ms\nPress Space+S+N' },
    { name: 'SLight → Nair → Rec', notation: 'SLight > Nair > Rec', steps: 'Press D+N\nWait 70ms\nPress Space+N\nWait 80ms\nPress M' },
    { name: 'Dair Spike', notation: 'Offstage Dair Spike', steps: 'Press Space\nWait 50ms\nPress Space+S+N' },
  ],
  Sword: [
    { name: 'Dair → GP', notation: 'Dair > GP (edge kill)', steps: 'Press Space+S+N\nWait 80ms\nPress S+N' },
    { name: 'NLight → Nair', notation: 'NLight > Nair', steps: 'Press N\nWait 60ms\nPress Space+N' },
    { name: 'DLight → Nair', notation: 'DLight > Nair', steps: 'Press S+N\nWait 60ms\nPress Space+N' },
  ],
  Spear: [
    { name: 'SLight Chain', notation: 'SLight > SLight > DLight > DLight > Nair > Sair', steps: 'Press D+N\nWait 60ms\nPress D+N\nWait 60ms\nPress S+N\nWait 60ms\nPress S+N\nWait 60ms\nPress Space+N\nWait 80ms\nPress Space+D+N' },
    { name: 'NLight Poke', notation: 'NLight (low startup poke)', steps: 'Press N\nWait 60ms\nPress N' },
  ],
  Hammer: [
    { name: 'DLight → Nair', notation: 'DLight > Nair', steps: 'Press S+N\nWait 70ms\nPress Space+N' },
    { name: 'SLight → Nair → GP', notation: 'SLight > Nair > GP', steps: 'Press D+N\nWait 70ms\nPress Space+N\nWait 80ms\nPress S+N' },
    { name: 'Rec Gimp', notation: 'Offstage Rec gimp', steps: 'Press Space\nWait 50ms\nPress M' },
  ],
  Lance: [
    { name: 'SLight → Dair', notation: 'SLight > Dair', steps: 'Press D+N\nWait 70ms\nPress Space+S+N' },
    { name: 'DJFF Sair → NLight', notation: 'DJFF Sair > NLight', steps: 'Press Space\nWait 30ms\nPress Space\nWait 30ms\nPress S\nWait 30ms\nPress D+N\nWait 80ms\nPress N' },
  ],
  Cannon: [
    { name: 'GP → Dair', notation: 'GP > Dair', steps: 'Press S+N\nWait 80ms\nPress Space+S+N' },
    { name: 'Rec Recovery', notation: 'Rec (safe return)', steps: 'Press Space\nWait 40ms\nPress M' },
    { name: 'SLight → Nair → GP', notation: 'SLight > Nair > GP', steps: 'Press D+N\nWait 70ms\nPress Space+N\nWait 80ms\nPress S+N' },
  ],
  Greatsword: [
    { name: 'NLight → Nair', notation: 'NLight > Nair', steps: 'Press N\nWait 70ms\nPress Space+N' },
    { name: 'DLight → GP', notation: 'DLight > GP (slow but powerful)', steps: 'Press S+N\nWait 80ms\nPress S+N' },
    { name: 'SLight → Nair → Rec', notation: 'SLight > Nair > Rec', steps: 'Press D+N\nWait 80ms\nPress Space+N\nWait 90ms\nPress M' },
  ],
  Boots: [
    { name: 'SLight → Nair → Sair', notation: 'SLight > Nair > Sair', steps: 'Press D+N\nWait 60ms\nPress Space+N\nWait 70ms\nPress Space+D+N' },
    { name: 'DLight → Nair → Rec', notation: 'DLight > Nair > Rec', steps: 'Press S+N\nWait 60ms\nPress Space+N\nWait 80ms\nPress M' },
    { name: 'Sair → DLight → Nair', notation: 'Sair > DLight > Nair', steps: 'Press Space+D+N\nWait 70ms\nPress S+N\nWait 60ms\nPress Space+N' },
  ],
}

const GLOSSARY = [
  { term: 'NLight', def: 'Neutral light attack (jab) — fast, low startup poke' },
  { term: 'SLight', def: 'Side light attack — good string starter' },
  { term: 'DLight', def: 'Down light attack — common combo opener' },
  { term: 'Nair', def: 'Neutral aerial attack' },
  { term: 'Sair', def: 'Side aerial attack — great for edge carries' },
  { term: 'Dair', def: 'Down aerial attack — spike/offstage tool' },
  { term: 'Rec', def: 'Recovery attack — used for combos AND recovering' },
  { term: 'GP', def: 'Ground Pound — hold Down then light (S+N)' },
  { term: 'NSig', def: 'Neutral heavy/signature — high risk, high reward' },
  { term: 'SSig', def: 'Side heavy/signature' },
  { term: 'DSig', def: 'Down heavy/signature' },
  { term: 'DJFF', def: 'Dash Jump Fast Fall — for aerial approach' },
  { term: 'PC', def: 'Platform Cancel — land on platform to cancel aerial' },
  { term: 'WT', def: 'Weapon Throw — throw weapon for pressure/combos' },
  { term: 'Neutral', def: 'Neither player has positional advantage' },
  { term: 'Advantage', def: 'You have more options than your opponent' },
  { term: 'Disadvantage', def: 'Opponent has more options; you need to escape' },
  { term: 'Footsies', def: 'Spacing game — controlling empty space between you and opponent' },
  { term: 'Hit Confirm', def: 'Safe move that leads to full combo on hit, safe on miss' },
  { term: 'Gimp', def: 'Interrupt or stop an opponent\'s recovery offstage' },
  { term: 'Overtuned', def: 'Weapon with little to no weaknesses' },
  { term: 'Undertuned', def: 'Weapon with little serviceable strengths' },
  { term: 'Hitbox', def: 'The area of effect of an attack' },
  { term: 'Hurtbox', def: 'The area on your character that gets hit' },
  { term: 'Startup Frames', def: 'Frames before a move becomes active (lower = faster)' },
  { term: 'Recovery Frames', def: 'Frames you\'re locked in after an attack' },
  { term: 'Priority', def: 'A move wins because of larger hitbox or fewer startup frames' },
  { term: 'GC', def: 'Gravity Cancel — land on a platform mid-aerial to cancel recovery frames and act sooner' },
  { term: 'CD / cDodge', def: 'Chase Dodge — dodge forward to chase the opponent and extend combos' },
  { term: 'X-Pivot', def: 'Tap opposite direction just before attacking to reverse the move\'s direction without losing momentum' },
  { term: 'Chunking', def: 'Consistently landing safe moves to build up % advantage over time' },
  { term: 'Blindspot', def: 'A position your opponent\'s attacks don\'t reach — exploit it to approach safely' },
  { term: 'Space Management', def: 'Actively controlling the space between you and your opponent to force favorable exchanges' },
  { term: 'Safe (neutral)', def: 'Less likely to land, but also less likely to be punished — prolongs neutral' },
  { term: 'Coverage', def: 'More likely to land a hit, but more likely to be punished if it misses — breaks neutral' },
  { term: 'Engaging', def: 'Pressing forward in neutral to force an interaction with your opponent' },
  { term: 'Disengage', def: 'Retreating from disadvantage or bad position to reset to neutral' },
  { term: 'PICKUP', def: 'Picking up a weapon from the ground' },
]

const MOVE_GROUPS = [
  { label: 'Lights',  color: 'light',   moves: ['NLight', 'SLight', 'DLight'] },
  { label: 'Aerials', color: 'air',     moves: ['Nair', 'Sair', 'Dair'] },
  { label: 'Special', color: 'special', moves: ['Rec', 'GP', 'WT'] },
  { label: 'Sigs',    color: 'sig',     moves: ['NSig', 'SSig', 'DSig'] },
  { label: 'Tech',    color: 'tech',    moves: ['DJFF', 'GC', 'CD', 'X-Pivot'] },
] as const

type ScoutState = {
  history: string[]
  predictions: Prediction[]
  habits: HabitEntry[]
  patterns: PatternEntry[]
  total: number
}

const defaultScoutState: ScoutState = { history: [], predictions: [], habits: [], patterns: [], total: 0 }

const defaultMacro: Omit<Macro, 'id' | 'running'> = {
  name: '',
  legend: 'Any Legend',
  weapon: 'Any Weapon',
  steps: '',
  aiPrompt: '',
}

function App() {
  const [macros, setMacros] = useState<Macro[]>([
    {
      id: 1,
      name: 'Katars — Safe Loop',
      legend: 'Asuri',
      weapon: 'Katars',
      steps: 'Press Space+N\nWait 80ms\nPress S+N\nWait 60ms\nPress Space+N',
      aiPrompt: '',
      running: false,
    },
    {
      id: 2,
      name: 'Gauntlets — DLight Combo',
      legend: 'Any Legend',
      weapon: 'Gauntlets',
      steps: 'Press S+N\nWait 60ms\nPress Space+N\nWait 80ms\nPress M',
      aiPrompt: '',
      running: false,
    },
    {
      id: 3,
      name: 'Sword — Dair Spike',
      legend: 'Bodvar',
      weapon: 'Sword',
      steps: 'Press Space\nWait 50ms\nPress S+N\nWait 80ms\nPress S+N',
      aiPrompt: '',
      running: false,
    },
    {
      id: 4,
      name: 'Scythe — Nair Chase',
      legend: 'Ember',
      weapon: 'Scythe',
      steps: 'Press N\nWait 60ms\nPress Space+N\nWait 80ms\nPress Space+S+N\nWait 80ms\nPress S+N',
      aiPrompt: '',
      running: false,
    },
  ])
  const [selected, setSelected] = useState<Macro | null>(null)
  const [form, setForm] = useState(defaultMacro)
  const [isEditing, setIsEditing] = useState(false)
  const [log, setLog] = useState<string[]>([])
  const [aiKey, setAiKey] = useState(() => localStorage.getItem('bh_openai_key') ?? '')
  const [aiResult, setAiResult] = useState('')
  const [aiLoading, setAiLoading] = useState(false)
  const [tab, setTab] = useState<'macros' | 'ai' | 'glossary' | 'scout' | 'settings' | 'simulation' | 'fight' | 'capture' | 'strategy' | 'intel'>('macros')
  const [glossaryFilter, setGlossaryFilter] = useState('')
  const scoutAI = useRef(new FightingAI())
  const [scout, setScout] = useState<ScoutState>(defaultScoutState)
  const [coachingMode, setCoachingMode] = useState(false)
  const [coachingCompact, setCoachingCompact] = useState(false)

  // API credentials — persisted in localStorage, pushed to Flask on save
  const [bhApiKey, setBhApiKey] = useState(() => localStorage.getItem('bh_api_key') ?? '')
  const [twitchClientId, setTwitchClientId] = useState(() => localStorage.getItem('twitch_client_id') ?? '')
  const [twitchClientSecret, setTwitchClientSecret] = useState(() => localStorage.getItem('twitch_client_secret') ?? '')
  const [ocrRegion, setOcrRegion] = useState(() => {
    try { return JSON.parse(localStorage.getItem('ocr_region') ?? '') } catch { return { top: 400, left: 1500, width: 300, height: 400 } }
  })
  const [settingsSaved, setSettingsSaved] = useState(false)

  async function saveAndPushSettings() {
    localStorage.setItem('bh_openai_key',       aiKey)
    localStorage.setItem('bh_api_key',           bhApiKey)
    localStorage.setItem('twitch_client_id',     twitchClientId)
    localStorage.setItem('twitch_client_secret', twitchClientSecret)
    localStorage.setItem('ocr_region',           JSON.stringify(ocrRegion))
    // Push to Flask if online
    try {
      await fetch('http://localhost:5000/api/meta/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bh_api_key:           bhApiKey,
          twitch_client_id:     twitchClientId,
          twitch_client_secret: twitchClientSecret,
        }),
      })
      await fetch('http://localhost:5000/api/meta/ocr-region', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ocrRegion),
      })
    } catch { /* server may be offline — credentials are saved locally */ }
    setSettingsSaved(true)
    setTimeout(() => setSettingsSaved(false), 2500)
  }

  function selectMacro(m: Macro) {
    setSelected(m)
    setForm({ name: m.name, legend: m.legend, weapon: m.weapon, steps: m.steps, aiPrompt: m.aiPrompt })
    setIsEditing(false)
  }

  function newMacro() {
    setSelected(null)
    setForm(defaultMacro)
    setIsEditing(true)
  }

  function loadTemplate(t: WeaponTemplate) {
    setForm(f => ({ ...f, name: t.name, steps: t.steps }))
  }

  function recordMove(move: string) {
    scoutAI.current.record(move)
    setScout({
      history:     scoutAI.current.getHistory(),
      predictions: scoutAI.current.predict(),
      habits:      scoutAI.current.getHabits(),
      patterns:    scoutAI.current.getPatterns(),
      total:       scoutAI.current.getTotal(),
    })
  }

  function resetScout() {
    scoutAI.current.reset()
    setScout(defaultScoutState)
  }

  const toggleCoachingMode = useCallback(async () => {
    const win = getCurrentWindow()
    if (!coachingMode) {
      // Activate: float on top, shrink to compact bar
      await win.setAlwaysOnTop(true)
      await win.setSize(new LogicalSize(520, 160))
      await win.setResizable(false)
      setCoachingCompact(true)
      setCoachingMode(true)
      setTab('scout')
    } else {
      // Deactivate: restore normal window
      await win.setAlwaysOnTop(false)
      await win.setResizable(true)
      await win.setSize(new LogicalSize(1100, 700))
      setCoachingCompact(false)
      setCoachingMode(false)
    }
  }, [coachingMode])

  function saveMacro() {
    if (!form.name.trim()) return
    if (selected) {
      setMacros(macros.map(m => m.id === selected.id ? { ...m, ...form } : m))
      setSelected({ ...selected, ...form })
    } else {
      const next: Macro = { id: Date.now(), ...form, running: false }
      setMacros([...macros, next])
      setSelected(next)
    }
    setIsEditing(false)
  }

  function deleteMacro(id: number) {
    setMacros(macros.filter(m => m.id !== id))
    if (selected?.id === id) { setSelected(null); setIsEditing(false) }
  }

  async function runMacro(m: Macro) {
    setMacros(macros.map(x => x.id === m.id ? { ...x, running: true } : x))
    const steps = m.steps.split('\n').filter(Boolean)
    const ts = new Date().toLocaleTimeString()
    setLog(l => [`[${ts}] Running "${m.name}"...`, ...l])
    try {
      await invoke('execute_macro', { steps })
      setLog(l => [`[${ts}] "${m.name}" finished.`, ...l])
    } catch (e) {
      setLog(l => [`[${ts}] Error: ${String(e)}`, ...l])
    }
    setMacros(prev => prev.map(x => x.id === m.id ? { ...x, running: false } : x))
  }

  async function askAI() {
    if (!form.aiPrompt.trim()) return
    setAiLoading(true)
    setAiResult('')
    try {
      const res = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${aiKey}`,
        },
        body: JSON.stringify({
          model: 'gpt-4o',
          messages: [
            { role: 'system', content: `You are an expert Brawlhalla combo and macro assistant with deep knowledge of the meta.
Move notation: NLight=neutral light, SLight=side light, DLight=down light, Nair=neutral aerial, Sair=side aerial, Dair=down aerial, Rec=recovery attack, GP=ground pound (hold S then N), NSig/SSig/DSig=signatures.
Keys: WASD=movement, N=light attack, M=heavy attack, Space=jump, S+N=ground pound, D+N=side light, S+M=ground slam.
Weapon tier (best to worst optimization): Gauntlets, Katars, Bow, Scythe, Blasters, Orb, Axe, Sword, Spear, Hammer, Lance, Cannon.
Known combos: Gauntlets: DLight>Dair>Rec, DLight>Nair>Rec. Katars: SLight>DLight>Dair, Dair>DLight>Rec. Bow: DLight>Sair, DLight>NLight>Rec. Scythe: aerial chains, NLight>Nair>Dair>GP. Blasters: DJFF Sair>NLight, NLight>Nair. Sword: Dair>GP (edge), NLight>Nair. Spear: SLight>SLight>DLight>DLight>Nair>Sair.
Output a macro as a list of steps (one per line): Press [KEY], Wait [MS]ms, Hold [KEY], Release [KEY]. Be precise with timing (typical step timing 50-120ms).` },
            { role: 'user', content: form.aiPrompt },
          ],
          max_tokens: 300,
        }),
      })
      const data = await res.json()
      const text = data.choices?.[0]?.message?.content ?? 'No response.'
      setAiResult(text)
      setForm(f => ({ ...f, steps: text }))
    } catch {
      setAiResult('Error contacting AI. Check your API key.')
    }
    setAiLoading(false)
  }

  return (
    <div className={`app${coachingMode ? ' coaching-active' : ''}`}>
      {/* ══ COACHING OVERLAY BAR (compact always-on-top mode) ══ */}
      {coachingCompact && (
        <div className="coaching-bar">
          <div className="coaching-bar-left">
            <span className="coaching-badge">🧠 LIVE</span>
            {scout.predictions.length > 0 ? (
              <>
                <span className="coaching-label">PREDICT</span>
                <span className="coaching-move">{scout.predictions[0].move}</span>
                <span className="coaching-conf">{scout.predictions[0].confidence}%</span>
                <span className="coaching-sep">│</span>
                <span className="coaching-counter">{scout.predictions[0].counter[0]}</span>
              </>
            ) : (
              <span className="coaching-idle">Recording… {scout.total} moves</span>
            )}
          </div>
          <div className="coaching-bar-right">
            {MOVE_GROUPS.map(g => g.moves.map(move => (
              <button key={move} className={`coaching-rec-btn scout-btn-${g.color}`} onClick={() => recordMove(move)}>
                {move}
              </button>
            )))}
            <button className="coaching-exit" onClick={toggleCoachingMode} title="Exit coaching mode">✕</button>
          </div>
        </div>
      )}
      <aside className={`sidebar${coachingCompact ? ' hidden' : ''}`}>
        <div className="sidebar-header">
          <span className="logo-icon">⚔</span>
          <span className="logo-text">BH · Macro Suite</span>
        </div>
        <BackendBridge />
        <nav className="sidebar-nav">
          <button className={tab === 'macros' ? 'nav-btn active' : 'nav-btn'} onClick={() => setTab('macros')}>⚡ Combos</button>
          <button className={tab === 'ai' ? 'nav-btn active' : 'nav-btn'} onClick={() => setTab('ai')}>🤖 AI Generate</button>
          <button className={tab === 'glossary' ? 'nav-btn active' : 'nav-btn'} onClick={() => setTab('glossary')}>📖 Glossary</button>
          <button className={tab === 'scout' ? 'nav-btn active' : 'nav-btn'} onClick={() => setTab('scout')}>🧠 Scout AI</button>
          <button className={tab === 'settings' ? 'nav-btn active' : 'nav-btn'} onClick={() => setTab('settings')}>⚙️ Settings</button>
          <button className={tab === 'simulation' ? 'nav-btn active' : 'nav-btn'} onClick={() => setTab('simulation')}>🎮 Simulation</button>
          <div className="nav-divider" />
          <button className={tab === 'fight' ? 'nav-btn active' : 'nav-btn'} onClick={() => setTab('fight')}>⚔️ Fight Engine</button>
          <button className={tab === 'capture' ? 'nav-btn active' : 'nav-btn'} onClick={() => setTab('capture')}>📷 Capture</button>
          <button className={tab === 'strategy' ? 'nav-btn active' : 'nav-btn'} onClick={() => setTab('strategy')}>🧬 Strategy AI</button>
          <button className={tab === 'intel' ? 'nav-btn active' : 'nav-btn'} onClick={() => setTab('intel')}>🧠 Intel Hub</button>
        </nav>
        <div className="macro-list">
          {macros.map(m => (
            <div
              key={m.id}
              className={`macro-item${selected?.id === m.id ? ' selected' : ''}`}
              onClick={() => selectMacro(m)}
            >
              <div className="macro-item-top">
                <span className="macro-name">{m.name}</span>
                <button
                  className={`macro-run-btn${m.running ? ' running' : ''}`}
                  disabled={m.running}
                  title="Run macro"
                  onClick={e => { e.stopPropagation(); runMacro(m) }}
                >
                  {m.running ? '⏳' : '▶'}
                </button>
              </div>
              <div className="macro-meta">
                <span className="macro-game">{m.legend}</span>
                {m.weapon && m.weapon !== 'Any Weapon' && <span className="weapon-tag">{m.weapon}</span>}
              </div>
            </div>
          ))}
        </div>
        <button className="btn-new" onClick={newMacro}>+ New Combo</button>
      </aside>

      <main className={`main${coachingCompact ? ' hidden' : ''}`}>
        {tab === 'macros' && (
          <>
            {(selected || isEditing) ? (
              <div className="editor">
                <div className="editor-header">
                  <h2>{isEditing ? (selected ? 'Edit Combo' : 'New Combo') : selected?.name}</h2>
                  <div className="editor-actions">
                    {!isEditing && selected && (
                      <>
                        <button className="btn btn-run" disabled={selected.running} onClick={() => runMacro(selected)}>
                          {selected.running ? '⏳ Running...' : '▶ Run'}
                        </button>
                        <button className="btn btn-edit" onClick={() => setIsEditing(true)}>✏️ Edit</button>
                        <button className="btn btn-delete" onClick={() => deleteMacro(selected.id)}>🗑 Delete</button>
                      </>
                    )}
                    {isEditing && (
                      <>
                        <button className="btn btn-save" onClick={saveMacro}>💾 Save</button>
                        <button className="btn btn-cancel" onClick={() => { setIsEditing(false); if (!selected) { setSelected(null) } }}>Cancel</button>
                      </>
                    )}
                  </div>
                </div>

                {isEditing ? (
                  <div className="form">
                    <label>Combo Name
                      <input value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} placeholder="e.g. Scythe Edge Carry" />
                    </label>
                    <div className="form-row">
                      <label>Legend
                        <select value={form.legend} onChange={e => setForm(f => ({ ...f, legend: e.target.value }))} className="legend-select">
                          {LEGENDS.map(l => <option key={l} value={l}>{l}</option>)}
                        </select>
                      </label>
                      <label>Weapon
                        <select value={form.weapon} onChange={e => setForm(f => ({ ...f, weapon: e.target.value }))} className="legend-select">
                          {WEAPONS.map(w => <option key={w} value={w}>{w}</option>)}
                        </select>
                      </label>
                    </div>
                    {form.weapon !== 'Any Weapon' && WEAPON_TEMPLATES[form.weapon] && (
                      <div className="template-section">
                        <span className="template-label">Quick Templates — {form.weapon}</span>
                        <div className="template-list">
                          {WEAPON_TEMPLATES[form.weapon].map(t => (
                            <button key={t.name} className="template-btn" onClick={() => loadTemplate(t)} title={t.notation} type="button">
                              {t.name}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    <label>Steps (one per line)
                      <textarea
                        rows={8}
                        value={form.steps}
                        onChange={e => setForm(f => ({ ...f, steps: e.target.value }))}
                        placeholder={"Press N\nWait 50ms\nPress Space\nWait 80ms\nPress N"}
                      />
                    </label>
                    <label>AI Prompt (optional)
                      <div className="ai-row">
                        <input
                          value={form.aiPrompt}
                          onChange={e => setForm(f => ({ ...f, aiPrompt: e.target.value }))}
                          placeholder="e.g. Katarina sword down-light into nair chase combo..."
                        />
                        <button className="btn btn-ai" onClick={askAI} disabled={aiLoading || !aiKey}>
                          {aiLoading ? '...' : '🤖 Generate'}
                        </button>
                      </div>
                      {aiResult && <div className="ai-result">{aiResult}</div>}
                    </label>
                  </div>
                ) : (
                  <div className="macro-detail">
                    <div className="detail-badges">
                      <span className="legend-badge">{selected?.legend || '—'}</span>
                      {selected?.weapon && selected.weapon !== 'Any Weapon' && <span className="weapon-badge">{selected.weapon}</span>}
                    </div>
                    {selected?.weapon && selected.weapon !== 'Any Weapon' && WEAPON_TEMPLATES[selected.weapon] && (
                      <div className="detail-notation">
                        <span className="template-label">Notation Reference</span>
                        <div className="template-list">
                          {WEAPON_TEMPLATES[selected.weapon].map(t => (
                            <span key={t.name} className="notation-chip" title={t.steps}>{t.notation}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    <div className="detail-row"><span className="detail-label">Steps</span></div>
                    <pre className="steps-preview">{selected?.steps}</pre>
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-icon">⚔️</div>
                <h2>Select a combo or create a new one</h2>
                <p>Use the sidebar to select a combo, or click <strong>+ New Combo</strong> to get started.</p>
                <div className="keybind-ref">
                  <div className="keybind-title">Default Keybinds</div>
                  <div className="keybind-grid">
                    <span className="key">N</span><span>Light Attack</span>
                    <span className="key">M</span><span>Heavy Attack</span>
                    <span className="key">Space</span><span>Jump</span>
                    <span className="key">S+N</span><span>Ground Pound</span>
                    <span className="key">S+M</span><span>Ground Slam</span>
                    <span className="key">WASD</span><span>Movement</span>
                  </div>
                </div>
                <button className="btn btn-new-lg" onClick={newMacro}>+ New Combo</button>
              </div>
            )}

            <div className="log-panel">
              <div className="log-header">📋 Run Log</div>
              <div className="log-body">
                {log.length === 0 ? <span className="log-empty">No activity yet.</span> : log.map((l, i) => <div key={i} className="log-line">{l}</div>)}
              </div>
            </div>
          </>
        )}

        {tab === 'ai' && (
          <div className="ai-panel">
            <h2>🤖 AI Combo Generator</h2>
            <p>Describe a Brawlhalla combo or technique in plain English, and AI will generate the macro steps for you.</p>
            <textarea
              rows={4}
              className="ai-prompt-box"
              value={form.aiPrompt}
              onChange={e => setForm(f => ({ ...f, aiPrompt: e.target.value }))}
              placeholder="e.g. Katarina sword side-light into jump cancel nair, then recover with dodge"
            />
            <button className="btn btn-ai-lg" onClick={askAI} disabled={aiLoading || !aiKey}>
              {aiLoading ? 'Generating...' : '🤖 Generate Macro Steps'}
            </button>
            {!aiKey && <p className="ai-warning">⚠️ Add your OpenAI API key in Settings to use AI generation.</p>}
            {aiResult && (
              <div className="ai-result-lg">
                <div className="ai-result-header">Generated Steps:</div>
                <pre>{aiResult}</pre>
                <button className="btn btn-save" onClick={() => { setTab('macros'); setIsEditing(true) }}>Use These Steps →</button>
              </div>
            )}
          </div>
        )}

        {tab === 'settings' && (
          <div className="settings-panel">
            <h2>⚙️ Settings</h2>

            {/* ── OpenAI ── */}
            <div className="settings-section">
              <div className="settings-section-title">🤖 AI Combo Generator</div>
              <label>OpenAI API Key
                <input
                  type="password"
                  value={aiKey}
                  onChange={e => setAiKey(e.target.value)}
                  placeholder="sk-..."
                />
              </label>
              <p className="settings-note">Used only for AI combo generation. Stored locally — never persisted to server.</p>
            </div>

            {/* ── Brawlhalla API ── */}
            <div className="settings-section">
              <div className="settings-section-title">📊 Brawlhalla Rankings API</div>
              <p className="settings-note" style={{ marginBottom: 10 }}>
                Get a free key at <span className="settings-link">dev.brawlhalla.com</span>. Used to pull live weapon trends from ranked data.
              </p>
              <label>BH API Key
                <input
                  type="password"
                  value={bhApiKey}
                  onChange={e => setBhApiKey(e.target.value)}
                  placeholder="e.g. abc123..."
                />
              </label>
            </div>

            {/* ── Twitch API ── */}
            <div className="settings-section">
              <div className="settings-section-title">🎮 Twitch Helix API</div>
              <p className="settings-note" style={{ marginBottom: 10 }}>
                Register an app at <span className="settings-link">dev.twitch.tv</span>. Used to detect live tournament viewership and adjust accuracy multipliers.
              </p>
              <label>Twitch Client ID
                <input
                  value={twitchClientId}
                  onChange={e => setTwitchClientId(e.target.value)}
                  placeholder="e.g. abc123xyz..."
                />
              </label>
              <label>Twitch Client Secret
                <input
                  type="password"
                  value={twitchClientSecret}
                  onChange={e => setTwitchClientSecret(e.target.value)}
                  placeholder="e.g. secret..."
                />
              </label>
            </div>

            {/* ── Extension OCR Region ── */}
            <div className="settings-section">
              <div className="settings-section-title">🔍 Twitch Extension OCR Region</div>
              <p className="settings-note" style={{ marginBottom: 10 }}>
                Screen region (pixels) where the Twitch extension overlay is rendered on your monitor.
              </p>
              <div className="settings-grid-4">
                <label>Top<input type="number" value={ocrRegion.top} onChange={e => setOcrRegion(r => ({ ...r, top: +e.target.value }))} className="bp-input" /></label>
                <label>Left<input type="number" value={ocrRegion.left} onChange={e => setOcrRegion(r => ({ ...r, left: +e.target.value }))} className="bp-input" /></label>
                <label>Width<input type="number" value={ocrRegion.width} onChange={e => setOcrRegion(r => ({ ...r, width: +e.target.value }))} className="bp-input" /></label>
                <label>Height<input type="number" value={ocrRegion.height} onChange={e => setOcrRegion(r => ({ ...r, height: +e.target.value }))} className="bp-input" /></label>
              </div>
            </div>

            {/* ── Save & Push ── */}
            <div className="settings-save-row">
              <button className="btn btn-save" onClick={saveAndPushSettings}>
                💾 Save & Push to Server
              </button>
              {settingsSaved && <span className="settings-saved-badge">✓ Saved</span>}
            </div>
            <p className="settings-note">
              API keys are stored in localStorage and never transmitted outside your machine except to the configured API endpoints.
            </p>
          </div>
        )}

        {tab === 'glossary' && (
          <div className="glossary-panel">
            <div className="glossary-header">
              <h2>📖 Move Glossary</h2>
              <p>Brawlhalla notation and strategic concepts from the COSMIC M.E.T.A guide.</p>
              <input
                className="glossary-search"
                placeholder="🔍 Search terms…"
                value={glossaryFilter}
                onChange={e => setGlossaryFilter(e.target.value)}
              />
            </div>
            <div className="glossary-grid">
              {GLOSSARY
                .filter(g => !glossaryFilter ||
                  g.term.toLowerCase().includes(glossaryFilter.toLowerCase()) ||
                  g.def.toLowerCase().includes(glossaryFilter.toLowerCase())
                )
                .map(g => (
                  <div key={g.term} className="glossary-card">
                    <span className="glossary-term">{g.term}</span>
                    <span className="glossary-def">{g.def}</span>
                  </div>
                ))
              }
            </div>
          </div>
        )}

        {tab === 'simulation' && <SimulationTab />}
        {tab === 'fight' && <FightTab />}
        {tab === 'capture' && <CaptureTab />}
        {tab === 'strategy' && <StrategyTab />}
        {tab === 'intel' && <IntelTab />}

        {tab === 'scout' && (
          <div className="scout-panel">
            {/* ── Header ── */}
            <div className="scout-header">
              <div>
                <h2>🧠 Scout AI</h2>
                <p>Record your opponent's moves live. The AI learns their patterns and tells you what's coming next.</p>
              </div>
              <div className="scout-header-actions">
                <span className="scout-count">{scout.total} moves recorded</span>
                <button
                  className={`btn ${coachingMode ? 'btn-coaching-on' : 'btn-coaching-off'}`}
                  onClick={toggleCoachingMode}
                  title="Shrinks to a tiny always-on-top bar so you can see it over your game"
                >
                  {coachingMode ? '🟢 Coaching ON' : '🎯 Coaching Mode'}
                </button>
                <button className="btn btn-delete" onClick={resetScout}>⟳ Reset</button>
              </div>
            </div>

            <div className="scout-body">
              {/* ── Left column: input + history ── */}
              <div className="scout-left">
                <div className="scout-input-section">
                  <div className="scout-section-title">🎮 Record Opponent Move</div>
                  <p className="scout-hint">Tap a button each time your opponent uses that move. Do it live during the match.</p>
                  {MOVE_GROUPS.map(g => (
                    <div key={g.label} className="scout-move-group">
                      <span className={`scout-group-label scout-group-${g.color}`}>{g.label}</span>
                      <div className="scout-move-btns">
                        {g.moves.map(move => (
                          <button
                            key={move}
                            className={`scout-move-btn scout-btn-${g.color}`}
                            onClick={() => recordMove(move)}
                          >
                            {move}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                {scout.history.length > 0 && (
                  <div className="scout-history">
                    <div className="scout-section-title">📜 Recent Moves (newest first)</div>
                    <div className="scout-history-list">
                      {scout.history.map((m, i) => (
                        <span key={i} className={`scout-history-pill scout-btn-${MOVE_GROUPS.find(g => (g.moves as readonly string[]).includes(m))?.color ?? 'tech'}`}>
                          {m}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* ── Right column: predictions + habits + patterns ── */}
              <div className="scout-right">
                {/* Predictions */}
                <div className="scout-card">
                  <div className="scout-section-title">🎯 Next Move Predictions</div>
                  {scout.total < 5 ? (
                    <p className="scout-empty">Record at least 5 moves to see predictions.</p>
                  ) : scout.predictions.length === 0 ? (
                    <p className="scout-empty">Not enough sequence data yet — keep recording.</p>
                  ) : (
                    scout.predictions.map((p, i) => (
                      <div key={p.move} className="scout-prediction">
                        <div className="scout-pred-top">
                          <span className={`scout-pred-rank rank-${i}`}>{['1ST', '2ND', '3RD'][i]}</span>
                          <span className="scout-pred-move">{p.move}</span>
                          <span className="scout-pred-conf">{p.confidence}%</span>
                        </div>
                        <div className="scout-conf-bar">
                          <div className="scout-conf-fill" style={{ width: `${p.confidence}%`, opacity: 1 - i * 0.2 }} />
                        </div>
                        <div className="scout-counters">
                          {p.counter.map(c => <span key={c} className="scout-counter-chip">{c}</span>)}
                        </div>
                      </div>
                    ))
                  )}
                </div>

                {/* Habit profile */}
                {scout.habits.length > 0 && (
                  <div className="scout-card">
                    <div className="scout-section-title">📊 Habit Profile</div>
                    {scout.habits.slice(0, 8).map(h => (
                      <div key={h.move} className="scout-habit-row">
                        <span className="scout-habit-move">{h.move}</span>
                        <div className="scout-habit-bar-bg">
                          <div className="scout-habit-bar-fill" style={{ width: `${h.pct}%` }} />
                        </div>
                        <span className="scout-habit-pct">{h.pct}%</span>
                        <span className="scout-habit-count">×{h.count}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Detected patterns */}
                {scout.patterns.length > 0 && (
                  <div className="scout-card">
                    <div className="scout-section-title">🔗 Detected Patterns</div>
                    <p className="scout-hint">Sequences your opponent repeats — anticipate the second move.</p>
                    {scout.patterns.map(p => (
                      <div key={`${p.from}-${p.to}`} className="scout-pattern-row">
                        <span className="scout-pattern-chain">
                          <span className="scout-pattern-from">{p.from}</span>
                          <span className="scout-pattern-arrow">→</span>
                          <span className="scout-pattern-to">{p.to}</span>
                        </span>
                        <span className="scout-pattern-meta">{p.pct}% of the time · ×{p.count}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
