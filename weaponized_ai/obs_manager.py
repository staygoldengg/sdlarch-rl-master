# obs_manager.py
"""
OBS Studio lifecycle management for AI screen capture.

Features:
  - detect_obs()           : find obs64.exe on this machine
  - is_obs_running()       : check via tasklist
  - launch_obs()           : start OBS with virtual camera flag
  - download_obs()         : fetch latest installer from GitHub releases (background thread)
  - install_obs()          : run silent NSIS install
  - ensure_obs()           : one-call: detect → install if missing → launch
  - get_status()           : dict with installed/running/progress
  - find_obs_camera_index(): enumerate cv2 cameras to locate OBS Virtual Camera

All heavy work runs in daemon threads; poll get_status() for progress.
"""

import os
import json
import subprocess
import threading
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# ── Known install paths ───────────────────────────────────────────────────────
_OBS_CANDIDATES = [
    r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
    r"C:\Program Files (x86)\obs-studio\bin\64bit\obs64.exe",
    r"C:\Program Files\OBS Studio\bin\64bit\obs64.exe",
    r"C:\Users\{user}\AppData\Local\obs-studio\bin\64bit\obs64.exe",
]

_GITHUB_API = "https://api.github.com/repos/obsproject/obs-studio/releases/latest"

# ── Shared install progress state ─────────────────────────────────────────────
_state = {
    "installed":       False,
    "install_path":    None,
    "running":         False,
    "obs_pid":         None,
    "camera_index":    None,
    # install progress
    "downloading":     False,
    "download_pct":    0,
    "installing":      False,
    "install_done":    False,
    "install_error":   None,
    "install_message": "",
}
_state_lock = threading.Lock()


def _update(**kw):
    with _state_lock:
        _state.update(kw)


# ── Detection ─────────────────────────────────────────────────────────────────
def find_obs() -> Optional[str]:
    """Return path to obs64.exe if found on this machine."""
    import winreg
    # Try registry first (most reliable)
    try:
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hive, r"SOFTWARE\OBS Studio") as key:
                    d, _ = winreg.QueryValueEx(key, "")
                    candidate = os.path.join(d, "bin", "64bit", "obs64.exe")
                    if os.path.exists(candidate):
                        return candidate
            except FileNotFoundError:
                pass
    except Exception:
        pass

    # Fall back to known paths
    user = os.environ.get("USERNAME", "")
    for p in _OBS_CANDIDATES:
        resolved = p.replace("{user}", user)
        if os.path.exists(resolved):
            return resolved

    # Search common program dirs
    for base in (r"C:\Program Files", r"C:\Program Files (x86)"):
        for root, dirs, files in os.walk(base):
            if "obs64.exe" in files:
                return os.path.join(root, "obs64.exe")
            # Don't recurse too deep
            if root.count(os.sep) - base.count(os.sep) > 3:
                del dirs[:]

    return None


def is_obs_running() -> bool:
    try:
        r = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq obs64.exe", "/NH"],
            capture_output=True, text=True, timeout=5
        )
        return "obs64.exe" in r.stdout
    except Exception:
        return False


# ── Installer download ────────────────────────────────────────────────────────
def _get_installer_url() -> str:
    """Fetch the latest OBS Windows x64 installer URL from GitHub releases."""
    try:
        req = urllib.request.Request(
            _GITHUB_API,
            headers={"User-Agent": "weaponized-ai/1.0", "Accept": "application/vnd.github.v3+json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        for asset in data.get("assets", []):
            name = asset.get("name", "")
            url  = asset.get("browser_download_url", "")
            # Prefer the Windows x64 installer .exe
            if name.endswith(".exe") and ("Windows" in name or "windows" in name):
                return url
        # Fallback: any .exe
        for asset in data.get("assets", []):
            if asset.get("name", "").endswith(".exe"):
                return asset["browser_download_url"]
    except Exception:
        pass
    # Hard-coded fallback
    return "https://github.com/obsproject/obs-studio/releases/download/31.0.3/OBS-Studio-31.0.3-Windows-Installer.exe"


def _do_download_install():
    """Background thread: download → install OBS."""
    _update(downloading=True, download_pct=0, install_message="Fetching installer URL…")
    try:
        url = _get_installer_url()
        _update(install_message=f"Downloading OBS from {url.split('/')[-1]}…")

        tmp = tempfile.mktemp(suffix=".exe", prefix="obs_installer_")

        def _hook(count, block, total):
            if total > 0:
                pct = min(99, count * block * 100 // total)
                _update(download_pct=pct, install_message=f"Downloading… {pct}%")

        urllib.request.urlretrieve(url, tmp, _hook)
        _update(downloading=False, download_pct=100,
                installing=True, install_message="Running OBS installer silently…")

        # NSIS silent install
        result = subprocess.run([tmp, "/S"], timeout=300)
        try:
            os.unlink(tmp)
        except Exception:
            pass

        path = find_obs()
        if path or result.returncode == 0:
            _update(installing=False, install_done=True,
                    install_error=None, install_message="OBS installed successfully.",
                    installed=True, install_path=path)
        else:
            _update(installing=False, install_done=True,
                    install_error=f"Installer exited {result.returncode}",
                    install_message="Install may have failed — check manually.")
    except Exception as e:
        _update(downloading=False, installing=False,
                install_error=str(e), install_message=f"Error: {e}")


def download_and_install_obs():
    """Start OBS download+install in a background thread. Poll get_status() for progress."""
    if _state.get("downloading") or _state.get("installing"):
        return  # already in progress
    t = threading.Thread(target=_do_download_install, daemon=True)
    t.start()


# ── Launch ────────────────────────────────────────────────────────────────────
def launch_obs(start_virtual_cam: bool = True) -> dict:
    """Launch OBS Studio (with optional --startvirtualcam flag)."""
    obs_path = find_obs()
    if not obs_path:
        return {"success": False, "message": "OBS not found. Install it first via /obs/install."}
    if is_obs_running():
        _update(running=True)
        return {"success": True, "message": "OBS is already running."}

    args = [obs_path, "--minimize-to-tray"]
    if start_virtual_cam:
        args.append("--startvirtualcam")
    try:
        proc = subprocess.Popen(args)
        _update(running=True, obs_pid=proc.pid)
        return {"success": True, "message": "OBS launched.", "pid": proc.pid}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ── OBS Camera Index Discovery ────────────────────────────────────────────────
def find_obs_camera_index(max_index: int = 6) -> int:
    """
    Try cv2.VideoCapture(0..max_index) and return the index for
    a device whose name contains 'OBS', or 1 as default.
    Falls back to 1 if cv2 is unavailable.
    """
    try:
        import cv2
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                name = cap.getBackendName()
                cap.release()
                if "OBS" in name.upper():
                    _update(camera_index=i)
                    return i
        _update(camera_index=1)
        return 1
    except Exception:
        return 1


# ── Brawlhalla Window Detection ───────────────────────────────────────────────
def find_brawlhalla_window() -> Optional[dict]:
    """
    Find the Brawlhalla game window using win32 APIs.
    Returns {"left": x, "top": y, "width": w, "height": h} or None.
    Works even when the game is borderless/fullscreen.
    """
    try:
        import ctypes
        import ctypes.wintypes as wt

        FindWindow  = ctypes.windll.user32.FindWindowW
        GetWindowRect = ctypes.windll.user32.GetWindowRect
        IsWindowVisible = ctypes.windll.user32.IsWindowVisible

        # Try exact class or title
        hwnd = FindWindow(None, "Brawlhalla")
        if not hwnd:
            # Enumerate all windows and find by partial title
            EnumWindows = ctypes.windll.user32.EnumWindows
            GetWindowTextW = ctypes.windll.user32.GetWindowTextW
            EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))

            found = ctypes.c_void_p(None)

            def _cb(h, _):
                buf = ctypes.create_unicode_buffer(256)
                GetWindowTextW(h, buf, 256)
                title = buf.value
                if "brawlhalla" in title.lower() and IsWindowVisible(h):
                    found.value = h
                    return False  # stop enumeration
                return True

            EnumWindows(EnumWindowsProc(_cb), 0)
            hwnd = found.value

        if not hwnd:
            return None

        rect = wt.RECT()
        GetWindowRect(hwnd, ctypes.byref(rect))
        w = rect.right  - rect.left
        h = rect.bottom - rect.top
        if w < 100 or h < 100:
            return None
        return {"left": rect.left, "top": rect.top, "width": w, "height": h}
    except Exception:
        return None


# ── One-shot helper ───────────────────────────────────────────────────────────
def ensure_obs(auto_install: bool = True, auto_launch: bool = True) -> dict:
    """
    Complete flow:
      1. Check if OBS is installed
      2. If not and auto_install=True → start background download+install
      3. If installed and auto_launch=True → launch it
    Returns current status dict.
    """
    obs_path = find_obs()
    _update(installed=obs_path is not None, install_path=obs_path,
            running=is_obs_running())

    if not obs_path and auto_install:
        download_and_install_obs()
        return get_status()

    if obs_path and auto_launch and not is_obs_running():
        launch_obs()

    return get_status()


# ── Status ────────────────────────────────────────────────────────────────────
def get_status() -> dict:
    obs_path = find_obs()
    with _state_lock:
        s = dict(_state)
    s["installed"] = obs_path is not None
    s["install_path"] = obs_path
    s["running"] = is_obs_running()
    return s
