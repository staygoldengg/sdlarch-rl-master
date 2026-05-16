# striker_server.spec
# -------------------
# PyInstaller spec file — builds the Striker AI backend server.
#
# Produces:  dist/striker-server/striker-server.exe
#            + all bundled Python, torch, fastapi, uvicorn libs in the same dir
#
# Usage (from project root with .venv active):
#   pyinstaller striker_server.spec
#
# The --onedir (default) layout keeps dll files alongside the exe so
# PyTorch CUDA extensions and ctypes DLLs load correctly at runtime.
# The build_installer.ps1 script calls this automatically.

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

PROJECT_ROOT = Path(SPECPATH)  # directory containing this .spec file

# ── Hidden imports ─────────────────────────────────────────────────────────────
hidden = []

# uvicorn — fully dynamic import of protocols/loops
hidden += [
    "uvicorn.main",
    "uvicorn.config",
    "uvicorn.server",
    "uvicorn.supervisors",
    "uvicorn.logging",
    "uvicorn.middleware.proxy_headers",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.lifespan.off",
    "uvicorn.lifespan.on",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.loops.uvloop",
    "uvicorn._subprocess",
]

# fastapi / starlette
hidden += collect_submodules("fastapi")
hidden += collect_submodules("starlette")
hidden += collect_submodules("pydantic")
hidden += collect_submodules("pydantic_core")
hidden += collect_submodules("anyio")
hidden += collect_submodules("h11")

# weaponized_ai — the whole package
hidden += collect_submodules("weaponized_ai")

# torch — collect everything (large but necessary)
hidden += collect_submodules("torch")
hidden += collect_submodules("numpy")

# ctypes / windows (for brawlhalla_memory.py and input_controller.py)
hidden += ["ctypes.wintypes", "ctypes._endian"]

# ── Data files ─────────────────────────────────────────────────────────────────
datas = []

# numpy / torch data
datas += collect_data_files("numpy")
datas += collect_data_files("torch", includes=["**/*.dll", "**/*.pyd", "**/*.so"])

# pydantic schemas
datas += collect_data_files("pydantic")

# ── Analysis ───────────────────────────────────────────────────────────────────
a = Analysis(
    [str(PROJECT_ROOT / "server_entry.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # strip test / doc modules we don't need at runtime
        "matplotlib", "IPython", "jupyter", "sphinx",
        "tkinter", "_tkinter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,      # onedir mode — dlls stay in the same folder
    name="striker-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                   # compress binaries where possible
    console=True,               # keep console so uvicorn logs are visible
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "src-tauri" / "icons" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="striker-server",      # output folder: dist/striker-server/
)
