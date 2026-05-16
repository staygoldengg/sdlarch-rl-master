# process_utils.py
"""
System-level process utilities.

HeadlessExecutionHost
  Spawns child Python processes without allocating a visible console window,
  keeping the taskbar clean during background submodule execution.

HighPriorityExecutionShield
  Elevates the current process to ABOVE_NORMAL priority via psutil to
  prevent background OS scheduling starvation when Brawlhalla is running.
  Deliberately avoids REALTIME_PRIORITY_CLASS which can lock OS input handling.
"""

import subprocess
import sys
import os

# Win32 flag — suppresses the CMD window that Popen would otherwise create
CREATE_NO_WINDOW = 0x08000000


class HeadlessExecutionHost:
    @staticmethod
    def spawn_silent_submodule(
        script_path: str,
        args: list[str] | None = None,
    ) -> subprocess.Popen:
        """
        Launches a Python script without a visible console window.

        Args:
            script_path: Absolute or relative path to the .py script.
            args:        Optional list of CLI arguments to pass.

        Returns:
            The running Popen object (stdout/stderr available as pipes).
        """
        cmd = [sys.executable, script_path] + (args or [])
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            creationflags=CREATE_NO_WINDOW,
            close_fds=True,
        )


class HighPriorityExecutionShield:
    @staticmethod
    def claim_cpu_dominance():
        """
        Elevates the current process to ABOVE_NORMAL priority class.

        Requires: pip install psutil
        Only effective on Windows.
        """
        if sys.platform != "win32":
            print("[SHIELD] Priority elevation only available on Windows — skipped.")
            return

        try:
            import psutil
        except ImportError:
            print("[SHIELD] psutil not installed — priority elevation skipped.")
            return

        p = psutil.Process(os.getpid())
        p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        print(
            "[SHIELD] Kernel scheduling elevated to ABOVE_NORMAL. "
            "Background throttling disabled."
        )
