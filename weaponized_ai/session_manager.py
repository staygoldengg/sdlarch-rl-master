# session_manager.py
"""
AutonomousSessionManager — context manager that wraps the full training
session lifecycle.

Usage:
    with AutonomousSessionManager(translator, watchdog) as session:
        # run training loop here
        pass

On exit (clean or crash), emergency_flush() is called unconditionally so the
keyboard is always returned to a usable state.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from weaponized_ai.input_controller import ActionTranslationEngine


class AutonomousSessionManager:
    def __init__(self, translator: "ActionTranslationEngine", watchdog_reader: Any):
        self.translator = translator
        self.watchdog = watchdog_reader

    def __enter__(self) -> "AutonomousSessionManager":
        print("[SESSION] Bootstrapping secure runtime workspace wrapper.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Catches all loop terminations to safely close active drivers.
        Always releases every held key before returning.
        """
        print("[SESSION] Terminating training context loop.")

        # Unconditional safety override — keyboard always returns to normal
        self.translator.emergency_flush()

        if exc_type is not None:
            print(f"[CRITICAL ERROR DETECTED]: {exc_type.__name__} - {exc_val}")
            # Return False to let the exception propagate so the caller can log / restart.
            return False

        return True
