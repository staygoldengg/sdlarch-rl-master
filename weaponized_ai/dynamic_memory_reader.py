# dynamic_memory_reader.py
"""
DynamicBrawlhallaReader — live pointer-chain memory reader.

Attaches to the running Brawlhalla.exe process via pymem and walks multi-level
pointer chains to reach dynamically allocated player state objects.

The returned state vector uses a layout that maps directly into the feature
indices expected by the SHM bridge:
  [0] player X position
  [1] player Y position
  [4] damage percentage (0-300)
  All other slots default to 0.0.

Requires: pip install pymem
"""

import numpy as np

try:
    import pymem
    import pymem.process
except ImportError:
    raise ImportError(
        "pymem is required for DynamicBrawlhallaReader. "
        "Install it with: pip install pymem"
    )


class DynamicBrawlhallaReader:
    def __init__(self, process_name: str = "Brawlhalla.exe"):
        """
        Attaches to an already-running Brawlhalla process.

        Raises:
            pymem.exception.ProcessNotFound if the game is not running.
        """
        self.pm = pymem.Pymem(process_name)
        print(f"[MEM] Attached to {process_name} (PID {self.pm.process_id})")

    def follow_pointer_chain(self, base_address: int, offsets: list[int]) -> int:
        """
        Walks a multi-level pointer chain.

        Args:
            base_address: The starting module base or absolute address.
            offsets:      Sequential dereference offsets.

        Returns:
            Final resolved address as an integer.
        """
        addr = self.pm.read_longlong(base_address)
        for offset in offsets:
            addr = self.pm.read_longlong(addr + offset)
        return addr

    def read_live_state(
        self,
        p1_base_offset: int,
        p1_chain: list[int],
    ) -> np.ndarray:
        """
        Returns a 64-float32 state vector for Player 1.

        Args:
            p1_base_offset: Module-relative starting address for P1 pointer chain.
            p1_chain:       List of pointer dereference offsets after the base.

        Returns:
            float32 ndarray of shape (64,) — other slots are 0.0.
        """
        state = np.zeros(64, dtype=np.float32)
        try:
            brawlhalla_module = pymem.process.module_from_name(
                self.pm.process_handle, "Brawlhalla.exe"
            )
            base = brawlhalla_module.lpBaseOfDll
            p1_object_addr = self.follow_pointer_chain(base + p1_base_offset, p1_chain)

            state[0] = self.pm.read_float(p1_object_addr + 0x10)  # X position
            state[1] = self.pm.read_float(p1_object_addr + 0x14)  # Y position
            state[4] = self.pm.read_float(p1_object_addr + 0x20)  # damage %
        except Exception as exc:
            print(f"[MEM] Read failed — partial state returned. Error: {exc}")

        return state
