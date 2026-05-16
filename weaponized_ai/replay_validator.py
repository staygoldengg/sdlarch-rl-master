# replay_validator.py
"""
ReplayDesyncValidator — post-match sanity check for simulation accuracy.

Compares the events produced by the deterministic simulation engine against
binary ground-truth events read from the game process memory.  If simulated
KO events drift more than max_allowed_drift_frames from real ones, the
internal model is considered desynced and must be corrected.
"""


class ReplayDesyncValidator:
    def __init__(self, max_allowed_drift_frames: int = 4):
        self.max_allowed_drift_frames = max_allowed_drift_frames

    def verify_simulation_integrity(
        self,
        simulated_trajectory: list[dict],
        binary_ground_truth_events: list[dict],
    ) -> bool:
        """
        Checks that simulated KO events align with ground-truth binary events.

        Args:
            simulated_trajectory: List of step dicts, each with an optional
                                  bool field 'is_ko'.
            binary_ground_truth_events: List of event dicts, each with a
                                        string field 'event_type'.

        Returns:
            True if the simulation is in sync; False if a drift is detected.
        """
        truth_kos = [
            e for e in binary_ground_truth_events
            if e.get("event_type") == "KO_EVENT"
        ]
        sim_kos = [
            s for s in simulated_trajectory
            if s.get("is_ko") is True
        ]

        if len(truth_kos) != len(sim_kos):
            print(
                f"[VALIDATOR] KO count mismatch: "
                f"truth={len(truth_kos)}, simulated={len(sim_kos)}"
            )
            return False

        for i, (truth, sim) in enumerate(zip(truth_kos, sim_kos)):
            truth_frame = truth.get("frame", 0)
            sim_frame   = sim.get("frame", 0)
            drift = abs(truth_frame - sim_frame)
            if drift > self.max_allowed_drift_frames:
                print(
                    f"[VALIDATOR] KO {i} frame drift of {drift} "
                    f"exceeds limit of {self.max_allowed_drift_frames} — desync detected."
                )
                return False

        return True
