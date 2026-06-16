from dataclasses import dataclass

import draccus


@dataclass
class ControllerConfig(draccus.ChoiceRegistry):
    """Base discriminated union for RTDE controller backends.

    Subclasses hold only the parameters specific to their low-level send
    path (forceMode or directTorque). Shared impedance parameters (kp, kd,
    wrench_limits, compliance settings) live in URV2Config.
    """
    pass
