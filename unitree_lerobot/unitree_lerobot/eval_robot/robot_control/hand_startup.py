"""Initialize command buffers from measured hand state before starting publishers."""
from contextlib import nullcontext
import numpy as np


def initialize_hand_targets(ready, left_state, right_state, left_command, right_command,
                            lock=None, state_out=None, action_out=None, timeout=10.0):
    if not ready.wait(timeout):
        raise TimeoutError("Timed out waiting for valid hand states")
    def read(value):
        with value.get_lock():
            return [value.value] if hasattr(value, "value") else list(value[:])
    left, right = read(left_state), read(right_state)
    values = np.asarray(left + right, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Hand states must be finite before starting command publication")
    for command, positions in ((left_command, left), (right_command, right)):
        with command.get_lock():
            if hasattr(command, "value"):
                command.value = positions[0]
            else:
                command[:] = positions
    with lock if lock is not None else nullcontext():
        if state_out is not None:
            state_out[:] = values
        if action_out is not None:
            action_out[:] = values
