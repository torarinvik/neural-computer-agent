# Stable controller value path — reward-shuffled null control

Status: rejected control; no learned retention signal.

At the original `704/64` budget, reward shuffling left the system at chance:
intact `0.490`, target-first `0.474`, target-last `0.479`, and mastered-parent
retention `0.465`. The run did not pass parent stability or any retention gate.

This control supports attributing the accepted three-slot results to verifier
outcomes rather than the architecture alone.
