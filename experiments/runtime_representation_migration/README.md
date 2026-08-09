# Multimodal runtime representation migration

This audit replaces the runtime’s event, controller-state, and intention
space identifiers while keeping the single controller boundary and two-stream
event structure unchanged. Paired held-out event windows are checked for
intention, execution, and continuation-state retention. A candidate with a
changed controller parameter is then required to fail the same verifier.

The audit promotes a copy-on-write compatibility gate across the
frontend/controller runtime. It does not claim that arbitrary representation
drift can be aligned without learned data, and it does not test external
memory migration.
