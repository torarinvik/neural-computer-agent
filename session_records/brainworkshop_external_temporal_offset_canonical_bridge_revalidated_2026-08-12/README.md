# Canonical external temporal offset bridge revalidation

The outcome-only temporal-offset growth experiment was rerun through
`AmodalControllerRuntime.step_streams_with_external_history()`. The bridge
reads the selected prior record before appending the current event, so the
public logical lag is translated to `lag - 1` only at the external history
boundary. The controller and learned event encoder remain frozen.

Seed `17` at `512` updates passed all `10/10` gates: old and new mastery,
retention, unchanged old file, correct offset preference, wrong-offset and
missing-history rejection, shuffled-outcome rejection, frozen controller and
frontend, and zero replay. Accounting was `311,296` unique verifier bits,
`147,456` control bits, `32,768` logical lifetimes, `1,024` optimizer updates,
and zero replayed examples.

This revalidates the canonical transport and scalar-credit offset result. It
remains one bounded external relative-address capability, not general memory
search, compression, unrestricted growth, or general continual learning.
