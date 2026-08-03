# Generic workspace-context successor (pre-registration, 2026-08-03)

The source-session diagnostics showed that richer learned working-memory
context can make successor-slot updates more effective, but earlier versions
opened on blank evidence. This arm gives a fresh zero-output slot only the
generic workspace content, workspace usage, and event-age tensors already
maintained by the controller. It does not expose span, operation, correct
actions, or verifier metadata.

The parent is the accepted missing-evidence frontier. Training uses 128 fresh
span-11 mixed lifetimes, 128 protected span-10 and span-9 lifetimes, and 128
protected blank span-11 lifetimes; 32 epochs, batch 512, learning rate
0.0005, binary complement/critic losses, and 0.1 gate/logit protection.

Promotion requires positive paired acquisition, newest-slot causality, old
retention, blank, and full-reset gates. A high raw score with a blank shortcut
is a rejection.
