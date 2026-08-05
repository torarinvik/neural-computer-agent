# Prior-only consumer undo-complement probe — rejected

The producer→consumer ABI was first tested on span 10, where the verifier
target was already solved by the frozen parent. The consumer read was causal,
but composition did not improve the parent: composed `69.53%` versus parent
`70.00%`. The run was not promoted.

This failure motivated the harder sequence-level global-parity task, which
later produced the promoted result in the neighboring record.
