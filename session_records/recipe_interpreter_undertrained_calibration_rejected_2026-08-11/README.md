# Learned recipe-interpreter calibration — rejected as undertrained

This is an in-repository calibration run for the expressibility audit. It
uses only random programs over abstract slots and no task-family labels. It is
not a replication of the exported session's learned-interpreter numbers.

At 500 optimizer updates and matched batch size `64`, the fixed Transformer
interpreter remained undertrained on the old basis. The extension learned the
new paired effect, but old-basis execution was not yet stable enough for a
cost or transfer claim:

| seed | arm | base length-2 | base length-4 | paired target |
| ---: | --- | ---: | ---: | ---: |
| 70421 | atomic | 0.4961 | 0.2500 | 0.0000 |
| 70421 | parallel | 0.5830 | 0.3496 | 0.8506 |
| 70422 | atomic | 0.7373 | 0.5703 | 0.0000 |
| 70422 | parallel | 0.5264 | 0.2822 | 0.9805 |

The positive paired-target scores are a useful sanity check that the new
encoding reaches the interpreter. They do not outweigh the low and unstable
old-basis retention curve. The correct action is to increase only the
interpreter-training rung and rerun the same paired seeds, not to claim that
the richer basis is cheaper or better.

The symbolic boundary remains exact and independent: the atomic basis rejects
the paired effect, while the generic parallel extension represents it.
