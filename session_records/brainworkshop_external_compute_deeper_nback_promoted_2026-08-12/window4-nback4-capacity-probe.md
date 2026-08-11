# Four-event n-back-4 capacity probe

This is a negative representation audit, not a promoted capability claim.

Configuration: direct isolated external file, frozen controller and event
encoder, 192 updates, batch size 32, 14 rendered steps, attempted-outcome BCE
credit, entropy weight 0.01, learning rate 0.003, and four fresh evaluation
lifetimes. No route learning or replay was used.

| Seed | Final training accuracy | Fresh evaluation accuracies |
| ---: | ---: | --- |
| 17 | 0.6969 | 0.7781, 0.7531, 0.7594, 0.7250 |
| 18 | 0.7250 | 0.7531, 0.7594, 0.7250, 0.7906 |

The corresponding n-back-3 probes reached 1.0000 on both seeds with the same
four-event window. Increasing only the generic event-window parameter to five
made n-back-4 reach 1.0000 on every fresh lifetime on both seeds. This
supports an information-capacity diagnosis: current plus the previous three
events cannot reliably expose a lag-four comparison, while a five-event
window can.
