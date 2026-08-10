# The sparsity fixes, and a wrong baseline exposed (F103)

F102 prescribed class-balanced loss, outcome-seeking collection, and a value
target. All three built.

| arm | F102 baseline | with all three fixes |
| --- | ---: | ---: |
| trained variants | -0.0045 (3.0/12) | -0.0006 (4.5/12) |
| held-out variants | -0.0024 (4.0/12) | +0.0007 (5.5/12) |
| entry withheld | -0.0036 | +0.0013 |
| stranger entry | -0.0033 | +0.0007 |
| untrained control | +0.0071 (10/12) | +0.0075 (10/12) |

Label density: outcome-seeking raised the food class 3.54% -> 12.18% (3.4x);
the value target alone moved it 3.54% -> 3.72%. Seeking mattered, horizon
barely.

## Still wrong

1. The untrained control WINS: +0.0075 vs +0.0057 same-seed, 10/12 vs 8/12.
2. The entry is decoration: correct +0.0007, withheld +0.0013, stranger +0.0007
   — gap exactly zero.

## The real finding: the floor was the wrong control

Beam search over a random-but-fixed value function produces PERSISTENT
DIRECTIONAL motion instead of jitter. In a grid with scattered food, persistent
motion covers more ground than a random walk and collects more by accident.

So "beats a random-action floor" is satisfied for free by anything that moves
consistently — and every games number in F100-F103 was scored against it. The
correct control is SEARCH WITH AN UNTRAINED MODEL, which isolates what learning
contributes from what search contributes. Against that control this system has
never won.

## A bug in the fix, caught by measuring

The first value-target implementation accumulated the discounted return to the
END of the sequence regardless of `--horizon`, so the flag only trimmed trailing
rows and every label was a full Monte-Carlo return. Caught by checking label
density across settings, not by reading the code.
