# Overview

Similar to pynx and others?, BcdiTrad implements projection-based algorithms in operator-style format. This means that the multiplication (*) and power (^) operators are used to apply operators to some current state. This may look like the following:

```
er = BcdiTrad.ER()
hio = BcdiTrad.HIO(0.9)
state = BcdiTrad.State(intensities, trues(size(intensities)))

(er * hio^20)^5 * state
```

This short script applies 20 HIO iterations and one ER iterations for a total of 5 times. This makes it easy to implement very complex recipes for phase retrieval algorithms.

# API

```@docs
State(intenities, recSupport)
ER()
HIO(beta)
Shrink(threshold, sigma, state)
Center(state)
```
