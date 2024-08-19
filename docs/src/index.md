# BcdiTrad.jl Documentation

## About

Bragg Coherent Diffraction Imaging (BCDI) Trad (Traditional) implements projection-based phase retrieval algorithms.

While this package is marked as BCDI specific, the methods are more general and can be used in many phase retrieval problems. In the future, this package may be incorporated into a more general phase retrieval core package.

Currently, this entire package must be run with access to GPUs. This may change in the future (especially if Issues requesting it are opened), but for our research group, using GPUs is a necessity.

## Installation

BcdiTrad.jl is registered in the Julia general registry and can be installed by running in the REPL package manager (```]```):

```
add BcdiTrad
```
