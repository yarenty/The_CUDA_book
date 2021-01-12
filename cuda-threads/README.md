# Threads

### What is a CUDA Thread

CUDA threads are very lightweight processes - much lighter than usual CPU threads

It is very-low cost to:

* create them \(at kernel launch\)
* schedule them
* destroy them \(at kernel end\)

They are building blocks of any CUDA computation.

Organised per sets called "blocks"

Blocks are further organised on "grids"















