---
description: Java - CUDA - JNI - intro
---

# Porting Java code to CUDA

What are we looking for? We want to run software faster!



Speeding software up - where to make changes:

* algorithms
* compilers
* execution environments
* hardware

Speeding algorithms:

* small fixes:
  * remove dead code
  * avoid context switch
  * trade memory usage for less processing \(avoid recomputing values\)
  * use faster library implementations
* large fixes:
  * rewrite entire algorithms 
  * rethink the software architecture
  * parallelise algorithms !

Speeding up the compiler:

* enable optimisation switches
* use compiler directives
* specify hardware architecture
* change the compiler \( to cuda ;-\) \)

Speeding up the Execution Environment

* remove unnecessary overheads
* avoid software virtualisation
* reduce time taken to initialise and finalise
* change the execution environment \(cuda\)!

Speeding up the hardware

* processors with higher clock frequencies
* make use of faster memory
* faster buses and ports
* processors with larger L1 and L2 caches
* change the hardware architecture





