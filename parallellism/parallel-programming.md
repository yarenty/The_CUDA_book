# Parallel programming

Decompose a problem into subproblems

Structure the algorithm so that concurrency can be exploited

Implement the algorithm in a parallel programing environment



Units of processing

The statements flow through what are called "units of processing":

* store data they process
* may communicate with each other
* are scheduled by the OS for execution

Processes

Haviest unit of processing

Has its own address space, i.e. does not share memory with other processes

Independent of Other processes and interact with them via system-provided interprocess communicaiton mechanisms

Context switching between processes is heavy







