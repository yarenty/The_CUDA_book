# Parallel programming

Decompose a problem into subproblems

Structure the algorithm so that concurrency can be exploited

Implement the algorithm in a parallel programing environment



### Definitions

#### Units of processing

The statements flow through what are called "units of processing":

* store data they process
* may communicate with each other
* are scheduled by the OS for execution

#### Processes

* Haviest unit of processing
* Has its own address space, i.e. does not share memory with other processes
* Independent of Other processes and interact with them via system-provided interprocess communicaiton mechanisms
* Context switching between processes is heavy

#### Threads

* Lighter unit of processing
* Shares its address space with other threads, i.e. may have shared portions of memory
* Exist within a process and interacts with other threads via shared memory
* Context swiching between threads is lighter

#### Communication modes

* Shared memory - all processors have common portions of memory where they read and write to
* Message passing - processors send data to other processors, which may or may not "block", sending/waiting for it



* Units of processing and communication models:
  * processors commonly employ message passing
  * threads commonly employ shared memory
* Communication models are models:

  * may reflect the hardware/software underneath
  * can be combined and used together

Shared memory is easier to work with

* writing to memory can be seen as an asynchronous broadcast to all processors
* no need to communicate explicitly as it is done in message passing

Shared memory is double-edged sword:

* writes to memory often need to be performed atomically, which may block other processors from doing useful work
* may issue too many memory transfers, causing a bottleneck in the buses due to memory bandwidth
* 


  
 





