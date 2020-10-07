# The challenges in parallel programming

### A price to pay

Esploiting parallelism comes at a cost:

* higher complexity and more difficult to program
* new issues are related overheads: synchronisation, communication, data sharing
* ponderations that becme pertinent: load balancing, granularity, salability



### Parallelism is complex

Parallel software allow multiple instruction streams to execute at the same time and data to flow between them.

Identifying and managing the dependencies between tasks is tough work, careful study: the order of task execution man not change some results, but may change others

The costs are higher for every aspect in software development life cycle:

* desing
* implementation
* veryfication
* validation
* debugging
* tuning
* deployment
* maintenance
* documentation

### Parallel overhead

Amount of time required to coordinate parallel tasks, instead of doing usefull work:

* task startup time
* synchronisation
* data communication
* software overhead - compilers, libraries, tools, OS
* task termination time



### Execution bottlenecks

Innhibitors to parallelism: cause other processors to stall or to remain idle

May or may not be explicit:

* explicit bottleneckes - parallel communication, sychronisation points
* implicit bottlenecks - I/O operations, serial libraries, task scheduling

### Load balancing

Refers to distribution the work among all tasks evenly, so that tasks are kept busy all the time

Tries to minimise task idle times

In case several tasks are spawned in parallel and they need to synchronise, the slowest task will mandate the overall time taken by the algorithm.



### Granularity

Parallel tasks are composed of periodas of computation and periods of communication

A grain is period of computation performed in a parallel task

Periods of computation are typically separated form periods of communication by synchronisation bariers



#### Fine-grain parallelism

RElatively small amount of computational work done between communication events:

* facilitates load balancing
* implies high communication overhead and less opportunity for performance enhancement
* if granularity is too fine, the parallel overhead may be larger that the computation intself



#### Coarse-grain parallelism

Releatively large amount of computational work done between communication events:

* harder to balance the load efficiently
* implies more opportunity for performence enhancement
* very low parallel overhead



#### Choosing the granularity level

Depends on the algorithm and the hardware where it runs

The parallel overhead may be high compared to the execution speed, so it has to be assessed to check if fine granularity is adventageous

Fine-grain paralelism may help reduce overheads due to load imbalance



### Scalability

The ability of algorithms to handle large amounts of work, without much performance penalty:

* scalabel algorithms remain roughly efficient even when applied to very large scenarions \(large dataset, large number of processors\)
* parallel overheads and bottlenecks contribute to poor scalability of an algorithm
* 






