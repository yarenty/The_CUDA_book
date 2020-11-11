# Device occupancy



Assigned blocks per SM depends on kernel resource usage



| Description \(per SM\) | Limit \(compute 1.3\) | Limit \(compute 2.0\) |
| :--- | :--- | :--- |
| Max threads | 1024 | 1538 |
| Max thread blocks | 8 | 8 |
| Available shared memory | 16384 | 49152 |
| Available 32-bit registers | 16384 | 32768 |

If limits are exceeded, number of blocks per SM is reduced as necessary

Greater occupancy is desirable because it helps to hide latency



### Programmer view of register file

There are 32768 registers in each SM in Fermi

* this is an implementation decision, not part of CUDA
* registers are dynamically partitioned across all blocks assigned to the SM
* once assigned to a block, the register is NOT accessible by threads in other blocks
* each thread in the same block only access registers assigned to itself

\[img\]



### Matrix Multiplication example

If each block has 512 threads and each thread uses 16 registers, how many thread can run on each SM?

* each block requires 16\*512 = 8192 registers
* 32768 = 4 \* 8192
* 4 blocks can run on an SM as far as registers are concerned

How about if each thread increases the use of registers by 1?

* each block requires now 17 \* to 512 - 8704 registers
* 32768 = 3 \* 8704 + 6656
* only 3 blocks can run on an SM, 25% reduction of parallelism!



### Dynamic partitioning

Dynamic partitioning gives more flexibility to compilers/programmers:

* one can run a smaller number of threads that require many registers each or a large number of threads that require few registers each - this allows for finer grain threading than traditional CPU threading models
* the compiler can trade-off between instruction-level parallelism and thread-level parallelism - using more registers might improve the kernel performance, and overcome the thread scheduling limitation



### NVCC - usefull flag

--ptxas-options=v

This gives information about used registers, shared memory per block \(user and system\), and constant memory.

\[add output of code example!!\]

 

### CUDA occupancy calculator

Can help find the sweet spot for the block size

Can highlight what are the limiting factors for occupancy..:

* register usage
* shared memory
* block size

May have to experiment with block size to see what works best

Occupancy isn't the most important thing







