# CUDA Memory Model



| Memory | Location, on/off chip | Cached | Access | Scope | Lifetime | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Register | on | n/a | R/W | 1 thread | thread | no latency, no sharing, TB/s |
| Local | off | &gt;=2.0 | R/W | 1 thread | thread |  |
| Shared | on | n/a | R/W | all threads in block | block | i.e: 64KB per block |
| Global | off | &gt;=2.0 | R/W | all threads + host | host allocation | GBs, cudamemcpy, cudamalloc |
| Constant | off | yes! | R | all threads + host | host allocation | 1500GB/s |
| Texture | off | yes | R  | all threads + host | host allocation |  |



CUDA memory spaces & scopes

* global
* local \(per-thread global memory\)
* shared
* constant
* registers



Global memory

Large - depends on card ie: 

* C1060 - 4GB
* C2050 - 3GB
* C2070 - 6GB

Has long latency: ~200 cycles

Can only be allocated/freed by host

Main way host can pass data to/from device/

Allocate, copy, free:

```c
//allocate
cudaError_t cudaMalloc(void **devPtr,size_t size);

//copy
cudaError_t cudaMemcpy(void *dst, const void *src,size_t count, enum cudaMemcpyKind kind);
//kind: type of copy: H->D, D->H, D->H, D->D

//setting data on device
cudaError_t cudaMemset(void *devPtr,int value, size_t count);

cudaError_t cudaFree(void *devPtr);
```









