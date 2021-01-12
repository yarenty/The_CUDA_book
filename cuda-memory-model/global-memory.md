# Global memory



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



  
Global memory has long latency, there are others:

* shared memory
* constant
* registers

![Memory types](../.gitbook/assets/mem_types.jpeg)



