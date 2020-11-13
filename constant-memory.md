# Constant memory

Constant memory is very fast but it had 2 downsides:

* as its name says is constant - only for read
* you can have 64kB

The CUDA language makes available another kind of memory known as constant memory. As the name may indicate, we use constant memory for data that will not change over the course of kernel execution.

There is a total of 64 KB constant memory on a device. The constant memory space is cached. As a result, a read from constant memory costs one memory read from device memory only on a cache miss; otherwise, it just costs one read from the constant cache.



```c
//declare constant memory
__constant__ float cangle[360];

__global__ void test_kernel(float* darray) {
    int index;
    //calculate each thread global index
    index = blockIdx.x * blockDim.x + threadIdx.x;
    
    #pragma unroll 10
    for(int loop=0;loop<360;loop++)
        darray[index]= darray [index] + cangle [loop] ;
    return;
}
```

When it comes to handling constant memory, NVIDIA hardware can broadcast a single memory read to each half-warp. A half-warp—not nearly as creatively named as a warp—is a group of 16 threads: half of a 32-thread warp. If every thread in a half-warp requests data from the same address in constant memory, your GPU will generate only a single read request and subsequently broadcast the data to every thread. If you are reading a lot of data from constant memory, you will generate only 1/16 \(roughly 6 percent\) of the memory traffic as you would when using global memory. But the savings don’t stop at a 94 percent reduction in bandwidth when reading constant memory! Because we have committed to leaving the memory unchanged, the hardware can aggressively cache the constant data on the GPU.

So after the first read from an address in constant memory, other half-warps requesting the same address, and therefore hitting the constant cache, will generate no additional memory traffic.

The trade-off to allowing the broadcast of a single read to 16 threads is that the 16 threads are allowed to place only a single read request at a time. For example, if all 16 threads in a half-warp need different data from constant memory, the 16 different reads get serialized, effectively taking 16 times the amount of time to place the request. If they were reading from conventional global memory, the request could be issued at the same time. In this case, reading from constant memory would probably be slower than using global memory.

