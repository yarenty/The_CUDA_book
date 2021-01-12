# Memory Model



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

![CUDA memory spaces](.gitbook/assets/mem_spaces.jpeg)





### Global memory

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

![Memory types](.gitbook/assets/mem_types.jpeg)



### Registers

Fast region of memory \(fastest\)

Thread local access

Number of 32-bit registers per SM:

* compute capability 1.3: 16K
* compute capability 2.x: 32K

![Registers](.gitbook/assets/mem_registers.jpeg)

{% hint style="info" %}
Compiler switch -ptxas="--verbose" shows info about used registers.

--maxregister=N no more than 32 per kernel
{% endhint %}



### Constant memory

A cached region of global memory

All threads in all blocks can access

64KB capacity, 8KB cache per SM

Cannot be written by device

Can be set by host using cudaMemcpyToSymbol

```c
//device code:
__constant__ float constData[256];

//host code:
fload data[256];
cudamMemcpyToSymbol(constData, data, sizeof(data));
```

Or in the declaration on device:  


```c
//device code:
__caonstant__ float constData[3] = {0.2, 0.666, 0.83};
```



### Shared memory

Fast region of memory

Block-local access

Size per SM:

* compute capability 1.3: 16K
* compute capability 2.x: 16 or 48K

![Shared memory](.gitbook/assets/mem_shared.jpeg)

Possible uses:

* storing intermediate values \(ie: accumulators\) before writing to global memory
* to share data with other threads in the block
* use as a cache to avoid redundant global memory access \(CC1.x\)

Usage - fixed size:

```c
#define SIZE 10

__global__ void myKernel(...) {

    ...
    __shared__ int sharedList[SIZE];
    sharedList[0] = ...
    sharedList[1] = ...
}
```

Usage - dynamic size:

```c
//device code:
__global__ void myKernel(...) {
    ...
    extern __shared__ int sharedList[];

}

...
// host code
size_t ns = numElements * sizeof(int);

//kernel call
myKernel<<grid,block, ns>>>(...);

```



### Textured memory



Texture cache, one per multiprocessor - originally for storing images to give illusion of textured object.

Whan item is read from global memory it is stored on the texture cache.

Allows subsequent readts to utilise this element rather than calling globel memory or constant memory.

Texture memory is NOT kept constant with global memory writes - a write to such address in the same kernel call will return undefined data when read again.

Texture memory is :

* an unusual combination of cache \(separate from registe, global, and shared memory\) and local processing capability - separate from the scalar processors.
* data is stored .in the device global memory, but it is accessed through texture cache - useful for caching \(coalescing is a problem\)
* support linear/bilinear and trilinear hardware interpolation \(graphics\)
* bound to linear memory \(1D problems only\)
* bound to CUDA arrays \(1D, 2D, 3D problems, hardware interpolation\)
* read only, cannot detect dirty data - co cache consistency



 

```c
// declare texture reference
texture<float,1,cudaReadMOdeElementType> texreference;
//must be global on main program
// type, dimension, 

int main(int argc, char** argv) {
    int size=3200;
    float* harray;
    float* diarray;
    float* doarray;
    
    //allocate host and device memory
    harray = (float*) malloc(sizeof(float)*size);
    cudaMalloc((void**) &diarray, sizeof(float)*size);
    cudaMalloc((void**) &doarray, sizeof(float)*size);
    
    //initialize host array before usage
    for(int loop = 0; loop<size; loop++) 
        harray[loop]=(float)rand()(RAND_MAX-1);
    
    //copy array from host to device memory:
    cudaMemcpy(diarray,harray,sizeof(float)*size, cudaMemcpyHostToDevice);
    
    //bind texture reference with linear memory
    cudaBindTexture(0,texreference,diarray, sizeof(float)*size);
    
    //execute kernel
    kernel<<<(int)ceil((float)size/64),64>>>(doarray,size);
    
    //unbind texture reference to free resource
    cudaUnbindTexture(texreference);
    
    //free host and device memory
    free(harray);
    cudaFree(diarray);
    cudaFree(doarray);
    
    return 0;
    
}
```



```c
__global__ void kernel(float* doarray,int size){
    //calculate each thread global index
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    
    //fetch global memory through texture reference
    doarray[index] = tex1Dfetch(texreference, index);

}


__global__ void offsetCopy(float* idata, float* odata, int offset){
    //compute each thread global index
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    
    //copy data from global memory: non-textured
    odata[index]= idata[index+offset];
    
    //copy data from global memory: textured
    odata[index] = tex1Dfetch(texreference,index+offset);
}
```



### Tips & tricks

#### Where to declare variables?

![](.gitbook/assets/tips_variables.png)

#### Variable type restrictions \(OLD\)

Pointers can only point to memory allocated or declared in global memory:

* allocated in the host and passed to the kernel: `__global__ void kernel KernelFunction(float* ptr)`
* obtained as the address of a global variable: `float* ptr = &GlobalVar;`

Since CUDA 4.0 - Unified Virtual Addressing for 64bit OS on CC 2.x overcomes that restrictions - any pointer is allowed.



#### A common programming strategy

Global memory resides in device memory \(DRAM\) - much slower access than shared memory

Profitable way fo performing computation on the device is to **tile data** to take advantage of fast shared memory:

* partition data into subsets that fit into shared memory
* handle each data subset with one thread block by:

  * loading the subset from global memory to shared memory, using multiple threads to exploit memory-level parallelism;
  * performing the computation on the subset from shared memory; each thread can efficiently multi-pass over any data element;
  * copying results from shared memory to global memory;

Constant memory also resides in device memory \(DRAM\) - much slower access than shared memory:

* cached!
* highly  efficient access for read-only data

Carefully divide data according to access patterns:

* R/Only -&gt; constant memory \(very fast if in cache\)
* R/W shared within block -&gt; shared memory \(very fast\)
* R/W within each thread -&gt; register \(very fast\)
* R/W inputs/outputs \(results\) -&gt; global memory \(very slow\)

#### GPU atomic operations

Atomic operations on integers in shared and global memory:

* associative operations on signed/unsigned ints:
  * `atomicAdd()`
  * `atomicSub()`
  * `atomicExch()`
  * `atomicMin()`
  * `atomicMax()`
  * `atomicInc()`
  * `atomicDec()`
  * `atomicCAS()`
* some operations on bits

{% hint style="info" %}
Starting with compute capability 1.1 for global and 1.2 for shared memory.

atomicAdd\(\) also available for float starting with compute capability 2.0
{% endhint %}



#### Typical structure of CUDA program

1. Global variables declaration
   * `__host__`
   * `__device__, __global__, __constant__, texture`
2. Function prototypes
   * `__global__ void kernelOne(...)`
   * `float helperFunction(...)`
3. Main \(\)
   * allocate memory space on the device - `cudaMalloc(&d_GlobalVariablePtr, bytes)`
   * transfer data from host to device - `cudaMemCpy(d_GlobalVariablePtr, h_GlobalVariablePointer,..)`
   * execution configuration setup
   * kernell call - `kernelOne<<<execution configuration>>>(args...)`
   * transfer results from device to host - `CudaMemCpy(h_GlobalVariablePtr, d_...)`
   * optional \(in test mode\): compare against golden \(host computed\) solution
4. Kernel - void kernelOne\(type args...\)
   * variables declaration - `__shared__`
   * automatic variables transparently assigned to register or local memory
   * `__syncthreads()...`
5. Other functions

   * `float helperFunction(int intVar...);`

#### CUDA device memory space

Each thread can:

* R/W per thread registers
* R/W per-thread local memory \(avoid\)
* R/W per-block shared memory
* R/W per-grid global memory
* Read only per-grid constantmemory
* Read only per-grid texture memory

The host can:

* R/W global constant and texture memory





![CUDA device memory space](.gitbook/assets/device_memory.jpeg)























