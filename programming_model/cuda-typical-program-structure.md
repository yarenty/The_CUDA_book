# CUDA - typical program structure

#### Global variables declaration

* \_\_host\_\_
* \_\_device\_\_
* \_\_global\_\_
* \_\_constant\_\_
* texture

#### Function prototypes

* \_\_global\_\_ void kernelOne\(...\)
* \_\_device\_\_ / \_\_host\_\_ float handyFunction\(...\)

#### Main\(\)

* allocate memory space on the device - cudaMalloc\(&d\_GlobalVarPtr, bytes\)
* transfer data from host to device - cudaMemCpy\(d\_GlobalVarPtr, h\_GlobalVa...\)
* execution configuration setup
* kernel call - kernelOne&lt;&lt;&lt;execution configuration&gt;&gt;&gt;\(args ...\);
* transfer results from device to host - cudaMemCpy\(h\_GlobalVarPtr, d\_Global...\)
* free memory space on device - cudaFree\(d\_GlobalVarPtr\);

#### Kernel - void kernelOne\(type args,...\)

* variables declaration:
  *  \_\_shared\_\_
  * automatic variables transparently assigned to registers or local memory 
* \_\_syncthreads\(\)...

```c
__global__ void cudaKernel(float *a, float *b, float *c)
{
    int tID = some_mean_to_identify_thread_index;
    c[tID] = a[tID] + b[tID];
}



int main(int argc, char* argv[]) {
    /* Allocate and initialize vector a,b and c on both CPU and GPU */
    /* Data transfer for copying input vectors a and b on GPU. */
    cudaKernel<<<some_environment_info>>>(a, b, c);
    /* Data transfer for copying output vector c on CPU */
    /* Free memory. */
    return 0;
}
```

