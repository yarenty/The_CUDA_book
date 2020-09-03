# Error checking



```c
cudaError_t cudaMalloc( void **devPtr, size_t size);
void *malloc(size_t size);

a_h = (int *) malloc(numElements * sizeof(int));

if(a_h == NULL) {
    printf("Error in memory allocation\n");
    exit(11);
}


cudaError_T error = cudaMalloc(&d_a, memSize);
if (error != cudaSuccess) {
    printf("Error in device allocation:%s\n", cudaGetErrorString(error));
    exit(11);
}

```





```c
cudaError_t error;
...

myKernel<<< grid, block >>>(a_d);

/* blocks until the device has completed all preceding requested taska
cudaThreadSynchronize() returns an error if one of the preceding tasks failed
*/

cudaThreadSynchronize();

/* returns the last error that has been produced by any of the runtime calls
 in the same host thread and resets it to cudaSuccess */
error = cudaGetLastError();
 
if(error != cudaSuccess) {
   printf("Error, 'myKernel' : %s\n", cudaGetErrorString(error));
}  
```



CUDA SDK comes with a convenient macro, for checking errors: CUDA\_SAFE\_CALL\(\), located in header file "cutil.h"

Need to add directory "/&lt;SDK&gt;/C/common/inc" in the compiler include search path: `"-I$CUDA_HOME/C/common/inc"`

```c
#define CUDA_SAFE_CALL(call){ \
    cudaError err = call; \
    if (cudaSuccess != err) {\
        fprintf(stderr, "Cuda error in file '%s' in line %i: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE); \
    } }
```













