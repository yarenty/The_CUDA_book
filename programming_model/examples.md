# Examples

### Thread creation - calling a kernel function

Kernel function must be called with an execution configuration:



```c
#define ASCII_SIZE 256

__global__ void myFirstKernel(char *ascii) {
    ascii[threadIdx.x] = threadIdx.x;
}


int main(void) {
    char ascii_h[ASCII_SIZE];
    char *ascii_d;
    
    cudaMalloc( &ascii_d, ASCII_SIZE);
    myFirstKErnel<<< 1, ASCII_SIZE >>>(ascii_d);
    cudaMemcpy(ascii_h, ascii_d, ASCII_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(ascii_d);
    return 0;
}
```



### Device properties

| Device property | Description |
| :--- | :--- |
| char name\[256\] | ASCII string identifying device |
| size\_t totalGlobalMem | Global memory available on device in bytes |
| int maxThreadsPerBlock | Maximum number of threads per block |
| intmultiProcessonCount | Number of multiprocessors on device |

```c
#include <stdio.h>

int main(void) {
    cudaDeviceProp prop;
    int n_devices;
    
    cudaGetDeviceCount(&n_devices);
    for( int i=0; i < n_devices; i++){
        cudaGetDevicProperties( &prop );
        printf("\n Info of device %d :", i);
        printf("\n Device name: %s", prop.name);
        double gMem = ((double) prop.totalGlobalMem) / ( 1024*1024*1024);
        printf("\n Global memory available: ~ %.3g GByte",gMem);
        printf("\n Max thread per block: %d", prop.maxThreadPerBlock );
    }
    
    return 0; 
}
```



### Vector multiplication with CUDA

```c
/* CUDA vector multiplication */

__global__ void vector_multi( float *A, float *B, float *C) {
    int idx = threadIdx.x;
    C[idx] = A[idx] * B[idx];
}


/* Initialize vectors A & B with random numbers */
void init_vector( float **vec, int size) {
    float *tmp;
    (*vec) = (float *) malloc( size * sizeof(float));
    tmp = (*vec);
    for (int i=0; i<size; i++){
        tmp[i] = 1 + (float)(100.0 * rand()/( RAND_MAX + 1.0 ));
    }
}



int main(void) {
    cudaDeviceProp prop;
    int my_device = -1, n_threads = 0;
    size_t vec_mem_size;
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    
    /* set maximum number of threads will be allocated per block */
    my_device = cudaGetDevice( &my_device);
    cudaGetDeviceProperties(&prop, my_device);
    n_threads = prop.maxThreadsPerBlock;
    vec_mem_size = n_threads * sizeof(float);
    
    printf("Program running on %d threads.", n_threads);
    
    /* Initialize vectors A,B,C on CPU - host */
    init_vector(&h_A, n_threads);
    init_vector(&h_B, n_threads);
    h_C = (float *) malloc(vec_mem_size);
    memset(h_C, 0, vec_mem_size);
    
    /* Initialize vectors A,B,C on GPU - device */
    cudaMalloc(&d_A, vec_mem_size);
    cudaMalloc(&d_B, vec_mem_size);
    cudaMalloc(&d_C, vec_mem_size);
    
    cudaMemcpy(d_A,h_A,vec_mem_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,vec_mem_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,0,vec_mem_size);
    
    /* Execute vector_multi on GPU */
    vector_multi<<< 1, n_threads >>>(d_A, d_B, d_C);
    
    /* copy GPU result back to CPU */
    cudaMemcpy(h_C, d_C, vec_mem_size, cudaMemcpyDeviceToHost );
    
    /* free both CPU and GPU memory */
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B);
    
    return 0;

}



```













