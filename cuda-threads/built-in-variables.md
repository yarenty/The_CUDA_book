# Built-in variables



### Language extensions: built-in variables

{% hint style="info" %}
You can use those variables inside kernel code ie: to go through loop 
{% endhint %}

```c
struct dim3 {int x, y, z}


dim3 gridDim;
// dimestions of the grid in blocks (gridDim.z unused prior to CUDA 4.0)

dim3 blockDim;
// dimensions of the block in threads

dim3 blockIdx;
// block index within grid

dim3 threadIdx;
// thread index within block
```

Example: 1D blocks and grids

{% hint style="info" %}
Simple Program Multiplt Data - concept!
{% endhint %}

![1D blocks and grid](../.gitbook/assets/block_1d.jpeg)



Example: 2D blocks and grids

![2D blocks and grid](../.gitbook/assets/block_2d.jpeg)



#### Computing global thread ID

Global thread ID can be used to decide what data thread will work on

![Computing global thread ID](../.gitbook/assets/global_thread_id.jpeg)



```c
int idx = blockIdx.x * blockDim.x + threadIdx.x
```

Global thread ID can be used to decida what data a thread will work on - reading memory array idx.





