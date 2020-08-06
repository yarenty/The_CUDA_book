# Introduction

## What is CUDA?

CUDA stands for Compute Unified Device Architecture, it is a parallel computing platform and programming model created by NVIDIA and implemented by the graphics processing units \(GPUs\) that they produce. CUDA gives developers direct access to the virtual instruction set and memory of the parallel computational elements in CUDA GPUs.

### What is difference between CPU and GPU?

Using CUDA, the GPUs can be used for general purpose processing \(i.e., not exclusively graphics\); this approach is known as GPGPU. Unlike CPUs, however, GPUs have a parallel throughput architecture that emphasizes executing many concurrent threads slowly, rather than executing a single thread very quickly.

![](.gitbook/assets/CUDA_processing_flow.png)

CUDA has several advantages over traditional general-purpose computation on GPUs \(GPGPU\) using graphics APIs:

* Scattered reads–code can read from arbitrary addresses in memory
* Unified virtual memory \(CUDA 4.0 and above\)
* Unified memory \(CUDA 6.0 and above\)
* Shared memory–CUDA exposes a fast shared memory region \(up to 48 KB per multi-processor\) that can be shared amongst threads. This can be used as a user-managed cache, enabling higher bandwidth than is possible using texture lookups.
* Faster downloads and readbacks to and from the GPU
* Full support for integer and bitwise operations, including integer texture lookups

Limitations

* CUDA does not support the full C standard, as it runs host code through a C++ compiler, which makes some valid C \(but invalid C++\) code fail to compile.
* Interoperability with rendering languages such as OpenGL is one-way, with access to OpenGL having access to registered CUDA memory but CUDA not having access to OpenGL memory.
* Copying between host and device memory may incur a performance hit due to system bus bandwidth and latency \(this can be partly alleviated with asynchronous memory transfers, handled by the GPU's DMA engine\)
* Threads should be running in groups of at least 32 for best performance, with total number of threads numbering in the thousands. Branches in the program code do not affect performance significantly, provided that each of 32 threads takes the same execution path; the SIMD execution model becomes a significant limitation for any inherently divergent task \(e.g. traversing a space partitioning data structure during ray tracing\).
* Unlike OpenCL, CUDA-enabled GPUs are only available from Nvidia
* No emulator or fallback functionality is available for modern revisions
* Valid C/C++ may sometimes be flagged and prevent compilation due to optimization techniques the compiler is required to employ to use limited resources.
* A single process must run spread across multiple disjoint memory spaces, unlike other C language runtime environments.
* C++ Run-Time Type Information \(RTTI\) is not supported in CUDA code, due to lack of support in the underlying hardware.
* Exception handling is not supported in CUDA code due to performance overhead that would be incurred by across many thousands of parallel threads running.
* CUDA \(with compute capability 2.x\) allows a subset of C++ class functionality, for example member functions may not be virtual \(this restriction will be removed in some future release\). \[See CUDA C Programming Guide 3.1–Appendix D.6\]
* In single precision on first generation CUDA compute capability 1.x devices, denormal numbers are not supported and are instead flushed to zero, and the precisions of the division and square root operations are slightly lower than IEEE 754-compliant single precision math. Devices that support compute capability 2.0 and above support denormal numbers, and the division and square root operations are IEEE 754 compliant by default. However, users can obtain the previous faster gaming-grade math of compute capability 1.x devices if desired by setting compiler flags to disable accurate divisions, disable accurate square roots, and enable flushing denormal numbers to zero.

