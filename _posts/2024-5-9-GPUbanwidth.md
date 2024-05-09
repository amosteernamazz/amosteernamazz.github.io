---
layout: article
title: GPU 带宽与检测工具
key: 100012
tags: GPU 带宽
category: blog
date: 2024-05-09 14:31:05 +08:00
mermaid: true
---


## global memory带宽检测

带宽包括理论带宽和实际带宽，实际带宽检测包括CPU方法和GPU方法

### 实际带宽计算

#### CPU计算带宽方法

相比起GPU来说，比较粗糙

   ```cpp
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

int main() {
    // 假设我们有一个足够大的数组
    size_t bytes = 1024 * 1024 * 1024; // 1GB
    float* host_ptr = new float[bytes / sizeof(float)];
    float* device_ptr;
    cudaMalloc((void**)&device_ptr, bytes);

    // 填充host_ptr...（如果需要）

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();
  
    // 同步设备（可选，但在这里确保没有之前的操作干扰）
    cudaDeviceSynchronize();

    // 执行内存传输
    cudaMemcpy(device_ptr, host_ptr, bytes, cudaMemcpyHostToDevice);

    // 同步设备以确保传输完成
    cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();

    // 计算时间差
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // 计算带宽
    double bandwidth = (bytes / 1024.0 / 1024.0) / (duration / 1000.0); // MB/s

    std::cout << "Estimated bandwidth: " << bandwidth << " MB/s" << std::endl;

    // 清理内存
    cudaFree(device_ptr);
    delete[] host_ptr;

    return 0;
}
   ```

#### GPU计算带宽方法

 GPU时间与操作系统无关，主要通过GPU的event计时完成

```c++
#include <cuda_runtime.h>  
#include <stdio.h>  
  
// 假设的kernel函数声明（这里只是示意）  
__global__ void kernel(float* d_odata, float* d_idata, int size_x, int size_y, int NUM_REPS) {  
    // ... kernel的实现 ...  
}  
  
int main() {  
    cudaEvent_t start, stop;  
    float time;  
    cudaError_t err;  
  
    // 创建事件  
    err = cudaEventCreate(&start);  
    checkCudaErrors(err);  
    err = cudaEventCreate(&stop);  
    checkCudaErrors(err);  
  
    // 记录开始事件  
    err = cudaEventRecord(start, 0);  
    checkCudaErrors(err);  
  
    // 假设的GPU内存分配和数据传输（这里只是示意）  
    // float* d_odata, d_idata; // 需要提前分配  
    // cudaMemcpy(...); // 需要进行数据传输  
  
    // 执行kernel  
    int grid = ...; // 定义grid大小  
    int threads = ...; // 定义block大小  
    kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y, NUM_REPS);  
  
    // 记录结束事件  
    err = cudaEventRecord(stop, 0);  
    checkCudaErrors(err);  
  
    // 等待结束事件  
    err = cudaEventSynchronize(stop);  
    checkCudaErrors(err);  
  
    // 计算时间（毫秒）  
    float milliseconds;  
    err = cudaEventElapsedTime(&milliseconds, start, stop);  
    checkCudaErrors(err);  
  
    // 假设我们知道数据传输的总量（以字节为单位）  
    size_t totalBytesTransferred = size_x * size_y * sizeof(float) * 2; // 假设输入和输出都是float数组，并且大小相同  
  
    // 计算带宽（字节/秒）  
    double bandwidth = (double)totalBytesTransferred / (milliseconds * 1e-3);  
  
    // 输出结果  
    printf("Execution time: %f ms\n", milliseconds);  
    printf("Estimated bandwidth: %f GB/s\n", bandwidth / (1024 * 1024 * 1024)); // 转换为GB/s  
  
    // 销毁事件  
    err = cudaEventDestroy(start);  
    checkCudaErrors(err);  
    err = cudaEventDestroy(stop);  
    checkCudaErrors(err);  
  
    // ... 其他清理代码 ...  
  
    return 0;  
}  
  
// 辅助函数来检查CUDA错误  
void checkCudaErrors(cudaError_t err) {  
    if (err != cudaSuccess) {  
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));  
        exit(EXIT_FAILURE);  
    }  
}
```

### 理论带宽计算

 **以V100为例的理论带宽计算**

  * V100使用HBM2作为带宽（时钟频率877MHz、为double data rate RAM 、内存带宽为4096 bit。得到的带宽为

  $
  0.877 * 10^9 * 4096 / 8 * 2 / 10^9 = 898GB/s
  $


  * 代码得到GPU带宽

```cpp
cudaDeviceProp dev_prop;
CUDA_CHECK( cudaGetDeviceProperties( &dev_prop, dev_id ) );
printf("global memory bandwidth %f GB/s\n", 2.0 * dev_prop.memoryClockRate * ( dev_prop.memoryBusWidth / 8 ) / 1e6 );
```

 **理论带宽的影响因素：GDDR中的ECC导致带宽下降**

  * ECC是一种用于检测和纠正数据传输或存储中错误的编码技术。通过在数据中添加冗余位（即ECC位），ECC技术能够检测和纠正某些类型的错误，从而提高数据的完整性和准确性。然而，这些额外的ECC位会占用原本用于数据传输的带宽，导致所谓的ECC overhead
  * 当在GDDR中加入ECC时，为了提供错误检测和纠正的能力，需要在每个数据传输单元（如一个字节或一组字节）中添加额外的ECC位。这些ECC位并不直接用于传输有效数据，因此会占用原本可以用于数据传输的带宽。这就是ECC overhead的来源，它会导致GDDR的理论带宽下降。
  * HBM2是一种高带宽内存技术，其设计初衷就是为了提供极高的数据传输速率。为了支持ECC而不牺牲带宽，HBM2采用了专门的硬件设计，为ECC位分配了独立的存储空间。这意味着ECC位不会占用原本用于数据传输的带宽，因此不会导致理论带宽的下降。这种设计使得HBM2能够在保持高带宽的同时，提供强大的数据错误检测和纠正能力。
  * 总结来说，ECC在GDDR中会导致ECC overhead，进而降低理论带宽，是因为ECC位占用了原本用于数据传输的带宽。而HBM2通过专门的硬件设计，为ECC位分配了独立的存储空间，从而避免了ECC对带宽的影响。这种设计使得HBM2在保持高带宽的同时，能够提供强大的数据错误检测和纠正能力。


#### Visual profiler 内核性能分析工具



 **requested throughput 和 global throughput** 

   * 系统的global throughput 相当于物理理论带宽（考虑到cache line 要一起传送的带宽）
   * 系统的requested memory 相当于实际带宽（并未考虑 cache line）
   * 应当让requested throughput尽可能接近global throughput。


  

