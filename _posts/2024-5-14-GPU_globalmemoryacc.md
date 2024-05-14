---
layout: article
title: GPU global memory内存优化
key: 100013
tags: GPU GlobalMemory 软件优化 内存优化
category: blog
date: 2024-05-14 10:58:21 +08:00
mermaid: true
---



# global memory 内存优化

## global memory 内存合并思想

  **DRAM局部性思想与 global memory 的局部性原理**

   * GPU对于内存数据的请求是以wrap为单位，而不是以thread为单位。如果数据请求的内存空间连续，请求的内存地址会合并为一个warp memory request，然后这个request由一个或多个memory transaction组成。具体使用几个transaction 取决于request 的个数和transaction 的大小。
     * 该方法从数据存取->数据利用中利用局部性思想，与DRAM硬件实现中在数据存储->数据读取中利用局部性思想同理

  **GPU与CPU在局部性原理的不同**

   * cache大小：CPU的片上有很大的cache，相对于CPU，GPU相对较少。
   * thread对于cache的处理：
     * 组织形式：CPU中的不同thread访问连续的内存会被缓存（局部性），GPU的thread通过warp进行封装，warp访问内存通过L1/shared memory等进行
     * 数据影响性：CPU的不同thread之间数据相互不影响，GPU的thread之间的数据会存在互相影响的问题
   * 充分利用内存方法：
     * CPU：由于thread相对稀缺，为充分利用core性能，每个core负责一段连续的内存。（e.g. thread 1 : array 0-99; thread 2 : array 100-199; thread 3 : array 200-299.）
     * GPU：由于cache相对稀缺，为充分发挥带宽，应当在warp的每个iteration中保证花费全部cache line。


<!--more-->

## global memory内存合并的优化方法


 * GPU对于内存数据的请求是以wrap为单位，而不是以thread为单位。如果数据请求的内存空间连续，请求的内存地址会合并为一个warp memory request，然后这个request由一个或多个memory transaction组成。具体使用几个transaction 取决于request 的个数和transaction 的大小。（前文所述）
 * 其中从一个request得到多个memory transactions时，时间消耗集中于多次的memory transactions流程中，因此有必要针对数据的分布和对齐方式进行优化

### 内存对齐优化

 * 内存对齐优化主要针对在内存读取中对某结构化数据让其尽量在同一个memory transaction完成数据的传输，以防止对后续数据需要多个不必要的memory transactions

 **内存对齐API**

  * 分配的数据均对齐

```cpp
// 1d, aligned to 256 bytes or 512 bytes
cudaMalloc();
cudaMemcpy();
cudaFree();

// 2d, aligned to 256 bytes or 512 bytes
cudaMallocPitch(); // with pitch
cudaMemcpy2D();

// 3d, aligned to 256 bytes or 512 bytes
cudaMalloc3D(); // with pitch
cudaMemcpy3D();
```

  * 其中，为防止对同一个row下的读写产生多个memory segment，导致速度变慢，在每一行末尾加入padding，让其大小与memory transaction大小一致
  * padded info叫做 `pitch`


![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_10.jpeg)



  **结构体与内存对齐**

   * 结构体大小应保证为 1, 2, 4, 8, or 16 bytes等倍数，如果不是这些大小，可能会产生多个transactions，同时需要考虑包括cache line等因素的影响。
   * 如果一个struct是7 bytes，那么padding成8 bytes。但是如果不padding的话则会是多个transactions。

```cpp
struct __align__(16) {
float x; // 4 bytes
float y;
float z;
};
```

### 数据分布优化

 **结构体的AoS结构与SoA结构**

  * 为了充分利用burst，GPU创建struct的时候，使用DA(discrete array)的结构
  * 图中Array of structures对应AoS结构，Structure of array对应DA方法（也可以是SoA结构）

  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0cb24edf605946a7b4fb0e957e27627e~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

```c++
// 假设有一个结构体表示3D点
struct Point3D {
    float x;
    float y;
    float z;
};

// 使用AoS方式存储多个点
Point3D pointsAoS[N]; // N个点

// 访问某个点的x坐标。在GPU上，如果我们要对所有的x坐标进行某种操作，我们需要遍历整个数组，并且每次都要从一个不同的结构体中读取x字段。这可能导致内存访问的不连续，从而降低性能。
float xCoord = pointsAoS[i].x;


// 使用SoA方式存储多个点的x、y、z坐标
struct Points{
  float xCoords[N];
  float yCoords[N];
  float zCoords[N]; // N个点
}

// 访问某个点的x坐标。在GPU上，如果我们要对所有的x坐标进行某种操作，我们可以直接对整个xCoords数组进行操作，而不需要跳转到不同的内存位置。这种连续的内存访问模式可以显著提高性能。
float xCoord = Points.xCoords[i];
```

 **广义DA方法**

  * 更广义上的SoA结构：ASTA(Array of Structures of Tiled Arrays)是一种 SoA的变体。相当于AoS of mini-SoA


```cpp
  struct foo{
    float bar;
    int baz;
  };
  // AoS方法
  __kernel void AoS(__global foo* f){
    f[get_global_id(0)].bar *= 2.0;
  }
  // DA方法
  __kernel void DA(__global float* bar, __global int* baz){
    bar[get_global_id(0)] *=2.0;
  }
// ASTA方法
struct foo_tile {
    float bar[4]; // 每个tile包含4个float值
};

// ASTA方法
__kernel void ASTA(__global foo_tile* tiles){
    int gid0 = get_global_id(0);
    int tile_id = gid0 / 4; // 计算tile的索引
    int elem_id = gid0 % 4; // 计算tile内部元素的索引
    tiles[tile_id].bar[elem_id] *= 2.0; // 修改对应tile中的bar值
}
```


 **ASTA应用举例**

  * 解决OpenCL需要对不同height、width有不同数据结构的kernel的问题
  * 解决`partition camping`问题：也就是数据集中在某一个bank/channel上，没有充分利用DRAM aggregate bandwidth

 **AoS、SoA和ASTA性能对比**

   * 在NVIDIA的arch下，DA（SoA）与ASTA的性能相似

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2495dce06e9748248a3cb3a6c1ce9cba~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




### 连续访问or合并内存访问(coalesced memory access)

 * 连续访问相邻的内存位置，这样可以利用GPU的缓存机制（如L1和L2缓存）来提高访问速度。CUDA中的coalesced access（合并访问）就是一种优化技术，它确保一个warp（线程束）中的所有线程都访问连续的内存位置，从而减少内存访问的次数。

```c++
#include <cuda_runtime.h>  
  
// 定义块和线程的大小  
const int BLOCK_SIZE = 256;  
const int THREADS_PER_WARP = 32;  
  
// 假设我们有一个数组，每个元素都是一个float  
__global__ void coalescedAccessKernel(float* d_array, int n) {  
    int index = threadIdx.x + blockIdx.x * blockDim.x;  
    if (index < n) {  
        // 确保warp内的线程访问连续的内存位置  
        // 这里假设n是warp大小的整数倍  
        float value = d_array[index];  
        // ... 对value进行一些操作 ...  
        d_array[index] = value; // 假设我们修改了value的值  
    }  
}  
  
int main() {  
    // 分配设备上的内存  
    int n = BLOCK_SIZE * (BLOCK_SIZE / THREADS_PER_WARP); // 确保n是warp大小的整数倍  
    float* h_array = new float[n];  
    float* d_array;  
    cudaMalloc((void**)&d_array, n * sizeof(float));  
  
    // 初始化h_array...  
  
    // 将数据从主机复制到设备  
    cudaMemcpy(d_array, h_array, n * sizeof(float), cudaMemcpyHostToDevice);  
  
    // 调用内核函数  
    coalescedAccessKernel<<<n / BLOCK_SIZE, BLOCK_SIZE>>>(d_array, n);  
  
    // 等待内核完成  
    cudaDeviceSynchronize();  
  
    // 将数据从设备复制回主机  
    cudaMemcpy(h_array, d_array, n * sizeof(float), cudaMemcpyDeviceToHost);  
  
    // 释放内存...  
  
    return 0;  
}
```


## global memory 其他优化方向

 * 主要从GPU内存结构对global memory进行优化和如何更好地利用带宽来进行优化和算法优化三方面介绍

### 内存结构层次优化

 **使用常量内存**

  * 如果数据在内核执行期间是只读的，并且数据的大小小于64KB，那么可以将它存储在常量内存中。
  * 常量内存具有高速缓存特性，可以被所有线程同时访问，而不需要任何同步。

```c++
__constant__ float constantData[64]; // 声明常量内存

__global__ void useConstantMemory() {
    int index = threadIdx.x;
    float value = constantData[index]; // 访问常量内存
    // ... 其他操作 ...
}
```

 **使用纹理内存**

  * 对于某些类型的数据访问模式（如2D图像），使用纹理内存可能更有效。纹理内存通过硬件插值来加速访问，并支持某些类型的内存过滤。
  * 但纹理内存在CUDA中不太常见，但在某些特定的应用中，它仍然可能是一个有效的优化策略。

 **使用共享内存**

  * 如果线程块内的线程需要频繁地访问相同的数据，那么可以将这些数据存储在共享内存中。共享内存的访问速度比全局内存快得多。

```c++
__shared__ float sharedData[32]; // 声明共享内存

__global__ void useSharedMemory(float* globalData, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int index = tid + bid * blockDim.x;

    if (index < n) {
        // 从全局内存加载数据到共享内存
        sharedData[tid] = globalData[index];
        __syncthreads(); // 等待所有线程完成加载

        // 使用共享内存中的数据
        // ...

        // 将结果写回全局内存（如果需要）
        // ...
    }
}
```

在利用带宽对GPU的global memory进行优化的方法中，包括从host到device数据传输的优化（zero-copy memory）、数据传输与计算的并发、流控制、使用更紧凑的数据结构

### 零拷贝内存优化与异步传输

 * 页锁定内存（也称为固定内存或零复制内存）是主机内存的一种特殊类型，它不会被操作系统分页或交换到磁盘上。这种内存可以被GPU直接访问，而不需要CPU的参与。

```c++
#include <cuda_runtime.h>

int main() {
    // 分配页锁定内存
    void* pinnedHostMemory;
    cudaHostAlloc(&pinnedHostMemory, sizeof(float) * N, cudaHostAllocDefault);

    // ... 填充pinnedHostMemory ...

    // 分配设备内存
    float* deviceMemory;
    cudaMalloc(&deviceMemory, sizeof(float) * N);
 
    // 异步传输数据到设备
    cudaMemcpyAsync(deviceMemory, pinnedHostMemory, sizeof(float) * N, cudaMemcpyHostToDevice);

    // ... 在这里启动GPU内核 ...

    // 等待数据传输完成（如果需要）
    cudaDeviceSynchronize();

    // 释放内存
    cudaFree(deviceMemory);
    cudaFreeHost(pinnedHostMemory);

    return 0;
}
```

### 重叠内存传输与计算(异步与流)

 * 在数据传输和内核执行之间使用异步操作可以重叠数据传输和计算，从而提高整体性能。这通常涉及到使用CUDA流（Streams）和事件（Events）。

```c++
#include <cuda_runtime.h>

// 假设我们有一个简单的内核函数
__global__ void myKernel(int *data, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        // 对数组中的每个元素进行操作
        data[index] *= 2;
    }
}

int main() {
    // 主机上的数据
    int hostData[1024];
    // 初始化数据...

    // 分配设备上的内存
    int *devData;
    cudaMalloc(&devData, sizeof(int) * 1024);

    // 创建两个CUDA流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 使用第一个流将数据从主机复制到设备
    cudaMemcpyAsync(devData, hostData, sizeof(int) * 1024, cudaMemcpyHostToDevice, stream1);

    // 在第二个流上启动一个内核
    myKernel<<<1, 1024, 0, stream2>>>(devData, 1024); // 注意指定了流stream2

    // 假设我们需要等待第一个流的数据传输完成，然后再进行其他操作
    cudaStreamSynchronize(stream1);

    // 此时，由于第二个流上的内核已经启动，并且与第一个流是并发的，
    // 因此我们可以继续执行其他操作，而不需要等待内核完成。

    // ... 在这里可以执行其他与流无关的操作，或者启动更多内核到不同的流 ...

    // 最后，等待第二个流上的内核完成
    cudaStreamSynchronize(stream2);

    // 如果需要，可以将结果从设备复制回主机（这里省略了代码）

    // 清理
    cudaFree(devData);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
```


### 优化数据结构

 * 重新设计数据结构来减少内存占用是一种常见的方法。
   * 使用更紧凑的数据类型（如float代替double），
   * 消除不必要的数组维度或数据结构来减少内存使用量。
   * 使用特殊的数据结构（如稀疏矩阵表示）来减少存储大量零值所需的内存。



### 算法优化
 
  * 算法优化以减少内存占用可能涉及更复杂的策略。
    * 使用迭代算法代替递归算法，
    * 使用在线（online）算法代替离线（offline）算法。
    * 多个计算步骤组合成一个步骤，从而避免中间结果的传输。
    * 设计算法以就地（in-place）方式更新数据，而不是创建新的数据集。
    * ......

