---
layout: article
title: GPU DRAM与优化
key: 100024
tags: GPU DRAM 硬件优化
category: blog
date: 2024-05-09 14:31:05 +08:00
mermaid: true
---


# Global Memory

## 带宽检测

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


  


## DRAM硬件与优化方法

 本部分主要讲述global memory的硬件实现DRAM


### Bit Line & Select Line定义

 ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_3.png)

 ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_2.png)

  **原理**

   * 一个电容决定一个bit， 通过select线选择读取哪个bit line，然后通过bit line选择对应的bit。
   * 需要不停检查和电容充放电，因此叫做DRAM

  **特点**

  * bit line的容量大，存在电容耦合与信号衰减，同时更容易遇到数据冲突，同时bit line的带宽有限。
  * 每个bit电容，需要信号放大器放大，进一步限制传统DRAM设计

 
### Core Array & Burst的DRAM数据传输

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_4.png)

 **改进**

  * 在bit line -> bus的传输中，添加column buffer将bit line数据进行备份，提高bit line其他数据传输到bus的效率（数据局部性）


 **传输过程**

  * 数据传输分为两个部分
    * core array -> column latches / buffer 
    * column latches / buffer-> mux pin interface

 **传输耗时**

  * core array -> column latches / buffer 耗时久
  * buffer -> mux pin interface 的耗时相对较小（原先耗时长）


 **burst与burst size/ line size**

  * burst：当访问一个内存位置的时候，select line选择的bit line的数据会全部从core array传输到column latches/ buffer中。使用数据根据mux来确定传输给bus哪些数据，这样可以加速
  * burst size/ line size：读取一次memory address，会有多少个数据从core array被放到buffer中


### Multiple Banks技术

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_5.png)


![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_6.png)


  **引入原因**

   * 从core array到buffer的时间长，单个传输导致实际使用的bus interface数据bandwidth未被完全利用($T_c:T_b = 20:1$)。如果只使用一个bank导致带宽大部分时间为空闲。
   * 所以需要在一个bus 上使用多个banks，来充分利用bus bandwidth。如果使用多个bank，大家交替使用interface bus，保证bus不会空闲，保证每个时间都有数据从bus传送过来。


  **一个bus需要多少个bank？**

   * 如果访问core array与使用bus传输数据的时间比例是20:1，那么一个bus至少需要21个bank才能充分使用bus bandwidth。
   * 一般bus有更多的bank，不仅仅是ratio+1
     * 并发访问：如果同时对多个bank进行操作，这些操作可以并发进行，因为它们访问的是不同的物理存储体。假设一个DRAM模块有两个bank，并且内存控制器同时接收到两个读请求。如果这两个请求都针对同一个bank，那么它们将顺序执行，导致等待时间和带宽的减少。但是，如果这两个请求分别针对两个bank，那么它们可以并发执行，从而提高带宽。
     * 避免bank冲突：当多个请求尝试同时访问同一个bank时，会发生冲突。为尽可能减少冲突可能性，使用更多的banks。
  
  **banks与burst的速度对比**

  * burst的速度快于banks的速度

### Multiple Channels技术

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_7.png)


  * 一般的GPU processor要求带宽达到128GB/s，HBM2要求带宽为898GB/s，在使用multiple banks后仍未满足要求（使用DDR DRAM clock为1GHz，每秒能传送8 bytes/ 1 words，得到的传输速率为16GB/s）因此需要多个channels


### 数据Interleaved（交织）分布技术

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_8.png)


  **数据交织原因**

   * 充分利用channel的带宽，实现max bandwidth。
   * 允许在core array -> buffer 的时间进行转化成多个channels获取数据（并行），减少总的时间。

  **如何实现数据交织**

  将数据平均分布在channels对应的banks中（上图中）


## global memory 内存合并和内存对齐及其优化方法


### global memory 内存合并

  **global memory 内存合并原因**

   * 在GPU中对于内存数据的请求是以wrap 为单位，而不是以thread 为单位。**<font color = purple >- - - - ->** *warp 内thread 请求的内存地址会合并为一个warp memory request，然后这个request 由一个或多个memory transaction 组成*。
   * 具体使用几个transaction 取决于request 的个数和transaction 的大小。

  **global memory 数据流向**
   * global memory request一定会经过L2，是否经过L1 取决于cc 和code，是否经过read only texture cache取决于cc 和code 。

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_9.png)


  **GPU 与CPU 对memory 的处理方式**

   * CPU
     * 片上有很大的cache，CPU thread访问连续的内存会被缓存。不同thread 由于不同core，数据相互不影响。
     * 充分利用内存的方法：每个core负责一段连续的内存。（e.g. thread 1 : array 0-99; thread 2 : array 100-199; thread 3 : array 200-299.）

   * GPU 
     * GPU 的线程以warp为单位，一个SM上有多个warp运行，warp线程操作内存通过L1/shared memory进行。
     * 使用warp操作内存，不同thread 对cache的结果会产生不同影响。thread0读取数据产生的cache会对thread1读取数据产生的cache产生影响。
     * 为充分发挥带宽，应当在warp的每个iteration中保证花费全部cache line。因为有很多warp同时在sm上运行，等下一个iteration的时候 cache line/DRAM buffer已经被清空了。
  
  

  **GPU常用优化方法**

   * 使用内存对齐和内存合并来提高带宽利用率
   * ***<font color = purple> sufficent concurrent memory operation 从而确保可以hide latency***
     * ***<font color = purple> loop unroll 从而增加independent memory access per warp, 减少hide latency所需要的active warp per sm***
     * ***<font color = purple>modify execution configuration 从而确保每个SM都有足够的active warp。***





### 内存对齐优化方法

 L1 cache line = 128 bytes, L2 cache line 32 bytes，warp的内存请求起始位置位于cache line的偶数倍，为了保证对global memory 的读写不会被拆为多个操作，应保证存储对齐。



  **image cache line**

   为了防止对不同row下的读写产生多个memory segment，导致速度变慢，在每一行末尾加入padding

   padded info叫做 `pitch` 

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_10.png)



  **CUDA API**

   align to 256 bytes

   ```cpp
   // 1d, aligned to 256 bytes
   cudaMalloc();
   cudaMemcpy();
   cudaFree();

   // 2d 分配, aligned to 256 bytes
   cudaMallocPitch();
   cudaMemcpy2D();

   // 3d, aligned to 256 bytes
   cudaMalloc3D();
   cudaMemcpy3D();
   ```
   


  **结构体大小对内存对齐的影响**

   * 结构体大小应保证为 1, 2, 4, 8, or 16 bytes，如果不是这些大小，则会产生多个transaction。
  
   * 如果一个struct是7 bytes，那么padding成8 bytes会用coarlesed access。但是如果不paddig的话则会是多个transaction。

   * 下面的marcro可以align struct从而确保coarlesed access

   ```cpp
   struct __align__(16) {
   float x;
   float y;
   float z; 
   };
   ```



### Global Memory 读流程

  注意： GPU L1 cache is designed for spatial but not temporal locality. Frequent access to a cached L1 memory location does not increase the probability that the data will stay in cache. L1 cache是用于spatial（连续读取array）而不是temporal（读取同一个位置的），因为cache line很容易被其余的thread evict。

  

  **内存读性能：global memory load efficency**

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_11.png)



   nvprof.gld_efficency metrics衡量了此指标



  **内存读模型：Simple model**

   *  在128 bytes/32 bytes的模式下，会产生128 bytes/ 32 bytes / 64 bytes的memory transaction （32 bytes当four segment的时候也会是128 bytes）。在L1中 memory被合并分块为 32-, 64-, or 128- byte memory transactions，在L2中被分块为32 bytes。



#### Read-only texture cache

   * 使用条件：CC 3.5+ 可以使用read only texture cache

   * cache line大小：The granularity of loads through the read-only cache is 32 bytes. 



#### CC 2.x Fermi 

   * 2.x default 使用 L1 + L2 cache

   * 2.x 可以通过config disable L1 cache

   ```shell
   // disable L1 cache
   -Xptxas -dlcm=cg

   // enable L1 cache
   -Xptxas -dlcm=ca
   ```



   **128 bytes transaction**

   * 每个thread的大小对request的影响
     * 如果每个thread请求的数据大于4 bytes（32 * 4 = 128)，则会被切分为多个128 bytes memory request来进行。
     * 如果每个thread请求8 bytes，这样保证了每个传送的128 bytes数据都被充分利用(16 threads * 8 bytes each)
     * 如果每个thread请求16 bytes，four 128-bytes memory request,这样保证了传送的128 bytes数据被充分利用
   * request拆分到cache line层面上，解决indenpent 问题
     
     * 当warp的内存请求位于同一个连续对齐的cache line内。 
     ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_12.png)




     * 数据都位于cache line内，但之间的关系无序。只要warp memory request是在128 bytes transaction内，只会进行一个memory transaction。
      ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_13.png)


     * warp连续但是并未对齐，导致产生两个128 bytes transaction
      ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_14.png)





     * 所有线程请求同一地址


      ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_15.png)

     * warp的线程以32 addresses为单位scatter到global memory的情况。

      ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_16.png)


   * 32 bytes transaction

     * transaction只由L2 cache获取global memory数据

   



     * 内存请求连续对齐的128 bytes，进行需要进行四次transaction。bus利用率100%
      ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_17.png)



     * 内存请求非连续，但都在128bytes内，请求四次transaction。
      ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_18.png)




     * 请求相同地址，只需要一次transaction
      ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_19.png)




     * 请求四个32 bytes的scatter分布，相对<font color= red>128 bytes的方式，缓存利用率高。
      ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_20.png)




#### CC 3.x Kepler cache line大小

   * 3.x default 使用 L2 cache，不使用L1 cache

   * 3.5 / 3.7 可以使用read only texture cache

   * 3.5 / 3.7 可以config使用L1 cache

   * L1 cache line size 128 bytes

   * L2 cache line size 32 bytes

   为什么L2 cache 需要以1、2、4倍数传输：为了避免DMA fetch ***<font color= purple>DMA FETCH: The DMA supports an AXI bus width of 128/64 bits. In the case where the source descriptor payload ends at a non-128/64 bit aligned boundary, the DMA channel fetches the last beat as the full-128/64 bit wide bus. This is considered an over fetch.***
   当使用L2 cache only的时候，memory transaction是32 bytes. Each memory transaction may be conducted by one, two, or four 32 bytes segments。可以减少over-fetch

  **L1/L2读取顺序**
   * 当使用L1 + L2时候，memory transaction是128 bytes。
   Memory request 首先会去L1，如果L1 miss会去L2，如果L2 miss会去DRAM。




#### CC 5.x Maxwell

   * 5.x default使用L2 cache，32 bytes transaction

   * 5.x 可以使用read only texture cache，32 bytes transaction

   * 5.x 可以config使用L1 cache（default不使用）


#### CC 6.x Pascal



### Global Memory 写流程

  **与读的区别**
   * 读可以用L1，写只能用L2，写只能用32 bytes
   * 多个non-atomic thread 写入同一个global memory address，只有一个thread写入会被进行，但是具体是哪个thread是不确定的


  **efficency** 

   * memory store efficency 与 memory load efficency的定义相似

   * nvprof.gst_efficency metrics 可衡量


  **transaction** 

   * 一个128 bytes的requested（4 个transactions）相对两个64 bytes(2*2 transactions)的requested来说，速度更快

   * 读transaction的例子

     * 128 bytes 的连续对齐内存request，需要4个transactions
      ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_21.png)




     * 128 bytes 的request，请求内存大小为64 bytes的连续内存空间，则需要2个transactions

      ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_22.png)



### global memory操作与硬件的关系

  第一次访问，全部4个数据都放到buffer里

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_23.png)


  第二次访问使用后面两个数据（连续内存访问），直接从buffer里读取数据，不用再去core array
  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_24.png)



  **burst 原因** 
   * 在从core array -> buffer 的过程需要的时间长，在每一次从core array 到buffer 的过程中，传输burst数据，在每一次读取中，应让数据充分使用。因为两次core array -> buffer 时间远远大于两次 buffer -> bus的时间。

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_25.png)
  


   * 蓝色的部分是core array到buffer的时间。红色的部分是buffer到pin的时间


 
 
 


## Minimize host to device transfer


### Batch Small Transfer

  * 为避免多个small memory copy，使用one large memory copy，将多个small memory打包成为一个large memory，然后进行copy

### Fully utilize transfer

  * 尽量将传输次数减少，如果GPU计算并不方便，也使用，减少数据传输次数。

