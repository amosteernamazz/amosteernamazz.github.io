---
layout: article
title: GPU 性能优化方法
key: 100025
tags: GPU性能优化
category: blog
date: 2023-02-08 00:00:00 +08:00
mermaid: true
---



# GPU加速常见优化方法
![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/984f7bd88de6490585deef103ba78ef9~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

 **总结**
  * data layout transformation
    * 用于针对struct等数据结构进行数据分布的优化
    * SoA结构、DA结构、广义DA结构
  * scatter & gather
    * 引出应使用gather编程方法，而非scatter方法
  * tailing & joint register and shared memory tailing 
    * 提供片上内存tailing，用于GEMM加速，同时为了减少访问次数采用引入register tailing
  * Grid-stride loop
    * 为解决数据量 > num of threads 引入grid-stride loop
    * 同时加入register tail，解决重复计算工作
  * Privatization
    * 为了解决atomic中造成并行度下降的问题，提供先计算block层面的atomic结果，然后再将结果写到global memory。
  * Algorithm Cascading
    * 为了减少并行开销，提出混合序列算法与并行算法的机制
  * Binning
    * 为了将scatter转变为gather编程方法，提供Binning映射
      * 其中编写gather的算法复杂度，影响数据的可伸缩性，应尽量保证复杂度在$O(n)$
  * Cutoff Binning
    * 针对特定领域：分子生物学
  * Compaction
    * 针对稀疏矩阵等矩阵的存储问题，提出一系列的矩阵压缩算法
  * Regularization
    * 为了解决threads之间的imbalance问题

<!--more-->

### 数据分布优化

 充分利用burst memory是很重要的优化方法

 **原因**
  * 如果burst内的数据没有立刻被使用的话（DRAM的buffer中存放burst），则会被下一个burst代替，需要重新传输。

 **CPU与GPU的区别**
  * 对于CPU来说，data layout对程序的影响没有那么显著。
    * 因为CPU有large cache per thread，可以cache部分数据，没有那么依赖于DRAM的burst data。下面的array of struct结构中，thread0的cache会储存整个struct的内容。

  * 对于GPU来说，data layout对程序的影响很显著。
    * 因为GPU的cache比较小。GPU的cache主要适用于memory coalesing，而不是locality

 **结构体优化方法**
  * <font color = red>为了充分利用burst，GPU创建struct的时候，使用**DA(discrete array)的结构**（如下struct of arrays的结构）</font>

  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0cb24edf605946a7b4fb0e957e27627e~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

  上图中Array of structures对应SoA结构，Structure of array对应DA方法

 **广义的DA方法**
  * <font color = red>更广义上的SoA结构：**ASTA**(Array of Structures of Tiled Arrays)是一种 SoA的变体。相当于AoS of mini-SoA(of size coarsening factor)</font>
  **解决问题**
    * 解决OpenCL需要对不同height、width有不同数据结构的kernel的问题
    * 解决`partition camping`问题：也就是数据集中在某一个bank/channel上，没有充分利用DRAM aggregate bandwidth

 **数据分布优化实例**
  * 通常`coarsening factor` (下面eg是4) **至少设置为block内的threads个数**。
  ```cpp
   struct foo{
      float bar;
      int baz;
   };
   // AoS方法
   __kernel void AoS(__global foo* f){
      f[get_global_id(0)].bar *=2.0;
   }
   // DA方法
   __kernel void DA(__global float* bar, __global int* baz){
      bar[get_global_id(0)] *=2.0;
   }
   // ASTA方法
   struct foo_2{
      float bar[4];
      int baz[4];
   }
   __kernel void ASTA(__global foo_2* f){
      int gid0 = get_global_id(0);
      f[gid0/4].bar[gid0%4] *=2.0;
   }
   ```

  * 结果
   在NVIDIA的arch下，DA与ASTA的性能相似
  
  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2495dce06e9748248a3cb3a6c1ce9cba~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




### 将数据Scatter转变为Gather以提高性能


 GPU应该避免使用scatter，应该使用gather的方法。
 在GPU上的程序改变scatter为gather可以提升性能


 **定义**

  scatter : 并行输入，将值写入非连续内存
  gather(owner compoutes rules) : 并行输出，值写入到连续内存。


 **缺点与优点**
  
  * scatter 缺点
    * contentious write (write conflict) 需要被hardware serialize。（下图红色的arrow）。当thread多的时候会有很多conflict，write到某一个位置会被serialized
    * random write无法充分利用memory burst
    * atomic的arch直到最近才被支持

  * gather 优点
    * write的时候充分利用burst
    * 没有write conflict，不需要serialize write
    * input会有重复的，可以利用好cache


 **为什么程序中常见scatter**


  * input一般是irregular的，output一般是regular的。
     * 从irregular data映射到regular data是简答的，这也是为什么很多程序是scatter的
     * input是particle coordinate(x,y,z), output是3d spatial grid

  * 有些时候each input只影响有限个output，所以conflict write的影响没有那么大



 **gather缺点**

  * 存在overlapping read
    * 可以被hardware使用cache/shared memory来缓解

  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/db6d3d414ca743a0923b836a123113fb~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



 **例子**

  Direct Coulomb Summation(DCS)



### Tiling

 **原因**
  减少global memory的带宽压力

 **定义**
  将buffer传输给片上存储，然后该段数据被多次读取


![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ff0d36cda88d4148976621b24de581a8~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



 **为什么有shared memory**

  * shared memory越大，tile size越大，越能减少global memory 的带宽压力

  * 如果片上存储空间只可被单独thread可见，则片上存储小。
    * 解决方法是建立所有threads的统一片上内存空间shared memory。



 **tail优化方法效果**

  * 在比较中，不适用tailing，只使用cache。
    * 在GPU上，因为cache相对于shared memory来说更大，所以使用tiling的提升效果不是很高。在UIUC 408 Lecture 14里面的例子里，使用tilning只提升了40%左右的速度
      * 原因是因为绝大多数access to global memory都是通过L1 cache的，cache hit rate有98%。



 **例子**
  * GEMM
  * Conv




### 联合 Register 和 Shared Memory Tiling


 **目的**
  由于register和shared memory的硬件实现不同，进行联合register and shared memory联合tailing 可增加 throughput

 **register 特点**

  * 延迟低
  * 高带宽: 每个thread每个clock可以进行多次register
  * 数据加载不能并发
  * 数据每个线程私有
  * ***进行register tiling需要thread coarsening***

 **shared memory 特点**

  * 延迟比register个位数比值
  * 带宽相对register低
  * 可以并发加载



 **联合shared memory和register tailing在GEMM中的原因**

  * 在shared memory tiling的时候，数据重复使用发生在shared memory的数据被多个thread访问，而不是来自于一个thread内部访问一个value多次。
    * Tile size是T * T的话，每一个thread load一个M，一个N到shared memory，sync（确保数据都在shared memory中），然后遍历一小行M和一小列N来计算一个P，sync（确保shared memory被使用完），然后处理下一个tile。
      * 对于每一个M的值，被T（结果P中tile的num col）个thread使用。
      * 对于每一个N的值，被T（结果P中tile的num row）个thread使用。
    * Tile size是s * s的话，每一个thread load一个M，一个N到shared memory，sync（确保数据都在shared memory中），然后遍历一小行M和一小列N来计算一个P，sync（确保shared memory被使用完），然后处理下一个tile
      * every M value reused U time
      * every N value reused T time

  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/08f8d3dff32248b1be07234021c11310~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7b512ce3b79b42c2866ea2c3438326ab~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



  * 因为计算P的会有两个sync（load to shared memory, wait for all comp on shared memory finish)， 所以S的大小也不能太小，否则sync会占用主要的时间

  * tile size不一定是square的。



 **例子**
  GEMM : joint register and shared memory tiling



### Grid-stride loop / thread granularity / thread coarsening 


 **目的**
  针对计算维度超高的场合，nvidia给出的blockDim = 1024、gridDim最大32位整数（约20亿）。对于计算数据量达到20000亿以上的单元素线程操作使用Grid-stride loop，可以拥有更好的并行计算效率

 **定义**
  * 对于数据规模 > 线程数的场景，将跨grid数据合并在一个thread运算，以减少线程启动销毁开销。
    * thread 0会处理 elem 0, elem 0 + num thread in grid, elem 0 * 2 * num thread in grid. 

 **优点**

  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/9dafcb9db7b84e53b28dc127a38bcfa2~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  * 对重复工作结果进行复用，从而减少instruction processing stream的带宽限制
    * 可以理解为一些会重复的computation，现在shared through register。本来register 是local to each thread, 无法shared across thread的
    * 访问register的throughput很大，per thread per cycle可以访问多个register file
    * 访问register的latency很小，只有1 clock cycle

  * 可扩展性。支持数据规模大于硬件线程数。

  * 可以微调block的大小，调节其为device中SMs的数目的倍数，来支持不同大小的block。

      ```cpp
      int numSMs;
      cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
      // Perform SAXPY on 1M elements
      saxpy<<<32*numSMs, 256>>>(1 << 20, 2.0, x, y);
      ```

  * 线程重用会分摊线程创建和销毁成本，以及内核在循环之前或之后可能执行的任何其他处理（例如线程私有或共享数据初始化）。

  * 更容易debugging：将block大小改为launch 1 thread 1 block的kernel来debug，不需要改变kernel内容

      ```cpp
      saxpy<<<1,1>>>(1<<20, 2.0, x, y);
      ```

  * 与sequential code一致都有for loop的存在，更好理解代码

  * 在循环展开时，循环体变大，可以更好的使用ILP



 **缺点**

  * 每个thread使用更多的register，可能导致一个sm内总的thread数量减少（因为register constrain）。导致insufficent amount of parallelism。
    * not enough block per sm to keep sm busy
    * not enough block to balance across sm (thread合并了以后，总的thread数量减小，总的block数量也就减少了，而且每个block的时间久了，容易导致imbalance)
    * not enough thread to hide latency。通过warp间swap来hide latency，但是当总thread减少，总warp减少

  * 因为有larger computation tiles. 会产生计算资源浪费。
     * 一般通过减少block内的threads个数解决
       * 如果一个thread在coarsening以后干了k*k个thread的工作，把原来的block size分别变为width/k和height/k来避免more padding and waste computation


  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1a38e97b6818439f86854af2f0a6865d~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

  (two output per thread的idle，更多idle)

  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a4f0daaa3e8f4377b3fc11bd17964deb~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



 **使用环境**
  当提高的效率大于并行度减少带来的效果


 **例子1**

  ```cpp
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
  add<<<32 * numSMs, blockSize>>>(N, x, y);

  // GPU function to add two vectors
  __global__
  void add(int n, float *x, float *y) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // 这里通过for loop对thread进行复用
    for (int i = index; i < n; i+=stride)
      y[i] = x[i] + y[i];
  }
  ```


 **例子**
  * DCS : thread granularity
  * 7 point stencil 
  * GEMM: thread granularity



### Privatization

 **目的**
  避免多个thread通过使用atomic同时写入一个内存地址，使用atomic会drastically decrease memory throughput。

 **定义**
  * 缓存需要多次写入的数据到片上存储，甚至是register。
    * 每个thread或每组threads都有自己的输出的copy，将数据首先合并到local result然后再进行写入。

  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a82c920e918a4263b8b82197443108ff~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



 **CPU与GPU的区别**

  * CPU上由于thread的数量较小，private copy of output不会是问题

  * GPU上由于thread的总数量很多，使用privitization需要注意以下两点

    * 使用shared memory或者是register是否会导致 thread  per sm 减少
    * 合并私有副本的开销会比较大，因为这里依旧需要atomic
      * 解决方法是将block atomic后的数据copy到shared memory。这样可以同时兼顾latency (5 cycle)与bandwidth（在shared memory上使用atomic的bandwidth依旧可以）


 **例子**


  * Histogram




### Algorithm Cascading


 **定义**
  * 混合序列与并行算法，让每个thread有足够的工作(sequential)来减少并行开销，而且允许线程之间通过并行来进行计算

 **并行开销**
  * 线程的建立和销毁、 线程和线程之间的通信、 线程间的同步等因素造成的开销。
  * 存在不能并行化的计算代码，造成计算由单个线程完成， 而其他线程则处于闲置状态。
  * 为争夺共享资源而引起的竞争造成的开销。
  * 由于各cpu工作负载分配的不均衡和内存带宽等因素的限制，一个或多个线程由于缺少工作或因为等待特定事件的发生无法继续执行而处于空闲状态。

 **例子**

  * Prefix-sum three phase
  * reduction 




### Binning


 **目的**
  * GPU计算中scatter计算比gather更容易实现，但就计算性能来看，gather更优秀。建立一个output element与 input element之间的映射关系，对每个output element通过binning映射都有相对应的input elements
    * 难点：一般input是irregular的，output是regular的，很难从regular到irregular找到映射，从irregular到regular的映射更简单一些(e.g.  atom 3d location is irregular, grid position is regular, use 3d location & divide to get grid location)


 **定义**
  * 把irregular input按照某种规则放入bin中。从regular output到irregular input的映射的时候，就可以到对应的regular bin中去找，以加快速度。
    * 每个bin都有其规则，把data按照bin的建立规则给合并。

  * 可以理解为data coarlesing，当访问bin的时候，访问all data inside bin。



 #### Data Scalability

  **定义**
   * 在GPU中为了将scatter转为gather程序，在建立gather与scatter之间映射关系时，存在算法复杂度问题。在该问题中因为GPU数据量大，因此算法复杂度与输入数据量之间的关系决定了GPU数据的可伸缩性。
     * complexity 与 input size 不是linear的情况，Data Scalability较差



  **Data Scalability较差问题**

   * 如果代码为了改变为parallel over gather而改变complexity为$O(nlog \ n)$ or $O(log \ n)$，对大量数据来说，效果不好。
     * 但是使用gpu的情况又是在数据量很大的情况。

   * sequential O(N) algorithm can easily outperform O(n log n) parallel algorithm

  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0a4e5bccb72f4208aa1db0bd8cb8b9ce~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)





  **HPC data scalability**

   * first thing when have parallel algorithm, <font color = red>how to change it to **O(n)**</font>so that it's data scalable



  **较差DCS例子**

   * DCS算法需要对每一个grid point计算每一个atom的contribution，算法复杂度是$O(V^2)$


  **例子**
   * CDS的简化算法



### Binning切断

 **目的**

  * 为了解决data scalability的问题，使用近似算法从而得到 linear complexity的结果

    * cutoff binning允许O(n)复杂度算法

 **应用领域**
   目前限定在分子生物学

 **定义**

  * 基于physica只考虑在一个cutoff threshold内的元素之间的关系，不考虑cutoff外的元素之间的关系或者cutoff外用简单的计算来近似




 **特点**

  * cutoff binning对于cpu来说容易adopt，因为CPU可以使用scatter的方法

  * cutoff binning对于gpu来说难adopt，因为gpu使用gather的方法



 **例子**

  * biomolecules


### 数据压缩优化

 **定义**
  压缩数据中的空数据，减少memory开销（global memory, shared memory, memory transfer bandwidth)

![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a965b3dd1a6d4335ae1d4268cc41aebe~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


 **常见稀疏矩阵压缩格式**
  * 稀疏矩阵的经典压缩为COO(Coordinate)每一个元素都需要用一个三元组表示[行号,列号,数值]
  * CSR(Compressed Sparse Row)[数值,列号,行偏移]
  * CSC(Compressed Sparse Column)[数值,行号,列偏移]
  * DIA(Diagonal)[按照对角线进行存储，从左下往右上]
  * ELL(ELLPACK)用两个和原始矩阵相同行数的矩阵来存储[第一个矩阵存的是列号，第二个矩阵存的是数值]
  * HYBELL+COO(Hybrid ELL+COO)为了解决ELL中某行特别多造成其他行浪费，将多出来的元素使用使用COO单独存储
  * <font color = red>JDS compaction</font>


  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6b22a0cdadbc48eeb88c97931a062786~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


 **例子**
  * SpMV


### Regularization

 **目的**

  解决threads之间的load imbalance问题

 **load imbalance**

  * 线程分支
  * 一个block内如果有load imbalance，会导致资源在在整个block运行结束之前不会释放，导致block占用有限的resource更多的时间（尽管在imbalance的时候，block不需要这么多的resource），导致num thread per SM降低
