---
layout: article
title: GPU Program Model
key: 100025
tags: CUDA
category: blog
date: 2023-02-08 00:00:00 +08:00
mermaid: true
---



# Program Model




## Program Hierarchy


**GPU硬件与软件的关系**
 * GPU的硬件依托于SMs、warp机制（warp的划分、调度）
 * 软件依托于Grid、block、thread的模型结构
 * GPU的SMs与block对应
 * SMs在处理block的时候，由硬件的warp调度器完成block的拆分（拆分为以32个threads为单位），然后由SM调度给SMs的cores



| 编程结构 | 函数对应 | 硬件实现 |
|---|---|---|
| grid | kernel function | null |
| block | block | SM |
| warp | 每32的threads | SM中的一个单元 |
| thread | thread | core |



### Grid, Block, Warp, Thread

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5d212c2838bb4a48ac6e48ad0af83e44~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  **Grid** 

   * 每个<font color = red>kernel function</font>被映射到一个grid上
   * Thread between block 通过<font color = red>global memory</font>进行交流
   * CPU call GPU kernel function 是 <font color = red>asynchronize</font> 的

  **grid层面的kernel function限制**
   * 仅仅只能得到<font color = red>device memory数据</font>
   * 返回值类型为<font color = red>void</font>
   * 不支持<font color = red>可变数量</font>参数
   * 不支持<font color = red>静态</font>参数
   * 不支持<font color = red>函数指针</font>
   * 行为为<font color = red>异步</font>行为

  **Block** 

  1. 每个block运行在一个<font color = red>sm</font> (GPU Core上)
  2. threads in same block 通过<font color = red>shared memory + sync</font>进行交流



  **Thread**

  1. CUDA Threads: <font color = red>SPMD</font> single program multiple data
  2. threads on GPUs are extremely lightweight

<!--more-->


  **软件层面与硬件层面的对应**
  ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/69524e29e5fd456c960dcf127f6c888c~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)





  **什么时候使用CUDA**

   * 大量数据，大量thread，<font color = red>每个thread有足够的计算</font>（从而CI不会太低），能够充分利用CUDA多线程硬件
   * Host->Device的内存拷贝是很花费时间的，所以只有在device上进行足够的op操作，才值得使用device

### 错误机制

 **CUDA API错误机制**
  cuda API会返回`cudaError_t`用于error code，如果没有error的话，则会返回 `cudaSuccess`

 **可读错误信息**
  可以使用`char* cudaGetErrorString(cudaError_t error)` 得到human readable error message



  ```cpp
  #define CHECK(call) \ 
  {\ 
    const cudaError_t error = call; \ 
    if (error != cudaSuccess) \ 
    {\ 
      printf("Error: %s:%d, ", __FILE__, __LINE__); \ 
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1); \ 
    }\
  }

  // example usage
  CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

  kernel_function<<<grid, block>>>(argument list); 
  CHECK(cudaDeviceSynchronize());
  ```






## Device Management

### device属性

 ```cpp
 cudaDevicePeop dev_prop;
 cudaGetDeviceProperties(&dev_prop, 0);

 dev_prop.maxThreadsPerBlock; // 每个block的最大thread数
 dev_prop.multiProcessorCount; // GPU拥有的SMs个数
 dev_prop.clockRate; // 时钟频率
 dev_prop.maxThreadsDim[0/1/2]; // 每个block每一维最大的threads数目
 dev_prop.maxGridSize[0/1/2]; // 包含grid的每一维的最大尺寸（threads层面）
 dev_prop.warpSize; // warp的threads数目
 dev_prop.regsPerBlock; // 每个sm可以使用的register的个数，通常与算法中总共的thread数量一起使用，从而确定每个thread可以使用多少个register，从而dynamic select which kernel to run。
 dev_prop.sharedMemPerBlock; // 每个sm可以使用的shared memory大小。希望host code根据不同的hardware从而使用不同的shared memory大小，从而充分使用硬件。

 // 是否支持async compute & memory copy
 // 1: 当执行内核的时候，支持host和device 之间并发复制数据
 // 2: 当执行内核的时候，支持host和device 之间双向并发复制数据
 // 3: 二者都不支持，为0
 dev_prop.asyncEngineCount; 
 ```
<!--  
 <font color = red>regsPerBlock</font> *每个sm可以使用的register的个数与算法中总共的thread数量一起使用*
 
 <font color = red>sharedMemPerBlock</font> *host根据不同需求大小，进行设置* -->






## Block

### 设置经验


  block
  SMs
  block's threads


  * 每个<font color = red>grid 的block数 > SMs 数目</font>
    * 为了更好利用SM资源性能
  * 每个<font color = red>SM 的block数目 > 1</font>
    * 减少线程同步 __syncthreads()对延迟的影响
  * 当block大小不同时，设置<font color = red>每个block的threads个数在128到256之间较好</font>
  * 设置值应该是<font color = red>warp大小的倍数</font>
  * 每个block的threads大小<font color = red>最小为64</font>，而且只能出现在每个SM有多个并发块。A minimum of 64 threads per block should be used, and only if there are multiple concurrent blocks per multiprocessor.
  * 如果延迟对性能影响大，选用block的threads小的blockdim可以提高性能。特别是<font color = red>对于经常调用__syncthreads()函数</font>。
    * 使用更大的blockdim并不会导致性能一定提高。

### Sync



  `__syncthreads()` 用来synchronize all threads in block

  **使用注意事项**
   * 如果有if else then statement + syncthreads 在branch内部的话，<font color = red>all threads inside block either all run if or all run else</font>. 否则就会wait forever
   * CUDA runtime针对sync函数会先确保所有resource都满足情况下才会执行。否则跑到一半发现resource不够，某些thread无法运行，其余的thread因为synchronize等待无法运行的thread。

### block之间的无关性




  * 当在不同配置的硬件计算中，同样的代码可以运行，被称为transparent scalability。
  * 在运行当中，不同block之间的运行顺序不确定。
  * 为了确保transparent scalability，<font color = red>不允许block之间synchronize，只在block内部允许synchronize</font>。


![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0dc9b628648c487fae1d8c080e6d9740~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)








## Thread和 Wrap

### Map block to warp

  **thread序号**
   * <font color = red>3D</font>
   $thread_3 =threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)$
   * <font color = red>2D</font>
   $thread_2 =threadIdx.x + blockDim.x * (threadIdx.y)$

  **warp id**
   ***<font color = purple>$warp \ id =thread \ id \  \% \  32$</font>***

  **例子**

   ***<font color = purple>dimBlock=(16, 16), this means blockdim.x = 16</font>***

   ***<font color = purple>threadIdx.y / 2 = same number = same warp = warp id</font>***

   ***<font color = purple>threadIdx.x + 16 * ( threadIdx.y % 2 ) = lane id</font>***

### 多线程分支



#### 是什么

   * 如果<font color = red>一个warp内多个thread会走不同的路径，会被序列化运行</font>

   ```cpp
   __global__ void mathKernel1(float *c) {
     int tid = blockIdx.x * blockDim.x + threadIdx.x; 
     float a, b;
       a = b = 0.0f;
     if (tid % 2 == 0) { 
       a = 100.0f;
     } else {
      b = 200.0f;
     }
     c[tid] = a + b; 
   }
   ```
   ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c3b3bb01a0bb4e888c99e500c20eb2e1~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




#### 如何避免
   * 如果代码中有branching (if else)，但是<font color = red>warp内的thread只走一个path</font>(都走了if condition)，不会有branching
   * 一个解决branching的常用方法就是<font color = red>设置branch granularity是warp的倍数</font>、这样就能<font color = red>保证一个warp内的全部thread都只走一个brach</font>，而且依旧two cotrol path

   ```cpp
   __global__ void mathKernel2(void) 
   {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    float a, b;
    a = b = 0.0f;
    // tid = 2k * warpSize
    if ((tid / warpSize) % 2 == 0) { 
      a = 100.0f;
    }
    // tid = (2k + 1) warpSize
    else {
      b = 200.0f;
    }
    c[tid] = a + b; 
   }
   ```



#### 变量预测
   **是什么**
    对于<font color = red>分支体较小的分支</font>，编译器会使用<font color = red>分支预测</font>来代替分支。

   **执行过程**
    在分支预测中，每个线程的预测变量利用<font color = red>条件流设为0或1</font>。条件流都会执行，但只有<font color = red>指令预测为1</font>的才会执行，预测为<font color = red>0</font>不会执行，但也不会进入stall。

   **例子**
    下面的例子里通过directely expose branch predication来调用compiler优化（运行的速度与上面以warp granularity切分的一样）。



   ```cpp
   __global__ void mathKernel3(float *c) 
   {
     int tid = blockIdx.x * blockDim.x + threadIdx.x; 
     float ia, ib;
     ia = ib = 0.0f;
     bool ipred = (tid % 2 == 0); 
     if (ipred) {
       ia = 100.0f; 
     }
     if (!ipred) { 
       ib = 200.0f;
     }
     c[tid] = ia + ib; 
   }
   ```

### warp内thread的register共享




  **是什么**

   * 使用shuffle指令，<font color = red>warp内的thread可以访问其余thread的寄存器</font>。

   * <font color = red>lane定义</font>：warp内的thread叫做lane。lane number from 0 to 31



  **引入原因**

   * 提供更强的<font color = red>可编程性</font>
   * 原来thread之间进行数据交换需要使用shared memory。现在在一个warp内部可以使用register，意味着得到同warp的register<font color = red>更小的latency以及更大的bandwidth</font>。



  **例子**

   <font color = red>__shfl_down_sync</font>
   ```cpp
   // warp shuffle for reduction
   val += __shfl_down_sync(0xffffffff, val, 16);
   val += __shfl_down_sync(0xffffffff, val, 8);
   val += __shfl_down_sync(0xffffffff, val, 4);
   val += __shfl_down_sync(0xffffffff, val, 2);
   val += __shfl_down_sync(0xffffffff, val, 1);

   #define FULL_MASK 0xffffffff
   for (int offset = 16; offset > 0; offset /= 2)
      val += __shfl_down_sync(FULL_MASK, val, offset);
   ```


   <font color = red>__shfl_xor_sync</font>

   ```cpp
   // Use XOR mode to perform butterfly reduction
      for (int i=16; i>=1; i/=2)
          value += __shfl_xor_sync(0xffffffff, value, i, 32);
   ```



#### 独立线程调度

  **开始**

   * 从volta版本开始的使用warp内线程的独立调度

     * Volta GV100 是第一个支持独立线程调度，支持更细粒度的并行线程同步和线程协作。

  **SIMT结构的改进**

   * pascal版本及之前使用SIMT方式执行。其通过使用reconverge(在SIMT模式下对于分支结构处理后，将线程再次融合)和减少追踪线程状态的资源数量，以最大化并行。
   * 上述方法特点为不确定的reconverge。如果下面的程序没有在Z之前reconverge，可能发生：Z也被分成两个step运行，尽管可以一个step运行
![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8391f8b92e9b42b0852857cb0b21df2c~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



   **Volta SIMT model**

   * 每一个thread都有自己的program counter

   * volta版本及之后的threads会从分支结构分开然后重写结合，过程仍然是通过一组thread进行，运行相同的指令。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1e1e0158b92d494d90971c85c6f1d4ee~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


   **运行过程**

   * 程序依旧是以SIMT来运行的。程序同时进行，只是不满足分支条件的被mask掉。

     * 允许更多的latency的可能。

   **引入原因**
   * 相对原先版本来说，虽然允许每个thread独立运行（通过SIMT），但是会尽量将线程融合起来，增加SIMT利用率，可以一起执行指令。

   * 原先版本：Z运行之前并没有进行融合，因为编译器认为Z可能与X Y有数据依赖。导致SIMT的efficency降低。

![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3ceb423a1aa74ffe8e08fcb17a7dcbb5~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


   * 新版本：可以使用cuda9的syncwarp()来保证线程之间融合，从而实现SIMT高效率。

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ae984ed855d14063af822711501cc0eb~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



#### warp shuffle & primative

  Synchronized shuffle内主要介绍了warp-level的原语，可以用于warp级别的加速。
  active mask query介绍了mask中的常用错误。
  Thread synchronization介绍了在warp内的thread synchronization用于同步warp内的线程

##### Synchronized shuffle

   **特点**

   * 从Kepler开始使用shuffle在warp内的thread之间交换数据。

   * 比shared memory的latency小，不需要额外的memory来进行数据交换

   * lane / lane idx：single thread within a warp

   ```cpp
   // for 1d threads block
   laneId = threadIdx.x % 32;
   warpId = threadIdx.x / 32;
   ```

   * 4个主要shuffle 方程，分别有int和float版本，一下列举int版本

   **使用条件**
   * 要求thread首先被sync（也就是不需要再单独使用__syncwarp()语句），所以在调用这些语句的时候，数据thread会被sync 
   * 在新版本中Mask 作用：使用mask 保证warp 内的threads 保持线程同步。
   * 老版本的warpprimative没有强制同步。使用老版本是一个危险的行为。

   **`int __shfl_sync(unsigned mask, int var, int srcLane, int width=warpSize);`**

   * 可以设定width大小并非warpSize大小，则将warp分割未对应大小，laneIndex+warpSize得到对应每个segment的shuffer值。

     * `int y = shfl(x, 3, 16);`
       * threads 0 到 15 would 得到thread 3 数据， threads 16 到 31 得到thread 19数据。

      * 当srcLane > width的话，会warp around the width
      * 当warp内的thread使用shuf使用同一个srcLane的时候，会发生broadcast。

   ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4ccfb34e327b4b60b5f1721f41ca9c04~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


   * 可以使用`shuf_sync` 实现shift-to-left wrap-around operation

   ```cpp
   #define BDIMX 16
   __global__ void test_shfl_wrap(int *d_out, int *d_in, int const offset) 
   { 
    int value = d_in[threadIdx.x];
    value = __shfl(value, threadIdx.x + offset, BDIMX);
    d_out[threadIdx.x] = value;
   }
   test_shfl_wrap<<<1,BDIMX>>>(d_outData, d_inData, 2);
   ```

![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7ed3a230b80241a38612a5f191345f1c~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



   **`int __shfl_up_sync(unsigned mask, int var, unsigned int delta, int width=warpSize)`**

   * 被 mask 指定的线程返回向前偏移为 delta 的线程中的变量 var 的值，其余线程返回0。
     * 调用 shfl_up_sync(mask, x, 2, 16); ，则标号为 2 ~15 的线程分别获得标号为 0 ~ 13 的线程中变量 x 的值；标号为 18 ~31 的线程分别获得标号为 16 ~ 29 的线程中变量 x 的值。

   * 其他数据不变

![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0d638373238147daa6af8d24cb461b58~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



   **`int __shfl_down_sync(unsigned mask, int var, unsigned int delta, int width=warpSize)`** 

   * 被 mask 指定的线程返回向后偏移为 delta 的线程中的变量 var 的值，其余线程返回0。
     * 调用 shfl_down_sync(mask, x, 2, 16); ，则标号为 0 ~13 的线程分别获得标号为 2 ~ 15 的线程中变量 x 的值；标号为 16 ~29 的线程分别获得标号为 18 ~ 31 的线程中变量 x 的值。
   * 其他数据不变

![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4b41e16fbe024dd990d0bbe17aa92e97~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




   **`int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width=warpSize)`** 

   * 被 mask 指定的线程返回向后偏移为 delta 的线程中的变量 var 的值，其余线程返回0。
     * 调用 shfl_down_sync(mask, x, 2, 16); ，则标号为 0 ~13 的线程分别获得标号为 2 ~ 15 的线程中变量 x 的值；标号为 16 ~29 的线程分别获得标号为 18 ~ 31 的线程中变量 x 的值。
     * 当  n = 2k 时，表现为将连续的 n 个元素看做一个整体，与其后方连续的 n 个元素的整体做交换，但是两个整体的内部不做交换。例如 [0, 1, 2, 3, 4, 5, 6, 7] 做 n = 2 的变换得到 [2, 3, 0, 1, 6, 7, 4, 5] 。

     * 当  n ≠ 2k 时，先将 n 拆分成若干 2k 之和，分别做这些层次上的变换。这种操作是良定义的（二元轮换满足交换律和结合律）。例如 [0, 1, 2, 3, 4, 5, 6, 7] 做 n = 3 的变换时，先做 n = 2 的变换，得到 [2, 3, 0, 1, 6, 7, 4, 5]，再做 n = 1 的变换，得到 [3, 2, 1, 0, 7, 6, 5, 4] 。
   * 常用于butterfly address pattern
  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8b5edec07c8d4559b42e39bfdb259c62~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



   * 使用`shuf_xor_sync` 实现butterfly addressing between two thread

   ```cpp
   __global__ void test_shfl_xor(int *d_out, int *d_in, int const mask) 
   { 
    int value = d_in[threadIdx.x];
    value = __shfl_xor (value, mask, BDIMX);
    d_out[threadIdx.x] = value;
   }
   test_shfl_xor<<<1, BDIMX>>>(d_outData, d_inData, 1);
   ```

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ac654fdc487844608ee37435f1e652cc~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




   使用`shuf_xor_sync` 实现warp级原语用于warp性能提高

   ```cpp
   __inline__ __device__ int warpReduce(int mySum) { 
    mySum += __shfl_xor(mySum, 16);
    mySum += __shfl_xor(mySum, 8);
    mySum += __shfl_xor(mySum, 4);
    mySum += __shfl_xor(mySum, 2); 
    mySum += __shfl_xor(mySum, 1); 
    return mySum;
   }


   // global reduction with warp level reduxtion
   // assume block only have 32 threads block
   __global__ void reduceShfl(int *g_idata, int *g_odata, unsigned int n) { 
    // shared memory for each warp sum
    __shared__ int smem[SMEMDIM];
    // boundary check
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x; 
    if (idx >= n) return;
    // read from global memory int mySum = g_idata[idx];
    // calculate lane index and warp index 
    int laneIdx = threadIdx.x % warpSize; 
    int warpIdx = threadIdx.x / warpSize;
    // block-wide warp reduce 
    // 进行warp 内部的threads 同步
    mySum = warpReduce(mySum);
    // save warp sum to shared memory
    if (laneIdx==0) smem[warpIdx] = mySum;
    // block synchronization 
    __syncthreads();
    // last warp reduce
    // 配置最后一层需要warp-level sync数据
    mySum = (threadIdx.x < SMEMDIM) ? smem[laneIdx]:0; 
    // 启用最后一层的warp-level reduce
    if (warpIdx==0) mySum = warpReduce (mySum);
    // write result for this block to global mem
    if (threadIdx.x == 0) g_odata[blockIdx.x] = mySum; 
   }
   ```



   **`unsigned __ballot_sync(unsigned mask, int predicate);`** 

   * 用于生成其他函数需要的mask，可以使用ballot_sync确定warp内只有部分thread参与计算，用于reducetion计算不是32的倍数。

   ```cpp
   // 使用ballot_sync决定warp内只有部分thread参与计算，从而允许reduction计算的时候不是32的倍数
   //  __ballot_sync() is used to compute the membership mask for the __shfl_down_sync() operation. __ballot_sync() itself uses FULL_MASK (0xffffffff for 32 threads) because we assume all threads will execute it.
   unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < NUM_ELEMENTS);
   if (threadIdx.x < NUM_ELEMENTS) { 
      val = input[threadIdx.x]; 
      for (int offset = 16; offset > 0; offset /= 2)
          val += __shfl_down_sync(mask, val, offset);
   }
   ```

   **others**

   ```cpp
   __all_sync, __any_sync, __uni_sync, __ballot_sync,
   __match_any_sync, __match_all_sync
   ```

##### Active mask query
   返回warp内32 threads哪个是active

   * __activemask()函数并没有强制thread进行synchronize。需要进行另外的synchronize。
   * activamask只能用来知道哪些threads是convergent。但不会set，仅仅只是查询使用

   ```cpp
   __activemask()
   ```

   * 错误例子:不保证activemask()是被多个thread同clock执行的

   ```cpp
   if (threadIdx.x < NUM_ELEMENTS) { 
      unsigned mask = __activemask(); 
      val = input[threadIdx.x]; 
      for (int offset = 16; offset > 0; offset /= 2)
          val += __shfl_down_sync(mask, val, offset);
   }
   ```

##### Thread synchronization
   synchronize threads in a warp and provide a memory fence. 更多memory fence的内容见memory部分

   **原因**
    之所以需要使用syncwarp是因为cuda9开始，warp内的thread并不保证在一个clock内同时运行相同的指令。
   **定义**
    同步warp内的thread，并提供memory fence保证同步。
   **应用条件**
    memory fence的道理是和syncthread与shared memory一起使用相同的。
    warp内的thread并不能保证lock step，在写入/读取shared memory的时候，需要使用syncwarp来确保memory fence。发生在thread数据与global或shared memory数据发生交换的时候

   ```cpp
   __syncwarp()
   ```



   ```cpp
   // 一个错误的例子
   // read write to shared memory 依旧可能导致race condition, 因为 += 代表read comp write 
   shmem[tid] += shmem[tid+16]; __syncwarp();
   shmem[tid] += shmem[tid+8];  __syncwarp();
   shmem[tid] += shmem[tid+4];  __syncwarp();
   shmem[tid] += shmem[tid+2];  __syncwarp();
   shmem[tid] += shmem[tid+1];  __syncwarp();

   // 正确的例子
   // CUDA compiler 会在针对不同版本的arch选择删除_syncwarp。 
   // 如果在cuda9之前的硬件上，threads in warp run in lock step, 则会删除这些syncwarp。
   unsigned tid = threadIdx.x;
   int v = 0;
   v += shmem[tid+16]; __syncwarp();
   shmem[tid] = v;     __syncwarp();
   v += shmem[tid+8];  __syncwarp();
   shmem[tid] = v;     __syncwarp();
   v += shmem[tid+4];  __syncwarp();
   shmem[tid] = v;     __syncwarp();
   v += shmem[tid+2];  __syncwarp();
   shmem[tid] = v;     __syncwarp();
   v += shmem[tid+1];  __syncwarp();
   shmem[tid] = v;
   ```


   **老版本syncwarp()**
    老版本的分支前后加入syncwarp()并不能保证在syncwarp()之后reconverge。


   * 老版本效果

   可以使用下面的command编译得到老版本的lock step效果

   ```cpp
   -arch=compute_60 -code=sm_70
   ```







## 资源配置

### 动态资源分配
 block 与SM支持blocks和threads个数
 shared memory大小、block threads数目、支持blocks、算力等

 **动态资源分配**
  * 对于编程grid的每个block，SM是动态分配资源，资源在分配时，需要确定资源是否够用。
  * 对资源分配是以block为单位分配。

 **资源限制**

  * 每个SM的threads数目与每个SM支持的Blocks个数之间的冲突
  * 每个SM的registers数目
  * 每个SM的shared memory大小

#### Block/SM & Thread/Block

  **资源分配以block为单位**
   * threads到到硬件的映射是以block为单位的。一个SM硬件有block 数量限制。

   * 如果SM内的一种或多种资源不够支持最多block运行，cuda runtime则会以block为单位减少同时在一个SM上运行的block。

  **举例**
   * SM 最多8 个block
   * SM 最多1536 个threads 
   * block分块方式： $8*8, 16*16, 32*32$

  **$8*8$**

   * threads：64 threads
   * threads支持最大block：1536 / 64 = 24 blocks
   * SM限制： 8 * 64 = 512 threads

  **$16 * 16$**
   * threads：256 threads
   * threads支持最大block：1536 / 256 = 6 blocks
   * 硬件上使用full threads & block capacity

  **$32 * 32$**
   * threads：1024 threads
   * threads支持最大block：1536 / 1024 = 1 blocks
   * 硬件上无法使用full threads




#### Shared memory

  **假设**
   * 每个SM的shared memory：64 KB
   * 每个SM的threads：2048
   * 带宽：150 GB/s
   * 每秒浮点运算次数：1000 GFLOPS 10000亿
   * 正方形矩阵采用tail size： 16/32

  **TILE_WIDTH 16**

   * shared memory内存计算：$2*16*16*4\ Bytes = 2048 \ bytes$

   * shared memory支持最大block计算：$64 \ KB/ 2048 \ Bytes =32 \ blocks$最大支持32 blocks
  

   * 算力计算：$150/4*16=600 \ GFLOPS$，并未完全利用算力

   * threads计算：$2048/(16*16)=8 \ Blocks$支持最多8 blocks


   * 每一个时间点会有$2*16*16*8= 4096$pendig load ***<font color = purple> (通过检查pending load来判断是否有让SM busy)。这个点很重要，因为使用多个thread的目的就是让sm有很多pending load，这样才能swap between warp</font>***



  **TILE_WIDTH 32**

   * shared memory内存计算：$2*32*32*4\ Bytes = 8kb \ bytes$

   * shared memory支持最大block计算：$64 \ KB/ 8 \ Bytes =8 \ blocks$最大支持32 blocks
  

   * 算力计算：$150/4*32=1200 \ GFLOPS$，理论可以实现全部算力应用

   * threads计算：$2048/(32*32)=2 \ Blocks$支持最多2 blocks
  

   * 每一个时间点会有 2 * 32 * 32 * 2 = 4096 pending loads，与16*16有同样的memory parallelsim exposed。尽管32的复用的内存更大，memory paralle与16一样。可能会存在一些block schedule的问题，因为更少的总block个数。

### Occupancy



 **定义**

  * SM的活动warp/最大可能活动warp

  * 理解为真实使用硬件能力硬件能力的比值


 **对程序的影响**

   * 在较低occupancy中提高occupance表明latency降低，但到达某点再提高性能不会再提升，因为latency已经都被hiding了

#### API



##### prediction

   * `cudaOccupancyMaxActiveBlocksPerMultiprocessor ( int numBlocks, const void func, int blockSize, size_t dynamicSMemSize )` 根据当前的使用的寄存器，SMem等信息判断每个SM上能驻存的Block数。


##### configure

   * `cudaOccupancyMaxPotentialBlockSize ( int minGridSize, int blockSize, T func, size_t dynamicSMemSize = 0, int blockSizeLimit = 0 ) [inline]` 根据当前信息算核函数达到最大占用率Occupy时的BlockSize和GridSize
   * 如果有动态信息，则需要用`cudaOccupancyMaxPotentialBlockSizeVariableSMem`








## 资源调度 & Latency Hiding

### 以warp为单位的调度单元

 **特点**
  * SM的调度单元为warp。
  * warp大小取决于硬件，目前为32
  * 每个SM可以同时运行多个warp的指令
  * warp内线程执行任意顺序

 **原因**

  为了share control unit

 **threads与warp映射**

  如果block是1D的，则每32个映射到一个warp上

  如果block是2D/3D的，会先把2D/3D project到1D上，然后每32个thread映射到一个warp上



  **同一时间只能跑有限个warp，且每个SM中需要放多个warp**

   * 如果一个warp等待前一个指令的资源，则这个warp不会执行。

   * 如果多个warp 都可以运行，则会采用latency hiding的优先机制来选择先运行谁。

   * 每个SM有多个warp，保证了hardware resource会被充分利用（通过schedule来hide latency）

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ee9d3cf7060446ecad65c86d975b3b3c~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




 #### Active Warp


  **定义**

   * active block: 当register和shared memory分配给该block的时候，则称为active block。

   * active warps: active block中包含的warp叫做active warp. 

  **调度器**
   * schedulers会在每个时钟周期内选择active的warp，然后将其分配给执行单元。

   * 根据cc 不同，不同的GPU有不同max num active warp per SM. Kepler support max 64 warp per SM.

  **active warp类型**

   * selected warp: 正在执行的warp
   * stalled warp: 并未准备好执行
   * eligible warp: 为active，但当前并未执行

 #### Zero-overhead scheduling

  **定义**
   * 选择准备好的warp，避免执行时间线中引入空闲和时间浪费

  **为什么没有时间浪费**
   * 因为GPU在运行kernel之前已经把全部的resource申请了，所以switch to new warp的时候这个new warp的资源已经在GPU上了

   * 如果 active warp足够多，硬件会在任何时间点上找warp执行。

  **与CPU区别**
   * 在不同warp内切换减少hide latency决定了GPU不需要大的cache，不需要branch prediction等硬件，可以把更多的硬件给floating point operation的原因
  
   * GPU的warp一旦分配到资源，就会占用资源直到block整个运行结束。CPU存在context switch把register保存到memory中的overhead。

  ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b588ca7bc66844a5822e472c4831fda9~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


 #### Understand Scheduling with Example

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e48319bcf8a64c069cd7ee2027347830~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  **注意**
   下面的讨论都是围绕着slides中特定的硬件版本

  **GPU并行**
   一般GPU支持两种parallel的方式。
   thread level parallel (也就是同时运行多个warp)。
   还有instruction level parallel（也就是一个thread内的前后两个independent instruction可以在一个clock cycle内运行）

  **选择activate warp**
   * 每个clock，会从64个可能来自不同blocks的active warp中选择4个active wap（active warp的定义是sm maintain warp execution context)。其中SM对block的调度是使用或不使用整个block。通过interleave这64个warp来hide latency。这里的平行是**simutaneous multi-threading**。
     * 之所以能选择4个warp是因为有4个warp scheduler (以及对应的pc)
   * SM可以维持64个active warp。通过维持他们的execution context。这64个active warp可以来自多个block。SM对block的schedule是使用或者不使用整个block。SM通过interleave这64个warp来hide latency。

  **选择指令**
   * 每个clock，每个warp (out of 4)，会选择两个独立的instruction来运行 (并非每个GPU arch都可以双发，Volta就不可以双发)。这里的平行是**ILP**。这里独立指令指的是使用各自独立的functional units in SM。
     * 如果程序中有两个独立的FMA，SM硬件中有两组FMA，则在一个clock cycle内这两个FMA会同时运行
   * 如果找不到两个独立指令来运行的话，则运行一个instruction。

  **计算**
   * 从最多8个可能的instruction里，最多4个是数学计算，可以同时被4个32长度的SIMTD ALU处理（same clock)，也可以同时处理load store

### Latency Hiding

#### Why & How GPU Hide latency

  **计算资源利用率**
   * 计算资源的使用率与SM的active warp数量直接相关。每次处理指令之后，warp scheduler会选择1个instruction来运行（可能来自于同一个warp/不同的warp）
  
  **latency定义**
   * latency是warp可以运行下一个指令(从waiting到ready status)的时钟周期的数值。
  
  **latency降低的思路**
   * 硬件：利用sm 硬件是通过让warp scheduler总能找到某些指令来处理，当等待前一个warp的latency。也就是我们希望有尽量让多个指令变为ready status。 

   * 软件：希望更多的warp resident in SM + instruction independent 

   * 举例：如果在等待全局内存访问完成时如果有的独立算术指令，则线程调度程序可以隐藏大部分全局内存延迟。
![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/38e691dc4977416684d095e9c96ff91b~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

  显示有无足够warp比较，scheduler 0有足够的eligable warp，可以通过运行其余的warp来hide latency。scheduler 1没有足够的eligable warp，只能通过stall来hide latency。

#### 实现hide latency的warp需求数目

  从里特尔法则，可以知道下面的公式
  $number \ of \ required \ warps = latency \ * \ throughput$

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c288d3d1d16f4158926c0c7800fba2a2~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



  **对不同cc的latency为L，需要的warp数目**
   适用条件：warp已经拥有指令输入的算子，可以执行下一条指令

   * cc 5.x 6.1 6.2 7.x 8.x : 4L。  因为4个warp scheduler, 每个clock cycle可以issue1个instruction per warp scheduler

   * cc 6.0 : 2L。  因为两个warp scheduler,  每个clock cycle可以issue1个instruction per warp scheduler

   * cc 3.x : 8L。  因为4个warp scheduler, 每个clock cycle可以issue 2个instruction per warp scheduler

   * cc 2.x: 20 warp 每1 个SM。 Fermi have 32 single-precision floating point pipelines (throughput)； latency of one arithemetic instruction is 20 cycles (latency), min of 20 * 32 = 640 threads = 20 warps per SM needed to keep device busy.

#### latency来源

##### 寄存器依赖

   **来源**
    当全部的操作数在register上的时候，操作数的前一条指令的依赖还没有写入。


   **举例**
   cc 7.x 算数指令 需要 16 warp来hide latency。因为计算操作一般是4 clock cycle，需要4*4(L=4)=16 instruction/warps (cc 7.x 每个warp scheduler issue 1 instruction per clock cycle)。如果ready warp数量不够16的话，会导致idle。

##### 片外内存 

   * 对片外内存来说，因为延迟高达几百个cycles，为了降低硬件复杂度，通过提高ILP来提高需求的warp数目。相关指标主要通过intensity（不使用片外内存的指令数与使用片外内存的指令书的比值）来衡量。
   * 当比值很小的时候，需要更多的warp。

##### block 内的sync
   **来源**
    syncthread 会导致latency，因为在warp会因为barrier不会去执行下一条指令。
   **解决**
    让sm有更多的resident block。以减少idle。
    当一个block存在syncthread idle的时候，其余block的warp可以运行来hide latency
   **larger block**
    larger block size并不意味着higher occupancy, 因为sync导致的idle以及resource按照block为单位进行分配










## 任务并行与流

### Stream


 **定义**

  从host到device发送的request形成的queue。cuda runtime保证stream运行的顺序性。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ce8d360e4c4147d39739048c99cdb2d8~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



 **stream资源删除**

  * 只有当stream全部相关的工作都结束以后，才会释放stream相关的resource。
  * 就算是在stream相关工作之前就调用cudaStreamDestroy也只是告诉runtime等这个stream相关工作都运行结束以后就释放resource，也并不是立刻释放resource



#### stream并发

  **stream并发和硬件**
   * Stream支持concurrency是软件层面的概念。
   * 在实际上硬件运行的时候，受限制于硬件，可能做不到不同cuda实现stream并发。



  **常见的stream并发**

   * 将host和device的计算重叠
   * 将host的计算和host与device数据传输重叠
   * 将device计算和host与device数据传输重叠
   * device的计算并发



  **例子**

   * 下面的例子里，尽管使用了3个stream，但是host到device的数据传输不能被concurrent运行。因为他们公用PCIe总线。所以他们的执行只能被序列化。

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a8d64bfa7dc44bb092a170da0eb20859~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


   * 如果device PCIe支持duplex PCIe bus, 那么使用针对不同方向的stream，可以使用PCIe总线传输，memory move是可以overlap的。下图里从device到host与从host到device就是使用了两个stream，同时数据传输是双向的，所以并发。

  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f2a47d8420e149b79d8e2d7e2f9bc445~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



  **resource constrain**

   * 在使用超量queue，开启32个work queue，当总stream < 32的情况下，依旧可能存在stream不并发的可能。

     * 这是因为每个kernel使用了过多的resource，gpu resource有限，每个kernel过多的resource，导致gpu resource不能同时支持多个kernel。

   * 下图是开启过多kernel，导致gpu resource不够，所以无法实现stream并发。

  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/932d0dc05e144990a136ab864aeb2d00~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




  **硬件能力**

   * Fermi 支持最高 十六路的stream并发，最大 16个 grids/kernels同时执行。

   * Kepler 支持最高 32路的stream并发



#### Stream工作原理

  * host stream (software) 可以无限
  * host stream 通过hardware work queue将操作发送给device
    * 单hardware work queue 会导致false dependency
    * 多hardware work queue会减轻false dependency问题，但仍然存在该问题
  * hardware work queue 将任务发送给copy engine或kernel engine
  * Engine 将任务分发给PCIe
    * 任务将并行运行

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/119bb816cd1a44a987567af80b90482a~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

   





#### Fermi GPU & False Dependency

##### 特点

   **硬件结构**

   * 在host和device之间只有一个hardware work queue。
   * 在device上的streams最终都是映射到一个hardware work queue上。

   **运行过程**

   * 在work queue的 queue.front()的task，只有前面全部的dependency task都运行完了，才可以被schedule运行。不在queue.front()的task不会被schedule

   **False Dependency**
   
   * single hardware queue 会导致false dependency。
   * 下面的例子中host stream被serialize的放到single hardware work queue中（A-B-C 接着 P-Q-R 接着 X-Y-Z）。hardware work queue的 queue.front()的task只有前面依赖的task都运行完才可以运行。
     * 当B在front的时候，需要等待A运行结束才可以被schedule，这个时候由于P还在queue中，不在queue front所以无法被schedule。
     * 当C在front的时候，需要等待B运行结束后才可以被schedule
     * 当C被schedule了，queue front是P，发现与C没有dependency关系，所以也可以立刻被schedule。
   *  这里在A schedule以后，本来可以scheduleP，但是因为stream是serialize的放到single hardware queue中，所以无法schedule P，所以产生了false dependency。
      * 虽然kernel engine同时可以管理16个grid,但由于A B C 存在dependency且task queue只有一个，没有运行到C的时候scheduler是看不到还有P这个可以分开执行的task。

  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/10c574bb92f84ead96b9a86c9187abbe~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

  ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3fa5aa9092db448e8e8fd6781572c258~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)





   **解决**

   * Fermi使用breath-first / depth-first 需要case by case的分析，不存在某一个一定就一直好的。





##### Breath-First(BFS)

   **定义**
    对每个stream都先执行第一个操作。



   **例子1**
    通过使用breath-first的方法，消除false dependency

    需要注意的是，这里使用了breath first让性能变好，只是针对于本程序。对于Fermi arch，有时候breath好，有时候depth好。

   ```cpp
   // dispatch job with breadth first way 
   for (int i = 0; i < n_streams; i++)
    kernel_1<<<grid, block, 0, streams[i]>>>(); 
   for (int i = 0; i < n_streams; i++)
    kernel_2<<<grid, block, 0, streams[i]>>>(); 
   for (int i = 0; i < n_streams; i++)
    kernel_3<<<grid, block, 0, streams[i]>>>(); 
   for (int i = 0; i < n_streams; i++)
    kernel_4<<<grid, block, 0, streams[i]>>>();
   ```

   ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d7b4391f180c4d36b5a69fb6cf122aa1~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)
  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/fd3844eb8ff5471fbcfe577fc18a90d5~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

   



   **例子2**：

   ```cpp
   for (int i=0; i<n; i+=SegSize*2) {
    cudaMemcpyAsync(d_A1, h_A+i, SegSize*sizeof(float),.., stream0);
    cudaMemcpyAsync(d_B1, h_B+i, SegSize*sizeof(float),.., stream0);
    cudaMemcpyAsync(d_A2, h_A+i+SegSize, SegSize*sizeof(float),.., stream1);
    cudaMemcpyAsync(d_B2, h_B+i+SegSize, SegSize*sizeof(float),.., stream1);
  
    vecAdd<<<SegSize/256, 256, 0, stream0>>>(d_A0, d_B0, ...);
    cudaMemcpyAsync(h_C+i, d_C1, SegSize*sizeof(float),.., stream0);
    vecAdd<<<SegSize/256, 256, 0, stream1>>>(d_A1, d_B1, ...);
    cudaMemcpyAsync(h_C+i+SegSize, d_C2, SegSize*sizeof(float),.., stream1);
   }
   ```

   * 启用两层处理方式用以解决循环之间的false dependecy。其中上述代码进行了reorder，依旧保证stream内部的顺序，但是把copy A.1 B.2放到copy C.0之前，从而避免了A.1 B.1需要等待C.0 & kernel 0。其中for-loop对应为i，其对应到h_A+i当i=0 为A.0

  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/42de2ae93e36426caa6f066258892fdc~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)
  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/93c9ab2cc385499aaf6f55ddaed230f4~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




##### Depth-First(DFS)

   **定义**
    在进入下一个stream之前，完成当前stream的全部操作



   **例子1**：
    使用了4个stream，运行4个comp kernel，但是由于single hw work queue导致false dependency，只有stream末尾与stream+1的开头可以overlap

   ```cpp
   dim3 block(1);
   dim3 grid(1);
   for (int i = 0; i < n_streams; i++) {
    kernel_1<<<grid, block, 0, streams[i]>>>(); 
    kernel_2<<<grid, block, 0, streams[i]>>>();
    kernel_3<<<grid, block, 0, streams[i]>>>();
    kernel_4<<<grid, block, 0, streams[i]>>>(); 
   }
   ```
  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6c30f184ac8545189582023b465122d5~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)
  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/db952a3ca1704dc5b0bfc679b36f529b~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




   **例子2**

   下面的例子里，copy A.2 copy B.2本可以运行，但是因为前面有copy C.1在queue中，copy C.1对kernel eng有依赖，所以只有kernel1运行结束，copy C.1运行结束，才可以运行copy A.2 B.2

   ```cpp
   for (int i=0; i<n; i+=SegSize*2) {
    cudaMemcpyAsync(d_A1, h_A+i, SegSize*sizeof(float),.., stream0);
    cudaMemcpyAsync(d_B1, h_B+i, SegSize*sizeof(float),.., stream0);
    vecAdd<<<SegSize/256, 256, 0, stream0>>>(d_A0, d_B0, ...);
    cudaMemcpyAsync(h_C+i, d_C1, SegSize*sizeof(float),.., stream0);

    cudaMemcpyAsync(d_A2, h_A+i+SegSize; SegSize*sizeof(float),.., stream1);
    cudaMemcpyAsync(d_B2, h_B+i+SegSize; SegSize*sizeof(float),.., stream1);
    vecAdd<<<SegSize/256, 256, 0, stream1>>>(d_A1, d_B1, ...);
    cudaMemcpyAsync(h_C+i+SegSize, d_C2, SegSize*sizeof(float),.., stream1);
   }
   ```
   ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/42de2ae93e36426caa6f066258892fdc~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)
  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d4a4975dbce645aeadb114babbe5cf75~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)






#### Kepler GPU & Hyper Queue

##### 特点

   * 在host和device之间有多个hardware work queue。
   * 在Kepler中，有32个hardware work queues。并且在每个stream中分配一个work。 
     * 如果host stream的个数超过32，则会序列化为一个hardware work queue。
     * 现在每个stream都有一个hardware work queue，所以可以做到max 32 stream level concurrency

   * 但依旧可能存在false dependency。
     * 如果num stream > num active hardware queue，则会被序列化为一个hardware work queue

  ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8bc9126b13e64d949c0244555aed05a0~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/99f0cbd8ccc04c71beddbdc3626415d9~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



##### GMU

   **目的**
    为了进一步解决false dependency，除了有32hardware working queue外，还有GMU帮助解决false dependency
   **与Fermi的不同**
    在Fermi中，grids从单queue中直接传送到CUDA Work Distributor (CWD)。
    在Kepler中，grids被送到GMU，GMU负责管理和决定谁的优先级高。
    有GMU来分析grid dependency, 也能够一定程度的eliminate false dependency



##### 配置 hw queue
   **如何配置**
    尽管超queue中有最多32个，但默认只开启8个hw queue，因为每个开启的hw queue都会占用资源。
   **API**
    可以通过下面的API来设定一个程序的max connection有多少

   ```cpp
   //shell
   setenv CUDA_DEVICE_MAX_CONNECTIONS 32

   // c code
   setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);
   ```



   **配置**
    下图的例子里使用了8个stream，但是只开启了4个hw queue，导致false dependency依旧存在。

  ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/35371d7c77bb4227aa481f2bbc74c7d6~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


   当开启8个hw queue的时候，就不存在false dependency了

   当使用BFS的时候（只有4个hw queue），也可以做到不存在false dependency

  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d7805705995a4959bbe7e6f72009a495~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




##### multi-queue中Depth-First性能影响
   对于kepler device来说，因为有了GMU的帮助，使用depth-first / breath first对整体的影响不大。



   **例子**
    与Fermi depth-first 例子1相同

  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/bdd7613ebcf14a4789b336d106709546~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)






#### Stream Priority
  **应用局限**
   * 从cc3.5之后stream可以配置优先级。
   * 应用在stream的全部计算kernel中，不对copy kernel起作用

  **设置**
   * 如果超过了范围的话，则会转到最大或最小
   * 更低的值代表更大的优先级

  **API**
   ```cpp
   cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority);

   cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
   ```



#### Stream callback
  一旦stream的所有操作完成后，host侧由stream callback指定的函数会被CUDA runtime调用。

  ```cpp
  cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int flags);
  ```
  ```cpp
  for(int i = 0; i < n_streams; i++){
    stream_ids[i] = i; 
    kernel_1<<<grid, block, 0, streams[i]>>>(); 
    kernel_2<<<grid, block, 0, streams[i]>>>(); 
    kernel_3<<<grid, block, 0, streams[i]>>>(); 
    kernel_4<<<grid, block, 0, streams[i]>>>(); 
    cudaStreamAddCallback(streams[i], my_callback, (void *)(stream_ids + i), 0);
  }
  ```
  my_callback 会block 其他stream中的operations，只有当callback运行结束以后，其他streams才可以继续运行。



  **限制**
   * 不可以在callback中调用CUDA API
   * 不可以在callback中调用sync。更深层次的是callback不可以用于管理stream中operation的order

### stream 同步

#### Named stream 和 null stream

##### Host

   named stream不会block host代码的执行

   null stream会block host代码的执行，除非是kernel是async。

##### Blocking stream 和 non-blocking stream

   Named stream分为两种。
   blocking stream 和 non-blocking stream



   **Blocking stream**

   * 使用API`cudaStreamCreate()`创建的stream是 blocking streams。

   * block stream需要等待在其前面的所有NULL stream执行完成才可运行execution of opera- tions in those streams can be blocked waiting for earlier operations in the NULL stream to complete. blocking stream上的operation，会等待null stream上在它前面被分配的operation运行结束后才会运行。

   * 当执行NULL stream的时候，会等待前面的所有block stream，在完成之后，才会执行NULL stream。



   **例子1**

   * kernel 2是null stream，会等到blocking stream 1上的kernel 1运行结束后才会运行，因为kernel 1在kernel2之前被assign

   * kernel 3是blocking stream，会等待null stream上kernel 2运行结束后才会运行，因为kernel 2在kernel 3之前被assign

   * 尽管这个例子里使用了两个stream，但是由于他们是blocking stream + 使用了null stream，导致三个kernel实际上是serialize运行的

   ```cpp
   // stream 1, 2 are blocking stream
   kernel_1<<<1, 1, 0, stream_1>>>(); 
   kernel_2<<<1, 1>>>(); 
   kernel_3<<<1, 1, 0, stream_2>>>();
   ```



   **例子2**

   * Block kernel会阻塞其他stream的执行

   ```cpp
   // dispatch job with depth first ordering 
   for (int i = 0; i < n_streams; i++) {
    kernel_1<<<grid, block, 0, streams[i]>>>(); 
    kernel_2<<<grid, block, 0, streams[i]>>>(); 
    kernel_3<<<grid, block>>>(); 
    kernel_4<<<grid, block, 0, streams[i]>>>();
   }
   ```
  ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b800d7ac146141b6963252115c369313~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



   **non-blocking stream**

   * stream created using 下面的API并且set flag是non-blocking stream

   * non-blocking stream不与null stream进行synchronize

   ```cpp
   cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);

   cudaStreamDefault: default stream creation flag (blocking)
   cudaStreamNonBlocking: asynchronous stream creation flag (non-blocking)
   ```

#### 显式同步

##### Device

   * 会将CPU端线程暂停，直到device端完成所有计算和迁移

   ```cpp
   cudaError_t cudaDeviceSynchronize(void);
   ```

##### Stream

   * 会阻断CPU线程的运行，直到指定的stream的运算完成

   ```cpp
   cudaError_t cudaStreamSynchronize(cudaStream_t stream); 
   cudaError_t cudaStreamQuery(cudaStream_t stream);
   ```

##### stream中的event

   * 会阻断CPU线程的运行，直到指定的stream上的event的运算完成

   ```cpp
   cudaError_t cudaEventSynchronize(cudaEvent_t event); 
   cudaError_t cudaEventQuery(cudaEvent_t event);
   ```

##### Sync across stream using event

   * 让当前stream等待某个event的完成，这个event可以不是当前stream的event

   * 用法：某个stream要等待其余stream运行到某个步骤，才可以开始运行。这时候就可以在其他stream使用event，让某个stream等待其他stream的event。
   * 当前函数对象需要等待参数event完成，之后才能运行参数stream

   ```cpp
   cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event);
   ```



   **例子**

   * cudaStreamWaitEvent 用于to force the last stream (that is, streams[n_streams-1]) to wait for all other streams

   ```cpp
   // dispatch job with depth first way 
   for (int i=0; i<n_streams; i++) {
    kernel_1<<<grid, block, 0, streams[i]>>>(); 
    kernel_2<<<grid, block, 0, streams[i]>>>(); 
    kernel_3<<<grid, block, 0, streams[i]>>>(); 
    kernel_4<<<grid, block, 0, streams[i]>>>();
    cudaEventRecord(kernelEvent[i], streams[i]);
    cudaStreamWaitEvent(streams[n_streams-1], kernelEvent[i], 0); 
   }
   ```

  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a727b052ba494d178f338a6418adbae5~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


#### Events

  **API**

  ```cpp
  cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);


  cudaEventDefault // host会时不时的check event是否完成，占用cpu资源
  cudaEventBlockingSync // 让host gives up the core it is running on to another thread or process by going to sleep until the event is satisfied. 当host resource有限的时候可以通过这个方法减少对host的占用。但是可能让event finish latency增加，因为需要等待thread重新从ready到execute
  cudaEventDisableTiming // 该event不用于timing，可以让cudaStreamWaitEvent & cudaEventQuery速度更快
  cudaEventInterprocess // event 用于inter-process
  ```

### Pinned Memory

#### 同步拷贝

  **内存拷贝**

   * host data是默认分页存储，而device并不能够直接读取，读取通过在host的一段array，又称pinned或页锁定内存，用于将host数据先送给pinned memory，然后再送给device。在此过程中，是从CPU的memory->memory，这个过程会经过CPU core，导致内存受到限制。（CMU的最新arch研究关于如何从cpu mem直接拷贝的mem，不经过cpu）

   * 如果内存不是pinned的，则访问的时候对应的内存可能在disk/ssd上，需要经过CPU进行page swap，拷贝到临时的pinned memory，再使用DMA从临时pinned memory拷贝到device global memory上

   ```cpp
   int *h_a = (int*)malloc(bytes);
   memset(h_a, 0, bytes);

   int *d_a;
   cudaMalloc((int**)&d_a, bytes);
   // synchronize copy
   cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
   ```

#### 异步拷贝

  **原因**

   * 为了避免CPU操作从pageable到pinned memory的数据操作

   * 将数据传输与计算重叠

  **特点**

   * 更高带宽
      * pinned内存可以直接使用DMA拷贝到GPU，不需要经过CPU，从而有更大的bandwidth。

  **使用注意**
   * 因为pinned内存是有限的资源，分配pinned内存可能会失败，所以一定要检查是否有失败
   * 不要过度使用pinned memory，这会导致系统整体速度变慢

  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c7b2677b01174e3bb814aa93cc1ed69c~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  **例子**

   ```cpp
   int *h_aPinned, d_a;
   // malloc()
   cudaMallocHost((int**)&h_aPinned, bytes);
   memset(h_aPinned, 0, bytes);
   // cudaMalloc()
   cudaMalloc((void**)&d_a, bytes);

   // synchronize copy
   // cudaMemcpy()
   cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

   // pin memory on the fly without need to allocate seprate buffer
   cudaHostRegister()
   ```

##### API

   * `cudHostAlloc()` and `cudaFreeHost` 分配pinned memory

   * `cudaHostRegister()` 将device的一段内存设置为页锁定内存

##### Portable Memory


   **目的**
   为了让所有device都能看到memory。pinned memory只能让指定device看到。

   **方法**
   * passing the flag cudaHostAllocPortable to cudaHostAlloc() 
   * passing the flag cudaHostRegisterPortable to cudaHostRegister().

##### Write-combining memory

   **目的**
   * 默认page lock host memory是cachable，可以read+write。
   * free up host L1 L2
   * 增加host to device memory transfer的带宽

   **方法**
   passing flag cudaHostAllocWriteCombined to cudaHostAlloc()。


   **限制**
   只能write。如果read from host的话会非常的慢

### 内存计算和拷贝重叠

  **查看是否支持**

   ```cpp
   cudaDevicePeop dev_prop;
   cudaGetDeviceProperties(&dev_prop, 0);

   // 是否支持async compute & memory copy 
   // 1: 支持 1 copy + 1 exec
   // 2: 支持 1 copy host2device, 1 copy dev2host, 2 exec
   dev_prop.asyncEngineCount; 
   ```

  **例子**

   ```cpp
   size=N*sizeof(float)/nStreams;
   for (i=0; i<nStreams; i++) 
   {
    offset = i*N/nStreams;
    cudaMemcpyAsync(a_d+offset, a_h+offset, size, dir, stream[i]);
    kernel<<<N/(nThreads*nStreams), nThreads, 0, stream[i]>>>(a_d+offset);
   }

   ```

  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/edebea7e56f0425fb515742101f390bc~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)








## 动态并行



 **是什么**

  * GPU kernel只能从CPU启动
  * 从Kepler开始，GPU Kernel也可以启动GPU Kernel，从而允许更多的编程flexible
    * 是通过改变GPU hardware和runtime library实现的dynamic parallle的支持

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/887dfb670d8b422eb01bcc33ff757518~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



 **例子**
   * 没有使用动态并行方法
  ```cpp
  __global__ void kernel(unsigned int* start, unsigned int* end, float* someData, float* moreData){
    unsigned int i = blockId.x * blockDim.x + threadId.x;
    doSomeWork(someData[i]);

    for(unsigned int j = start[i]; j< end[i]; j++){
      doMoreWork(moreData[j]);
    }
  }
  ```
   * 使用动态并行方法
  ```cpp
  __global__ void kernel_parent(unsigned int* start, unsigned int* end, float* someData, float* moreData){
    unsigned int i = blockId.x * blockDim.x + threadId.x;
    doSomeWork(someData[i]);
    kernel_child<<<ceil((end[i] - start[i]/ 256))>>> (start[i], end[i], moreData);
  }
  __global__ void kernel_child(unsigned int start, unsigned end, float* moreData){
    unsigned int j = blockId.x * blockDim.x + threadId.x;
    if(j <end){
      doMoreWork(moreData[j]);
    }
  }
  ```

### 好处和坏处

 **好处**

  * 有更多的并行。之前一个thread做多个工作，现在多个工作被打包成新的kernel来平行运行。
  * 负载平衡。之前一个thread做多个工作的时候，多个工作有imbalance。现在每个thread的工作成为新的kernel，平行运算，允许了更多的balance
  * 减少闲置资源。之前可能存在idle SM，当使用child kernel以后这些idle sm会被使用
  * 减少传输次数。减少了CPU GPU之间kernel parameter传输的次数，从而减少等待kernel launch的时间。
  * 由于CPU速度慢，无法在单位时间内启动足够多的kernel给gpu来计算，导致GPU underutilize。
    * 当有了dynamic paralle以后，可以在device code上 call library function，避免了CPU速度慢无法满足GPU工作需要的问题。
  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3358e76bb3d24765aba6df1374694366~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  
 **wrapper kernel**

  负责在device 上启动library kernel的kernel叫做wrapper kernel。

  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b13c5a85dcbb470b8f3cbba73ce4c9e8~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/9def35bd09d24defbf7bd03dfd933671~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)





 **坏处**

  * 带来索引计算。希望memory access可以hide这部分的latency
  * kernel launch的时间开销。launch kernel is 100 - 1000 cycles
  * 如果child kernel过小，使用很少的线程，对GPU资源使用不足，一般child kernel需要在千为单位，跨多个block

### Memory & Stream

#### Global memory

  **CUDA内存一致性特点**

   * 全局内存一致性只能在kernel结束的时候保证。
     * 因为针对全局内存使用cache技术。将变量存储在cache，需要将其写入才能保证全局内存一致性 
     * 被一个kernel / block / thread 所更改的数据，可能对其他的 kernel / block / thread 不可见。

  * 例子：block 0 写入global memory location，block 1 读取相同的global memory location，block 1 不保证能读取到block 0写入的数据。

  **解决方法**
   使用memory fence function来解决这个问题



  **动态并行在Global memory解决方法**

   * 保证两种parent child的global memory consistency

     * 在调用child kernel之前，parent thread的所有全局内存造作对child kernel可见，不管parent thread是否处于运行状态。
     * chile kernel执行后对parent 的这个thread可见


   * 需要注意这两种consistency都是针对parent kernel单独thread的，而不是针对整个parent thread block的。



  **例子**


   ```cpp
   __global__ void parent_launch(int *data) {
    data[threadIdx.x] = threadIdx.x;

    if (threadIdx.x == 0) {
        child_launch<<< 1, 256 >>>(data);
        cudaDeviceSynchronize();
    }
    
   }
   ```

   * child kernel只能保证看到 data[0] = 0, 并不能保证看到 data[1] = 1, data[2] = 2, data[3] = 3

   * parent kernel只有thread 0保证能看到child kernel对内存的更改，parent kernel thread 1,2,3不保证看到child kernel对global memory的更改


  **按照block进行动态并行**

   ```cpp
   __global__ void parent_launch(int *data) {
    data[threadIdx.x] = threadIdx.x;

    // ensure child kernel see all parent threads block memory access
    __syncthreads();
    
    if (threadIdx.x == 0) {
        child_launch<<< 1, 256 >>>(data);
        cudaDeviceSynchronize();
    }
  
    __syncthreads();
   }
   ```
   * 通过使用syncthread within block的方法，

   * 保证child kernel可以看到parent threads block内全部对global memory的操作

   * 保证parent thread 1,2,3能看到child kernel对global memory的更改

#### Constant memory

  只能在host设定，在parent kernel上无法改变

#### Local memory

  **特点**

   * local memory是thread的私有变量，不能在child kernel上使用

  * 为了避免把local memory传入child kernel，传入child kernel的ptr必须是显式从global memory分配的空间



  **API**

   ```cpp
   // ok
   __device__ int value;
   __device__ void x()
   {
     child<<1, 1>>(&value);
   }

   // illegle
   __device__ void x()
   {
     // value is on local storage
     int value[100];
     child<<1, 1>>(&value);
   }
   ```



  **安全检查**

   * 可以在runtime的时候使用 `__isGlobal()` intrinsic来检查某pointer是指向global memory的，从而决定参数传递是否安全

#### Shared memory

  * shared memory 不可以与child kernel共享。

    * child kernel可能运行在任意一个SM上，所以共享某个SM对应的shared memory就是没意义的。

#### Memory allocation

  * device kernel可以调用 cudaMalloc, cudaFree来分配global memory。

  * device kernel上能分配的global memory可能比实际上的global memory要小。因为host和device上分配内存走了两个逻辑
    * device kernel上实现memory allocation：通过对thread进行scan，计算block的所有threads需要的内存空间，合并为cuda malloc call指令，然后将大的空间分配给多个threads来执行。
  * device kernel上分配的global memory，只能使用device kernel 来释放，可以不用是同一个kernel。

#### Stream

  **stream特点**

   * stream的scope是创建stream所在的block。

   * 由threads创建的stream可以被同block的其他threads使用。

   * 如果parent kernel有两个threads block，则每个 block各有自己的null stream，一共有两个null stream

   * stream不可以传递给child kernel使用。
     * 原因：在上一级创建的streams在本级不能用。从host创建的stream，在kernel中无法使用，同理，在parent kernel创建的stream不能在child stream中使用。

   * 用户可以启动无限的named stream，但是hw只支持有限的concurrency。如果用户启动的stream > max concurrency的话，runtime可能选择alias / serialize stream来满足max concurrency
   
   * 可以使用多个stream来达到concurrency，但是runtime由于hw的限制不保证不同的stream的kernel一定是concurrent的

  **stream使用选择**

   * 当GPU 性能未完全利用的时候，使用named stream的效果最好。当GPU 已经完全利用硬件，使用default stream。

   * 下图当x轴向右，GPU充分被使用，default stream 和 named stream 的差异就不大了

  ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/9937e76aeaaf419888e4e2079773fbae~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

  **null stream与动态并行**

   * null stream是block scope的，如果启动child kernel的时候没有使用named stream，则默认使用null stream，则所有child kernel的都会使用parent stream的fault stream，产生序列化。

   * 图片左边是使用default stream，右边是使用per thread named stream

  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3ea30b8b174c4ca199830bf74cc3f4fd~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  **配置**

   * 编译的时候可以使用这个flag，从而默认使用per thread named stream，而不是default null stream

   ```shell
   --default-stream per-thread
   ```

  **stream的synchronize与动态并行**

   * synchronize的时间：当block内的thread退出的时候，在block内创建的streams会隐性同步。
     * 也就是block结束的时候，所有由threads创建的block会隐性同步

   * cudaStreamSynchronize()不可用
   * cudaDeviceSynchronize()可用，可以用于显式等待work完成。

  **动态并行中的kernel的destruction**

   * 等到stream的所有算数完成。

  **API**

   ```cpp
   cudaStreamCreateWithFlags( cudaStreamNonBlocking ) ;
   ```

#### Event

  **特点**

   * event在对block内所有的threads都可见。
   * child kernel不可见

   ***<font color =purple>针对某一个特定的stream的event是不支持的。</font>***

   ***<font color =purple>cudaEventSynchronize(), timing with cudaEventElapsedTime(), and event query via cudaEventQuery() are not supported on device</font>***

### 同步

#### 启动流程

  * 分配全局内存空间存储child kernel参数

  * 由SM发送命令到达KUM(kernel management unit)的启动池

  * 当kernel到达启动池顶端将kernel从KMU调度到调度器

  * 在一些SMs中运行kernel

#### Running

  * 所有的child kernel相对于启动thread来说都是异步。

  * child kennel是独立于parent kernel，只有在parent kernel sync child kernel的时候，才能保证child kernel是开始运行的。 

  * child kernel只能在当前device上运行，不可能跨device运行

  * 和host代码相似，launch不代表运行，只有parent sync的时候才一定保证child的运行

#### Explicit Parent Child Sync

  **特点**

   * 从device启动的kernel都是non-blocking。

   * 调用 `cudaDeviceSynchronize()`的thread会让host等到block内线程开启的所有kenrels都运行完成之后才继续执行。

   * 如果只针对某个thread发出的kernel，如果想让整个thread block都等待child kernel运行结束，还需要使用`__syncthreads()`
  
   * parent block中任何一个thread都可以sync 由parent block启动的child block



  **block wide sync**

   * 如果block内的thread都有 child kernel，并且显性sync child kernel。需要确保block内全部的child kernel 都会被调用，在此之后执行所有threads同步函数`cudaDeviceSynchronize()`



   下面的代码可能会出现launch child kernel只被部分部分thread运行，导致launch pool中只有部分child kernel。然后就有thread运行cudaDeviceSync，导致launch pool中的kernel被drain out。

   ```cpp
   __device__ void func()
   {
    child_kernel<<1, 1>>();
    
    cudaDeviceSync();
   }
   ```



   正确的做法是使用syncthread从而确保全部的child kernel都被launch了

   ```cpp
   __device__ void func()
   {
    child_kernel<<1, 1>>();
    
    __syncthread();
    cudaDeviceSync();
   }
   ```

#### Implicit Parent Child Sync

  * 如果没有使用显式同步，CUDA runtime会保证隐式同步保证没有parent kernel在child kernel结束之前结束。
  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/58e65bd02ba94406bbf5e91d89ed6c9b~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)
  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4dbe9e46a47f45ef957148b31d789af7~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


### 启动配置

#### GPU启动前

  * kernel和memory管理的函数都是由CPU driver code执行。

  * 内存分配需要CPU的支持

#### 启动环境配置

  * 所有全局设备配置设置（例如，从 `cudaDeviceGetCacheConfig()`返回的共享内存和L1缓存大小，以及从 `cudaDeviceGetLimit()`返回的设备限制）将从parent kernel继承。

#### 嵌套深度

  * 支持最大24 nested depth。

#### 同步深度

  **locked down parent state**

   * 一般来说，parent kernel resources (registers, shared memory) 对child kernel不可见。

     * 因为saving and restoring parent resource是复杂的

  **swap out parent state**

   * 当显性sync的时候，parent kernel可能会被swap out execution

     * swap out execution使用memory储存parent kernel state。

     * 需要有足够的memory来储存max num of thread的kernel state（而不是实际上使用的thread）限制了sync depth。sync的default=2

#### Launch Pool

  **特点**

   * launch pool用于监控正在执行的kernel和等待的kernel。

   * 有两种launch pool。
     * finite pending launch buffer 2048
     * 如果超过了的话则使用virtualized extended pending launch buffer

   * 在CUDA 6之前，如果溢出到hardware launch pool的话，会导致runtime error

   * 可以通过API设定virtualized extended pending launch buffer，将其溢出到software queue中

  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f81cd7c33efd43f98c8ea7beda772e03~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  **API**

   ```cpp
   // query
   cudaDeviceGetLimit();

   // setting
   // 改变fixed size launch pool的大小，从而避免使用virtualize pool
   cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, N_LINES);
   ```

#### Error

  * child kernel返回的error是per parent thread based的

  * 对于device的异常，例如地址错误、child kernel的grid错误，错误将返回host侧，而不是由parent kernel的cudaDeviceSynchronize()调用返回。

#### 启动错误

  如果resource不够的话，cuda runtime会error

### Aggregation


 **定义**

  * aggregate 将众多由block的threads发出的child kernels合并到一个更大的child kernel，然后将这个大的child kernel进行launch。

 **作用**

  * 减少kernel launch开销

  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f3559f12735d4a5e99e653bdbde70d1a~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


 **假设**

  * 假设child kernel的代码是一样的，唯一不一样的是负责计算不同部分的data

 **启动**

  * 储存args into args array

  * 储存 gd, bd

  * new gd = sum of gd，new bd = max of bd
    * 确保全部的kernel launch have enough thread

  * 选择thread 0启动aggregated launch

  ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/dca0ec300eaa4311bf4cbe8997beed3e~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


 **child kernel内使用**

  计算出没有aggregate之前parent thread idx （红色）

  计算出没有aggregate之前parent block idx（黄色）

  使用计算出来的信息，运行kernel
  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/42236dc343714794b4f14cda47d7e06d~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




  parent thread idx 和 parent block idx 可以使用二分法查找

  ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b2f378d7104244d69866bd5a5c64d5ee~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)








## Cooperative Groups

 **定义**

  * 原来的sync是within block的
  * 从CUDA9开始，现在支持自定义thread group，可以是smaller than block，也可以是across block，甚至across gpu。
  * group内的thread可以进行synchonize

![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c6185d93ba8d4459aa1cd01477e59777~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

 **问题**

  不确定效率上来讲是好还是坏
