---
layout: article
title: GPU Shared Memory Model
key: 100014
tags: GPU SharedMemory
category: blog
date: 2023-02-08 00:00:00 +08:00
mermaid: true
---


## Shared memory

### Basic


 **与global memory 区别**

  * 为<font color = red>片上内存，SRAM实现</font>
  * SRAM 支持<font color = red>随机存储</font>，不需要global memory 的burst 存取
  * <font color = red>延迟</font>比global memory低20/30倍，<font color = red>带宽大</font>10倍
  * shared memory不管是否有没有bank conflict问题，都比global memory要快很多
  * 有些时候为了避免bank conflict从而做了很多优化，但是由于额外的增加intrinsic，导致perf反而变差



 **什么时候使用**

  * 数据有复用的话，考虑使用shared memory



 **使用时需要注意**

  * 使用shared memory<font color= red>解决了global memory中的对齐和合并以及非原子操作的顺序问题</font>，但存在<font color= red>bank conflict</font>的问题，使用时要注意<font color= red>不要忘记synchronize的使用</font>
  * 从Volta开始，warp内部不是lock step的，所以<font color= red>warp内部使用shared memory有时候也需要memory fence</font>
  * shared memory使用后，应当<font color= red>考虑一个sm能有多少个thread和block</font>



 **从global memory到shared memory的加载过程**

   * <font color=red>global memory -> cache (optional L1)L2 -> per thread register -> shared memory</font>

   * 不经过register的方法(CUDA11有async，<font color=red>不经过register的方法</font>，see below section)

 **使用shared memory用于计算**

   * 首先<font color=red>从shared memory </font>读取数据，放到<font color=red>tmp register </font>上，然后在<font color=red>register 上计算</font>，最终再把结果从register 放到<font color=red>对应目的地</font>。

   * 计算中的<font color=red>latency</font>来自从shared memory到register

 **shared memory & occupancy**

   * <font color=red>occupancy:</font> 每个SM的活动wrap与最大可能活动warp的比值，其表征了系统性能。
   * shared memory 提高可能对occupancy产生<font color=red>负面影响</font>(限制SM的occupancy的因素包括<font color=red>warp个数、block个数、register个数、shared memory数量、block中register & shared memory大小</font>)
     * 在很多现实场景中，不再采用shared memory与threads一对一的关系，一个shared memory大小为64\*64，block最大thread为1024，则采用 32\*32的thread个数，然后<font color=red>一个thread处理4个elements</font>，该种方法性能同样可以提高。
     * <font color=red>评估</font>occupancy对性能的影响程度通过动态分配shared 来实验，在配置的第三个参数中指定，增加此参数，可以有效减少内核占用率，测量对性能的影响。

 **occupancy**
  * 每个SM的活动wrap与最大可能活动warp的比值
  * 线程块大小：较小的线程块可以更好地利用 GPU 的资源，因为 GPU 上的资源是按线程块来分配的。通常来说，**线程块大小越小**，GPU 可用资源的利用率就越高。
  * 寄存器使用：每个线程块在运行时需要使用一定数量的寄存器来保存中间结果和变量。寄存器的数量是有限的，当**线程块使用的寄存器数量过多**时，就会导致 GPU 上可以同时运行的线程块数量减少，从而降低 GPU occupancy。
  * 共享内存使用：共享内存是一种高速缓存，能够在线程块内部共享数据。但是，共享内存的大小也是有限的，当**线程块使用的共享内存过多**时，就会导致 GPU 上可以同时运行的线程块数量减少，从而降低 GPU occupancy。
  * 全局内存访问：全局内存是 GPU 上的主要存储区域，但是**访问全局内存需要消耗大量的时间和资源**。当线程块频繁地访问全局内存时（**需要等待**），就会导致 GPU 上可以同时运行的线程块数量减少，从而降低 GPU occupancy。
  * 硬件限制：每个 GPU 都有自己的硬件限制，包括**寄存器数量**、**共享内存**大小、**带宽**等等。当线程块使用的资源超过了硬件的限制时，就会导致 GPU 上可以同时运行的线程块数量减少，从而降低 GPU occupancy。

#### API

  **dynamic use**

   * 只分配<font color=red>1D</font>

   * 比起使用static shared memory，dynamic shared memory有一些小的overhead。

   ```cpp
   extern __shared__ int tile[];

   MyKernel<<<blocksPerGrid, threadsPerBlock, isize*sizeof(int)>>>(...);
   ```



  **static use**

   * <font color=red>1/2/3D</font>

   ```cpp
   __shared__ float a[size_x][size_y];
   ```



  **shared memory位置**
   * cc 2.x、cc 3.x、cc 7.x、cc 8.x、cc 9.x、L1和shared memory一块配置
   * cc 5.x、cc 6.x L1和 texture memory 一块配置，shared memory有自己的空间


  **shared memory配置**

   * L1 + Shared 一共有64 kb memory (在某些device上)

     * shared memory使用32 bank访问。L1 cache使用cache line来访问。

     * 如果kernel使用<font color = red>很多shared memory</font>，prefer larger shared memory

    * 如果kernel使用<font color = red>很多register</font>，prefer larger L1 cache。因为register会spilling to L1 cache



  **shared memory配置 API**

   * 设置device
   ```cpp
   cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);
   ```
   * 设置可选项

   ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_26.png)
   

   如果当前kernel的setting与前一个kernel的不一样，可能会导致implicit sync with device

   * 设置func
   ```cpp
   cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCacheca cheConfig);
   ```

   * 如果当前内核的设置与前一内核不一样，可能会导致implicit sync with device

### Memory Bank




#### Shared memory & transaction

  * shared memory 与global memory 类似，<font color= red>以warp 为单位</font>进行读写。warp 内的多个thread 首先会合并为一个或多个transaction，然后访问shared memory。增加数据的利用率，减少总访问次数。
    * <font color= red>最好的情况</font>只会引发一次transaction。<font color= red>最坏的情况</font>引发32次transaction。

    * 如果都<font color= red>访问同一地址</font>，则只产生一次transaction，然后broadcast到每一个threads。同样的broadcast在使用constant cache，warp level shuffle的时候都有。

   * 在warp操作shared memory中，如果<font color= red>load/store指令在每个bank只有一块数据，且数据小于bank带宽</font>，则可充分利用每个bank自己的带宽。
     * <font color= red>最好的使用</font>shared memory的办法就是确保请求的数据分布在每个bank中，每个bank充分的利用bank自己的bandwidth



#### Memory bank & bank conflict

  **特点**
   * shared memory<font color= red>底层被切分为多个banks 来使用</font>。可以同时访问使用多个banks，shared memory<font color= red>带宽为bank的n倍</font>。
   * 对于shared memory的多个访问<font color= red>如果都落到一个bank中不是same word</font>，则request会被replay。将一次request转化为serialized requests。(图 21 中间)。花费的时间：num replay * one bank free time
   * 如果多个thread的内存请求到<font color= red>相同bank的same word</font>，访问不会serialize。如果是read则会broadcast，如果是write则只要一个thread写，但是哪个thread写不确定。bandwidth的使用依旧很低，因为num banks * bank bandwidth这么多的数据只用于传送一个word的数据。


  **三种访问shared memory方法**
   * <font color= red>parallle access</font>: 没有bank conflict
   * <font color= red>serial access</font>: 带有bank conflict触发serialized access
   * <font color= red>broadcast access</font>: 现有单个读当前的bank的word，之后进行broadcast到所有warp内的threads中

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_27.png)
  

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_28.png)

#### Access Mode 32/64-bit

  **定义**
   * <font color = red>shared memory bank width</font>: defines shared memory 地址和shared memory bank 之间的映射关系。

  **分类**
   * <font color = red>32 bits</font> for 除了 cc 3.x之外的版本

   * <font color = red>64-bits</font> for cc 3.x

##### 32 bits

   * <font color = red>一共32 banks，每个bank 32 bits</font>。
   $
   bank \ index=(byte \ address / 4 \ bytes)\% \ 32 \ banks
   $

   * 下图word index 为 bytes address 对应的word index，然后从word index 对应到bank index
   ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_29.png)
   


##### 64 bits


   * 一共32 banks，每个bank 64 bit，word大小由32/64 bits。
   * <font color = red>word大小为64 bits中</font>，
     * 有32 banks，每个bank 64 bits，bytes的分割以8 bytes为单位分隔
     $
     bank \ index=(byte \ address / 8 \ bytes)\% \ 32 \ banks
     $

     * 当<font color = red>thread 查询/写入word子字时，发生boradcast或write undifined</font>

   * <font color = red>word大小为32 bits中</font>，
     * 有32 banks，每个bank 64 bits，bytes的分割以4 bytes为单位分隔。
     $
     bank \ index=(byte \ address / 4 \ bytes)\% \ 32 \ banks
     $
     * 32-bit mode下，访问同一个bank的2个32-bit word不一定产生bank conflict，因为bank width是64，可以把2个word都传送出去。
     * <font color = red>bank conflict的本质：是bank width小，所以无法传送过多的数据</font>
     * <font color = red>当thread 查询/写入word时，发生boradcast或write undifined</font>

     ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_30.png)

  **bank width对性能的影响**
   bank width提高会带来<font color = red>更大的带宽</font>，但是会有<font color = red>更多的bank conflict</font>



  **配置cc 3.x及以上版本的API**

   * 改变Kepler下shared memory bank可能会导致implicit sync with device

   ```cpp
   // query access mode 
   cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig *pConfig);

   cudaSharedMemBankSizeFourByte
   cudaSharedMemBankSizeEightByte

   // setting access mode
   cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);
   cudaSharedMemBankSizeDefault 
   cudaSharedMemBankSizeFourByte 
   cudaSharedMemBankSizeEightByte
   ```



#### Stride access

  **stride access**
   * 对于<font color = red>global memory是waste bandwith</font>，同时<font color = red>解决了对齐问题</font>。
   * 对于shared memory是<font color = red>解决bank conflict的方法</font>。

  **stride在shared memory** 
  <font color = red>stride定义：warp连续thread访问内存的间隔。</font>如果t0访问float array idx0，t1访问float array idx1，则stride of one 32 bit word
   * 不同的stride，对应的conflict类型不同 (假设2.x 32-bit mode)

     * stride of one 32 bit word : conflict free

     * stride of two 32 bit word : 16 x 2-way (2-way表示会被serialize为两个access) bank conflict

     * stride of three 32 bit word : conflict free 

     * stride of 32 32 bit word : 1 x 32-way (32-way表示会被serialize为32个access) bank conflict

   * <font color = red>对于32 bits mode，奇数的stride是conflict free的，偶数的stride是有conflict</font>


#### Avoid bank conflict



  **stride of 32 32 bits word (32-bit mode)**

   * stride of 32 32 bits word 产生 1 x 32-way bank conflict 经常发生在使用shared memory处理<font color = red>2D array of 32 x 32</font>，每个thread负责一个row。这样<font color = red>每个thread对应的row开始都会是在同一个bank中</font>。

   * 解决方法是<font color = red>pad 2d array to size 32 x 33</font>, 这样每个thread负责的一个row的开始都不是在一个bank中 (stride of 33 33 bit word是conflict free的)

  对于padding 64-bit与32-bit mode的方法是不一样。有些在32-bit上是conflict free的，在64-bit上就有conflict了



  **stride of 1 32 bits words**

   满足coarlesed global memory access & shared memory conflict free。

  **global memory noncolaesced 和bank conflict冲突**

   当发生冲突时，如果对shared memory操作更多则<font color = red>先解决bank conflict，然后再考虑noncolaesced</font>，但同样得需要做benchmark

### Data Layout

#### Square Shared Memory
  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_31.png)
  



  ```cpp
  __shared__ int tile[N][N];

  // contigious thread access row of shared memory (contigious)
  tile[threadIdx.y][threadIdx.x];

  // contigious thread access col of shared memory (stride)
  tile[threadIdx.x][threadIdx.y];
  ```



  **padding**

   ```cpp
   __shared__ int tile[BDIMY][BDIMX+1];
   ```



#### Rectangle Shared Memory

  ```cpp
  #define BDIMX 32 // 32 col
  #define BDIMY 16 // 16 row

  dim3 block (BDIMX,BDIMY); 
  dim3 grid (1,1);

  // row-major read (setRowReadRow)
  // the length of the innermost dimension of the shared memory array tile is set to the same dimension as the innermost dimension of the 2D thread block:
  // contigious thread read row of shared memory (contigious)
  __shared__ int tile[BDIMY][BDIMX];


  // col-major read (setColReadCol)
  // the length of the innermost dimension of the shared memory array tile is set to the same dimension as the outermost dimension of the 2D thread block:
  // contigious thread read col of shared memory (stride)
  __shared__ int tile[BDIMX][BDIMY];

  ```



  **举例**
   * 配置

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_32.png)

   * 计算
    word：32 bits，bank 64bits，对列主序来说16*32/64 = 8 bank -> a column are arranged into eight banks.



   * 列主序代码

    ```cpp
    __global__ void setRowReadCol(int *out) { 
      // static shared memory
      __shared__ int tile[BDIMY][BDIMX];
      // mapping from 2D thread index to linear memory
      unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
      // convert idx to transposed coordinate (row, col) 
      unsigned int irow = idx / blockDim.y;
      unsigned int icol = idx % blockDim.y;
      // shared memory store operation 
      tile[threadIdx.y][threadIdx.x] = idx;
      // wait for all threads to complete 
      __syncthreads();
      // shared memory load operation
      out[idx] = tile[icol][irow]; 
    }
    ```



  **padding** 

   * 对于32-bit bank，padding 1

   * 对于64-bit bank，padding取决于data

   * 在上面的例子里，<font color = red>padding 2 element for 64-bit bank</font>

   ```cpp
   #define NPAD 2
   __shared__ int tile[BDIMY][BDIMX + NPAD];
   ```

### Corner-Turning



#### 定义


 Corner-Turning是一种在GPU计算中用于**优化内存访问模式**的技术。它的原理是将数据在**内存中重新排列**，原本分散在内存中的数据按照**每个线程块可以同时访问一段连续的内存**而整理到一起，使得每个线程块内部的访问就可以利用GPU的内存带宽，从而提高计算性能

#### 特点与性能分析
排列后的数据可以被看作是一个**多维数组**，由于其**存储方式的改变**，数据在访问时也需要进行一定的转换。

在某些情况下**可以显著提高性能**，但在其他情况下**可能导致性能下降**。需要仔细评估其适用性，并进行充分的测试和调优。


#### 例子

 * 待排序二维数组A，大小为**M x N**，其中M表示行数，N表示列数
 * 使用一个大小为**P x Q**的线程块来处理A
   * 为了最大程度地利用GPU的内存带宽，使用 Corner-Turning 将A重新排列为一个大小为**P x (M/P) x Q x (N/Q)**的四维数组B。其中，每个线程块可以处理B中的一个二维子数组。
     * 将A中的每个元素按照其所在行和列的顺序依次存储到一个一维数组C中。即，C[0]表示A[0][0]，C[1]表示A[0][1]，C[N]表示A[1][0]，以此类推。
     * 将C中的元素按照一定的方式重新排列，使得相邻的P个元素在行上连续。`C[0], C[N], C[2N], ..., C[(M/P-1)N] // 第一个线程块处理的数据`、`C[1], C[N+1], C[2N+1], ..., C[(M/P-1)N+1]`、`...`、`C[P-1], C[N+P-1], C[2N+P-1], ..., C[(M/P-1)N+P-1]`
     * 将排列后的元素存储到B中。具体而言，将C中的前**P*(M/P)**个元素依次存储到B的第一维中，每个线程块处理的数据存储在B的第二维和第四维中。
     * `B[i][j][k][l] = C[i*(M/P)N + jN + k*Q + l]`
     * 每个线程块可以同时访问B中的一个大小为(M/P) x (N/Q)的子数组，从而最大程度地利用GPU的内存带宽。




#### Example GEMM data access

  当使用tilnig+每个thread读取一个M N到shared memory的时候，读取M也是burst的。这是因为比起上面的simple code使用iteration读取，这里使用多个thread读取，一次burst的数据会被临近的thread使用(M00 M01分别被2个thread读取，每个thread只读取一个M elem)，而不是下一个iteration被清空。

  这里对于M没有使用显性的memory transpose，但是因为使用多个thread读取数据，依旧保证了burst，这与CPU代码需要使用transpose是不一样的。

  同时shared memory使用的是SRAM，不像DRAM有burst的问题，所以读取M的shared memory的时候尽管不是连续读取也没有问题。shared memories are implemented as intrinsically high-speed on-chip memory that does not require coalescing to achieve high data access rate.

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_33.png)


### Async global memory to shared memory

 **支持版本**
  CUDA 11.0



 **优点**

  * 将<font color = red>数据复制与计算重合</font>
  * 避免使用中间寄存器，<font color = red>减少寄存器压力</font>，减少指令流水压力，提高内核占用率
  * 相对于sync，async<font color = red>延迟更少</font>

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_34.png)




 **async与Cache关系**

  * <font color = red>可以选择是否使用L1 cache</font>. 
  
  * 因为shared memory是per SM的，所以不涉及使用L2 cache(across SM)



 **优化参考图**

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_35.png)

  
  * 对于<font color = red>sync拷贝，num data是multiply of 4 最快</font>

  * 对于<font color = red>async拷贝，data size是8/16是最快的</font>。



#### 例子

  ```cpp
  template <typename T>
  __global__ void pipeline_kernel_sync(T *global, uint64_t *clock, size_t copy_count) {
    extern __shared__ char s[];
    T *shared = reinterpret_cast<T *>(s);

    uint64_t clock_start = clock64();

    for (size_t i = 0; i < copy_count; ++i) {
      shared[blockDim.x * i + threadIdx.x] = global[blockDim.x * i + threadIdx.x];
    }

    uint64_t clock_end = clock64();

    atomicAdd(reinterpret_cast<unsigned long long *>(clock),
              clock_end - clock_start);
  }

  template <typename T>
  __global__ void pipeline_kernel_async(T *global, uint64_t *clock, size_t copy_count) {
    extern __shared__ char s[];
    T *shared = reinterpret_cast<T *>(s);

    uint64_t clock_start = clock64();

    //pipeline pipe;
    for (size_t i = 0; i < copy_count; ++i) {
      __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                              &global[blockDim.x * i + threadIdx.x], sizeof(T));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    
    uint64_t clock_end = clock64();

    atomicAdd(reinterpret_cast<unsigned long long *>(clock),
              clock_end - clock_start);
  }

  ```



#### API

  **__pipeline_memcpy_async()**

   * 代码将global memory数据存储到shared memory


  **__pipeline_wait_prior(0)**

   * 等待，直到剩余0条指令未执行



















