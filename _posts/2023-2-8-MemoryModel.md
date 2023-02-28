---
layout: article
title: GPU Memory Model
key: 100024
tags: GPU架构
category: blog
date: 2023-02-08 00:00:00 +08:00
mermaid: true
---



# Memory Model



## Memory Hierarchy

 ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/fd42780ecc814652ae3db95508b152b4~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


 <font color=red>指针</font>是被用来<font color=red>指向global memory</font> 的数据


 **latency** 

  | type                       | clock cycle                             |
  | -------------------------- | --------------------------------------- |
  | register                   | 1                                       |
  | shared memory              | 5                                       |
  | local memory               | 500 (without cache)                     |
  | global memory              | 500                                     |
  | constant memory with cache | 1(same as register)-5(same as L1 cache) |
  | L1 cache                   | 5                                       |


 **device memory 可以分为两类**

  * <font color=red>linear</font> memory
  * <font color=red>CUDA arrays</font> -> 常用于texture


 **各种内存的物理实现**
  * global memory -> DRAM
  * shared memory -> SRAM
  * const memory -> 专门存储器
  * texture memory -> DRAM，专门用于高速图像缓存，具备二维结构
  * L1/L2 cache -> 利用SRAM和逻辑电路实现

<!--more-->

 **为什么重视内存访问**

  * 必须清楚<font color=red>desirable</font> (getter)和<font color=red>undesirable</font> (scatter)的内存访问方式。
  -><font color=red>*gather*</font> 和<font color=red>*scatter*</font>
    * getter:memory到thread有多个输入，只有一个输出（图像模糊算法）
    * scatter:thread到memory有一个输入，多个输出（每个thread将结果scatter到各个memory，将附近值+1）


**内存模型在GPU编程中优化方向**

 * global memory
   * 带宽、内存合并和内存对齐
 * shared memory
   * bank conflict、数据分布、conner-turning、async
 * const cache
   * broadcast、serialization
 * texture cache
   * 内存对齐、纹理过滤、纹理大小、内存访问模式
 * L2 cache
   * persisting
 * local memory
   * 数据重用、数据对齐、同步原语
 * register
   * 数据重用、循环展开




## Global Memory


### Bandwidth

带宽有理论带宽和实际带宽，该部分是实际带宽，包括CPU方法和GPU方法

#### 带宽的计时器

实际带宽的计算主要通过定时器实现，包括CPU方法和GPU方法

##### CPU计时

相比起GPU 计时来说，<font color=red>比较粗糙</font>

   ```cpp
   // sync all kernel on device before timer
   cudaDeviceSynchronize();

   // start CPU timer

   // do some work

   // sync all kernel on device before timer
   cudaDeviceSynchronize();

   // end CPU timer

   // compute time
   ```



##### GPU 计时 


使用  <font color=red>GPU时间</font>，因此与操作系统无关

   ```cpp
   cudaEvent_t start, stop;
   float time;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   // start event
   cudaEventRecord( start, 0 );

   // do some work
   kernel<<<grid,threads>>> ( d_odata, d_idata, size_x, size_y, NUM_REPS);

   // end event
   cudaEventRecord( stop, 0 );

   // wait for end event
   cudaEventSynchronize( stop );

   // compute time
   float milliseconds;
   cudaEventElapsedTime( &milliseconds, start, stop );
   cudaEventDestroy( start );
   cudaEventDestroy( stop );

   // bandwidth = bytes data / 1e6 / millisecond
   // 			     = bytes data / 1e9 / second
   ```

主要通过GPU的event计时完成

#### 带宽计算

主要包括理论带宽计算、实际带宽、和性能分析工具
##### 理论带宽


   **HBM2 example**

   * V100使用HBM2作为global memory的带宽（时钟频率<font color= red>877MHz</font>、为<font color = red>double data rate RAM</font> 、内存带宽为<font color = red>4096 bit </font>。

   $
   0.877 * 10^9 * 4096 / 8 * 2 / 10^9 = 898GB/s
   $

   ```cpp
   cudaDeviceProp dev_prop;
   CUDA_CHECK( cudaGetDeviceProperties( &dev_prop, dev_id ) );
   printf("global memory bandwidth %f GB/s\n", 2.0 * dev_prop.memoryClockRate * ( dev_prop.memoryBusWidth / 8 ) / 1e6 );
   ```


   **GDDR中的ECC导致带宽下降**
   * 在GDDR中加入ECC(Error Correction Codes)会导致ECC overhead，会导致理论带宽下降，HBM2因为有专门给ECC的部分，所以<font color = red>理论带宽没有下降</font>






##### 有效带宽

   $
   ((B_r + B_w) / 10^9 ) / time
   $

   $10^9 = 1024 * 1024 * 1024$ 是将$bytes$转化为$GB/s$




#### Visual profiler 内核性能分析工具



 **<font color=red>requested throughput</font> 和 <font color = red>global throughput</font>** 

   * 系统的global throughput 相当于物理理论带宽（考虑到cache line 要一起传送的带宽）
   * 系统的requested memory 相当于实际带宽（并未考虑 cache line）
   * 应当让requested throughput尽可能接近global throughput。


  


### Device Memory DRAM

 本部分主要讲述global memory的硬件实现<font color = red>DRAM</font>


#### Bit Line & Select Line


![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b437a3d682914a929f5d1397eabb2081~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  **原理**
   * 一个<font color= red>电容</font>决定一个bit， 通过<font color= red>select线</font>选择读取哪个bit，一个<font color= red>bit line</font>操作只能读取一个bit的数据，即从多个select中选择一个
   * 需要不停<font color= red>检查</font>和<font color= red>电容充放电</font>，因此叫做DRAM



  **特点**

  * <font color= red>bit line的容量大</font>，导致速度慢

  * 每个<font color= red>bit电容</font>，需要信号放大器放大

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ce8a3806169e4a26a84da1a02770dd6b~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


#### DRAM 的数据传输(Core Array & Burst)
 ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4ce77fdf7d1c48bc8e6d571ba0e77004~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  **传输过程**
   * 数据传输分为两个部分。<font color = red>core array -> column latches / buffer </font>和<font color = red>column latches / buffer-> mux pin interface</font>
   
   * *core array*是由多多个bit line组成


  **传输耗时**
  * core array -> column latches / buffer 耗时<font color = red>久</font>

  * buffer -> mux pin interface 的耗时相<font color = red>对较小</font>
  

  **传输中的名词**
  

  **burst** 
  当访问一个内存位置的时候，<font color = red>多个bit line的数据都会从core array传输到column latches</font>，然后再使用mux来选择传送给bus哪些数据

  **burst size/ line size** 
   * <font color = red>读取一次memory address，会有多少个数据从core array被放到buffer中</font>

     * 常见1024 bits / 128 bytes (from Fermi)。

     * 当L1 cache disabled, burst size是32 bytes. 



  
  
  

#### Multiple Banks

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/19828fb956904e38aa6c75a78a1a12f4~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/26fe0b485c3b41a9b128a935c51081fa~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  **引入原因**
   * 从<font color = red>core array到buffer的时间长</font>，单个传输导致实际使用的<font color = red>bus interface数据bandwidth未被完全利用</font>（$T_c:T_b = 20:1$)。如果只使用一个bank导致带宽大部分时间为空闲。
   * 所以需要在一个bus 上使用多个bank，来充分利用bus bandwidth。如果<font color = red>使用多个bank，大家交替使用interface bus</font>，保证bus不会空闲，保证每个时间都有数据从bus传送过来


  **一个bus需要多少个bank？**

   * 如果访问core array与使用bus传输数据的时间比例是20:1，那么一个bus<font color = red>至少需要21个bank</font>才能充分使用bus bandwidth。
   * 一般bus有更多的bank，<font color = red>不仅仅是ratio+1</font>
     * <font color = red>为减少burst的可能性，将数据分布在不同的bank</font>，可以节约总时间
     * ***<font color = purple>每个bank可以存储的诗句有限，否则访问一个bank的latency会很大。</font>***

  
  
  

#### Multiple Channels

 ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/dcea6f6180db45128b25cc4cbc94eafc~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  * <font color = red>一般的GPU processor要求带宽达到128GB/s，HBM2要求带宽为898GB/s，在使用multiple bank后仍未满足要求</font>，因此使用multiple channel的方法。
  * 假设使用DDR DRAM clock为1GHz，每秒能传送8 bytes/ 1 words，得到的传输速率为16GB/s，因此需要多个channel
  


#### 数据分布方法Interleaved（交织） 

![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/95a1c7b5eb2a4a239aa533dd5557e5b0~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  **数据交织原因**
   * 充分<font color = red>利用多个channels的带宽，实现max bandwidth</font>。
   * 允许在<font color = red>core array -> buffer 的时间进行转化成多个channels获取数据</font>，减少总的时间。
  **如何实现数据交织**
  将数据平均分布<font color = red>在channels对应的banks</font>中
  

 
 
 

### global memory 内存合并和内存对齐

 
 

#### global memory 内存合并


  **global memory 内存合并原因**
   * 在GPU中对于<font color = red>内存数据的请求是以wrap 为单位</font>，而不是以thread 为单位。**<font color = purple >- - - - -></font>** *warp 内thread 请求的内存地址会合并为一个<font color = red>warp memory request</font>，然后这个request 由<font color = red>一个或多个memory transaction</font> 组成*。
   * 具体使用几个transaction 取决于request 的个数和transaction 的大小。

  **global memory 数据流向**
   * global memory request<font color = red>一定会经过L2</font>，是否经过<font color = red>L1 </font>取决于<font color = red>cc 和code</font>，是否经过read only texture cache取决于<font color = red>cc 和code </font>。

  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/58d61cd9fdca4555a1eab0b9664fbf4c~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

  **GPU 与CPU 对memory 的处理方式**

   * CPU
     * <font color = red>片上有很大的cache</font>，CPU thread访问连续的内存会被缓存。不同thread 由于不同core，数据<font color = red>相互不影响</font>。
     * 充分利用内存的方法：<font color = red>每个core负责一段连续的内存</font>。（e.g. thread 1 : array 0-99; thread 2 : array 100-199; thread 3 : array 200-299.）

   * GPU 
     * GPU 的<font color = red>线程以warp为单位</font>，一个SM上有多个warp运行，warp<font color = red>线程操作内存通过L1/shared memory</font>进行。
     * 使用warp操作内存，<font color = red>不同thread 对cache的结果会产生不同影响</font>。thread0读取数据产生的cache会对thread1读取数据产生的cache产生影响。
     * <font color = red>为充分发挥带宽，应当在warp的每个iteration中保证花费全部cache line</font>。因为有很多warp同时在sm上运行，等下一个iteration的时候 cache line/DRAM buffer已经被清空了。
  
  

  **GPU常用优化方法**

   * 使用内存对齐和内存合并来提高带宽利用率
   * ***<font color = purple> sufficent concurrent memory operation 从而确保可以hide latency</font>***
     * ***<font color = purple> loop unroll 从而增加independent memory access per warp, 减少hide latency所需要的active warp per sm</font>***
     * ***<font color = purple>modify execution configuration 从而确保每个SM都有足够的active warp。</font>***


 
 

#### 内存对齐

 L1 cache line = 128 bytes, L2 cache line 32 bytes，warp的<font color = red>内存请求起始位置位于cache line的偶数倍</font>，为了保证对global memory 的读写不会被拆为多个操作，应保证存储对齐。



  **image cache line**

   为了<font color = red>防止对不同row下的读写产生多个memory segment</font>，导致速度变慢，在<font color = red>每一行末尾加入padding</font>

   padded info叫做 `pitch` 

   ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8b663c3f737346bf99cba00a6a6961b5~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  **CUDA API**

   <font color = red>align to 256 bytes</font>

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

   * 结构体大小应保证为<font color = red> 1, 2, 4, 8, or 16 bytes</font>，如果不是这些大小，则会产生多个transaction。
  
   * 如果一个struct是7 bytes，那么padding成8 bytes会用coarlesed access。但是如果不paddig的话则会是多个transaction。

   * 下面的marcro可以align struct从而确保coarlesed access

   ```cpp
   struct __align__(16) {
   float x;
   float y;
   float z; 
   };
   ```



#### Global Memory 读

  注意： GPU L1 cache is designed for spatial but not temporal locality. Frequent access to a cached L1 memory location does not increase the probability that the data will stay in cache. L1 cache是用于spatial（连续读取array）而不是temporal（读取同一个位置的），因为cache line很容易被其余的thread evict。

  

  **内存读性能：global memory load efficency**

  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/94e7c7c2fe3d4b48a942f1524465523c~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


   nvprof.gld_efficency metrics衡量了此指标



  **内存读模型：Simple model**

   *  在128 bytes/32 bytes的模式下，会产生128 bytes/ 32 bytes / 64 bytes的memory transaction （32 bytes当four segment的时候也会是128 bytes）。在L1中 memory被合并分块为 <font color = red>32-, 64-, or 128- byte</font> memory transactions，在L2中被分块为<font color = red>32 bytes</font>。



##### Read-only texture cache

   * 使用条件：<font color = red>CC 3.5+ </font>可以使用read only texture cache

   * cache line大小：The granularity of loads through the read-only cache is<font color = red> 32 bytes</font>. 



##### CC 2.x Fermi 

   * 2.x default 使用 L1 + L2 cache

   * 2.x 可以通过config disable L1 cache

   ```shell
   // disable L1 cache
   -Xptxas -dlcm=cg

   // enable L1 cache
   -Xptxas -dlcm=ca
   ```



   **128 bytes transaction**

   * 每个<font color = red>thread的大小对request的影响</font>
     * 如果每个thread请求的数据大于4 bytes（32 * 4 = 128)，则会被<font color = red>切分为多个128 bytes memory request</font>来进行。
     * 如果每个thread请求8 bytes，这样保证了每个传送的128 bytes数据都被充分利用(16 threads * 8 bytes each)
     * 如果每个thread请求16 bytes，four 128-bytes memory request,这样保证了传送的128 bytes数据被充分利用
   * <font color = red>request拆分到cache line层面</font>上，解决indenpent 问题
     
     * 当warp的内存请求位于同一个连续对齐的cache line内。 
     ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/371d913f971f445cbac083a44e78767e~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




     * 数据都位于cache line内，但之间的关系无序。只要warp memory request是在128 bytes transaction内，只会进行一个memory transaction。
      ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b6f5043ba4894392a675e3b512be5643~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


     * warp连续但是并未对齐，导致产生两个128 bytes transaction
      ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1670dec5bc1a4f78941ed329c8b3cf54~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)





     * 所有线程请求同一地址


      ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e2bba1c5321f4ea0b10585f69a47756b~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

     * warp的线程以32 addresses为单位scatter到global memory的情况。

      ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4e150726ea384b37857330a96f1efd3b~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


   * 32 bytes transaction

     * transaction只由L2 cache获取global memory数据

   



     * 内存请求连续对齐的128 bytes，进行需要进行四次transaction。bus利用率100%
      ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b1abc9989a424743b9edb85502ff053c~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



     * 内存请求非连续，但都在128bytes内，请求四次transaction。
      ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/95b2fd2869cb4b728dcdebebf7255fc1~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




     * 请求相同地址，只需要一次transaction
      ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/192d74b62c2e4e728285d371ee23e69d~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




     * 请求四个32 bytes的scatter分布，相对<font color= red>128 bytes的方式，缓存利用率高</font>。
      ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3c3971b84b3845abbec0da7c76b60502~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




##### CC 3.x Kepler cache line大小

   * 3.x default 使用 L2 cache，不使用L1 cache

   * 3.5 / 3.7 可以使用read only texture cache

   * 3.5 / 3.7 可以config使用L1 cache

   * L1 cache line size 128 bytes

   * L2 cache line size 32 bytes

   为什么L2 cache 需要以1、2、4倍数传输：为了避免DMA fetch ***<font color= purple>DMA FETCH: The DMA supports an AXI bus width of 128/64 bits. In the case where the source descriptor payload ends at a non-128/64 bit aligned boundary, the DMA channel fetches the last beat as the full-128/64 bit wide bus. This is considered an over fetch.</font>***
   当使用L2 cache only的时候，memory transaction是32 bytes. Each memory transaction may be conducted by one, two, or four 32 bytes segments。可以减少over-fetch

  **L1/L2读取顺序**
   * 当使用L1 + L2时候，memory transaction是128 bytes。
   Memory request 首先会去L1，如果L1 miss会去L2，如果L2 miss会去DRAM。




##### CC 5.x Maxwell

   * 5.x default使用L2 cache，32 bytes transaction

   * 5.x 可以使用read only texture cache，32 bytes transaction

   * 5.x 可以config使用L1 cache（default不使用）


##### CC 6.x Pascal



#### Global Memory 写

  **与读的区别**
   * <font color = red>读可以用L1，写只能用L2，写只能用32 bytes</font>
   * 多个non-atomic thread 写入同一个global memory address，<font color = red>只有一个thread写入会被进行，但是具体是哪个thread是不确定的</font>


  **efficency** 

   * memory store efficency 与 memory load efficency的定义相似

   * nvprof.gst_efficency metrics 可衡量


  **transaction** 

   * 一个128 bytes的requested（4 个transactions）相对两个64 bytes(2*2 transactions)的requested来说，<font color = red>速度更快</font>

   * 读transaction的例子

     * 128 bytes 的连续对齐内存request，需要4个transactions
      ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e16280d148b642109f546638af9ef7d3~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




     * 128 bytes 的request，请求内存大小为64 bytes的连续内存空间，则需要2个transactions

      ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3a96d324395e4cafae139b8e42a8aaaa~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



#### From hardware

  第一次访问，<font color = red>全部4个数据都放到buffer</font>里

  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d699c3abcaca4813bfd57a5902fc540d~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


  第二次访问使用后面两个数据（连续内存访问），<font color = red>直接从buffer里读取数据，不用再去core array</font>
  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/16e8a1d94ced46f4869874e3ae7bcdc5~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



  **burst 原因** 
   * 在从core array -> buffer 的过程需要的时间长，在每一次从core array 到buffer 的过程中，传输burst数据，在每一次读取中，应让数据充分使用。因为<font color = red>两次core array -> buffer 时间远远大于两次 buffer -> bus的时间</font>。

  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/72a748346cc7408a8c4d7f12155df1d3~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)
  


   * 蓝色的部分是core array到buffer的时间。红色的部分是buffer到pin的时间


 
 
 

### Minimize host to device transfer



#### Batch Small Transfer

  * 为避免多个small memory copy，使用one large memory copy，将<font color = red>多个small memory打包成为一个large memory，然后进行copy</font>

#### Fully utilize transfer

  * 尽量将<font color = red>传输次数减少</font>，如果GPU计算并不方便，也使用，减少数据传输次数。





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

   ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/486202600e694ca6a85219aac41c7d2d~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)
   

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

  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/01c76f7625444f9c9a1e0de137a817cf~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)
  

  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6a65d5ed7ea04da08fdca199e423c622~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

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
   ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a7f94c5958194dcb8a04d3c0aa586cf6~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)
   


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

     ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e6691a2448d3426294bc38bab12dc065~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

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
  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a1286c80c94042c4b15d0f11668061ca~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)
  



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

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/20c6c29ec99549a9ab915e24a9cb7838~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

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

![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/06cb7b840d1248fa841390bccbcc68a5~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


### Async global memory to shared memory

 **支持版本**
  CUDA 11.0



 **优点**

  * 将<font color = red>数据复制与计算重合</font>
  * 避免使用中间寄存器，<font color = red>减少寄存器压力</font>，减少指令流水压力，提高内核占用率
  * 相对于sync，async<font color = red>延迟更少</font>

  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/794e0e9a908e4601bd85dfd247927e0e~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




 **async与Cache关系**

  * <font color = red>可以选择是否使用L1 cache</font>. 
  
  * 因为shared memory是per SM的，所以不涉及使用L2 cache(across SM)



 **优化参考图**

  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1c86fb3e886a4c7f83f8c56defd256c1~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

  
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




















## Constant cache & Read-Only Cache

### 不同点


**不同cc的const memory, texture memory & L1 cache**
  * cc 2.x -> 只有L1
  * cc 3.x -> L1/shared

 对于不同cc ，constant cache，read-only texture cache, L1 cache的关系是不太一样的。


 **GPU cache类型**

  * <font color = red>L1 Cache</font>
  * <font color = red>L2 Cache</font>
  * <font color = red>read-only constant cache</font> (through constant memory)
  * <font color = red>read-only texture cache</font> (thorugh texture memory / ldg load global memory)


 **constant cache 和 read-only cache**
  * <font color = red>constant cache</font>相对小，且存取格式统一，更适合warp内的所有thread访问同一地址。
  * <font color = red>read-only cache</font>适用于更大的场景，数据传输没有必要统一。
  * 使用constant cache对于warp内的所有thread访问同一地址性能更好，因为constant对broadcast优化更好。

### Constant Memory & Constant Cache


 **Constant Memory物理存储特性**
  * Constant Memory用于在device上的uniform read。
  * 位置： 物理上与global memory都在<font color = red>off-chip device memory</font>上
  * 大小： <font color = red>64 kb constant memory for user</font>, 64 kb for compiler。内核参数通过constant memory传输
  * <font color = red>速度</font>： 与register相同
  * <font color = red>带宽与延迟</font>： 带宽大于 L1，延迟与 L1 一致(5 cycle)



 **Constant Memory 使用**
  * constant memory不仅仅可以被相同file的全部grid可见，还是<font color = red>visibale across soruce file</font>的
  * 常用于储存<font color = red>formula的coefficent</font>。warp threads会一起访问某一个coefficent，这样是最适合constant memory的。
    * 之所以不用register储存coefficent是因为有太大的register pressure，<font color = red>导致num block/SM下降</font>



#### Broadcast

  * 在constant内部有<font color = red>专门用于broadcast的接口</font> 

  * 当warp thread访问相同的constant memory location的时候，会进行broadcast



#### Serialization

  * 出现在warp内的threads访问不同constant memory location的时候。
  * warp内对于constant cache<font color = red>不同地址的访问是serialized</font>的。且<font color = red>消耗为线性的，与需要的不同地址数成正比</font>。
  * 如果t0访问constant cache addr 0， t1访问constant cache addr 1，这两个对constant cache的访问会serialized。
  * 对于使用constant cache，最好的访问方法是all threads within warp only access a few (serialization not too much)  / same memory address (use broadcast) of constant cache. 




#### API

  ```cpp
  // copy host to constant memory on host
  cudaError_t cudaMemcpyToSymbol(const void *symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind)
  ```



  **API with example**

   ```cpp
   __constant__ float coef[RADIUS + 1];


   __global__ void stencil_1d(float *in, float *out) { 
     // shared memory
     __shared__ float smem[BDIM + 2*RADIUS];
     // index to global memory
     int idx = threadIdx.x + blockIdx.x * blockDim.x;
     // index to shared memory for stencil calculatioin 
     int sidx = threadIdx.x + RADIUS;
     // Read data from global memory into shared memory 
     smem[sidx] = in[idx];
     // read halo part to shared memory 
     if (threadIdx.x < RADIUS) {
       smem[sidx - RADIUS] = in[idx - RADIUS];
       smem[sidx + BDIM] = in[idx + BDIM]; 
     }
     // Synchronize (ensure all the data is available) 
     __syncthreads();
     // Apply the stencil
     float tmp = 0.0f;
    
     #pragma unroll
     for (int i = 1; i <= RADIUS; i++) {
       tmp += coef[i] * (smem[sidx+i] - smem[sidx-i]); 
     }
    // Store the result
     out[idx] = tmp; 
   }
   ```

### Read-Only Texture Cache




 **read-only texture cache物理存储特性**
  * Kepler开始，GPU支持对global memory使用<font color = red>per SM read-only cache</font>。
  * 底层使用GPU texture pipeline as read-only cache for data stored in global memory

 **与global memory传输数据区别**
  * Global memory 通<font color = red>L1 + L2/ L2</font>存取数据，是否采用L1取决于cc和config。
  * Global read-only memory 通过<font color = red>texture和L2 cache 得到数据</font>
  * 通过read-only texture cache (也会通过L2 Cache) 读取global memory比起normal global memory read (会通过L1+L2 cache)有<font color = red>更大的bandwidth</font>
  * read-only cache 的cache line是<font color = red>32 bytes</font>。
  * 相比起L1，对于<font color = red>scatter read使用read-only cache更有效</font>。



#### API




  **intrinsic**

   对于<font color = red>cc > 3.5 </font>的设备，可以使用intrinsic来强制得到对应data type T的数据。

   ```cpp
   // input is read-only texture memory
   __global__ void kernel(float* output, float* input) 
   { 
     ...
     output[idx] += __ldg(&input[idx]);
     ... 
   }
   ```



  **compiler hint**

   对于<font color = red>compiler的hint</font>，让compiler生成read-only cache读取

   对于复杂的kernel，有些时候compiler hint可能不管用，还是推荐ldg读取

   ```cpp
   void kernel(float* output, const float* __restrict__ input) 
   { 
     ...
     output[idx] += input[idx]; 
   }
   ```














## L1 & L2 Cache





### Cache VS Shared Memory
  

 **相同点**

  * 都是片上内存
  * 如果是volta架构，则都为SRAM

 **不同点**

  * shared memory 可控 
  * cache 是由CUDA控制

   

### L2 cache persisting
 

 **L2 cache persisting使用条件与场合**

  <font color=red>CUDA 11.0 + cc 8.0</font> 可以配置 L2 persistence data

  定义: <font color=red>经常被global memory访问</font>的数据需要被设为persisting

  

 **设定L2 persistance**


  ```cpp
  cudaGetDeviceProperties(&prop, device_id);
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize); 
  /* Set aside max possible size of L2 cache for persisting accesses */
  ```
  persisting L2是<font color=red>优先使用</font>的部分，当<font color=red>*persist数据或streaming*</font>首先处理的时候，会先看persisting L2是否有数据，如果数据可用，则将优先使用persisting L2，且<font color=red>*当为persisting*</font>时候则只能通过此传输



   

 **设定num_bytes和hitRatio**

  ```cpp
  cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
  stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
  stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persisting accesses.
                                                                                // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
  stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                          // Hint for L2 cache hit ratio for persisting accesses in the num_bytes region
  stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
  stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

  //Set the attributes to a CUDA stream of type cudaStream_t
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   
  ```

  `hitRatio` 设置global memory有多少<font color=red>比例的数据需要设为persisting</font>。如果为0.6，表示有60%的数据需要persisting，剩余40%为streaming。


  需要确保<font color=red>num_bytes * hitRatio < L2 persistance</font>(当num_bytes * hitRatio <font color=red>*<*</font> L2 persistance，剩余部分会使用streaming访问，如果<font color= red>超过</font>了L2 persistance的大小，CUDA依旧会尝试把数据放到L2 persistance的部分，导致thrashing）

  

 **例子**
  A100有40M L2 memory，`cudaStreamSetAttribute()`设置用于data <font color=red>persistance的数据大小为30M</font>
  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d94878c071dc4634b2a98fbf38fb6556~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

  
  使用<font color=red>10-60M</font>需要persistance的数据进行实验，<font color=red>hitRatio大小设为1</font>

  ```cpp
  __global__ void kernel(int *data_persistent, int *data_streaming, int dataSize, int freqSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
     /*Each CUDA thread accesses one element in the persistent data section
      and one element in the streaming data section.
      Because the size of the persistent memory region (freqSize * sizeof(int) bytes) is much
      smaller than the size of the streaming memory region (dataSize * sizeof(int) bytes), data
      in the persistent region is accessed more frequently*/

    data_persistent[tid % freqSize] = 2 * data_persistent[tid % freqSize];
    data_streaming[tid % dataSize] = 2 * data_streaming[tid % dataSize];
  }

  stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data_persistent);
  stream_attribute.accessPolicyWindow.num_bytes = freqSize * sizeof(int);   //Number of bytes for persisting accesses in range 10-60 MB
  stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                      //Hint for cache hit ratio. Fixed value 1.0
  ```

  ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/263598932faa4060931bca09b77d5686~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

  
  得到结果，在没有超过L2 persisting size时，性能提高（在nums_types = 20时，性能提高<font color=red>50%</font>），超过L2 persisting size，则会对性能下降约<font color=red>10%</font>
  </font>。
  
  改变hitRatio与type_nums时，有：

  ```c++
  stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data_persistent);
  stream_attribute.accessPolicyWindow.num_bytes = 20*1024*1024;                                  //20 MB
  stream_attribute.accessPolicyWindow.hitRatio  = (20*1024*1024)/((float)freqSize*sizeof(int));  //Such that up to 20MB of data is resident.
  ```
  其中配置num_bytes = 20M, hitRatio大小设置根据输入数据的规模判断，<font color=red>当规模大，设置ratio小，当规模小，设置ratio大</font>。
  此时hitRatio*num_bytes < L2 persisting cache 对应性能如下：
  ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6de68b3e0ec44185936ed21949d6cf93~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


 **hitProp配置**

  * <font color = red>cudaAccessPropertyStreaming</font>: 当cache hit之后，不将其 persist
  * <font color = red>cudaAccessPropertyPersisting</font>: 当cache hit之后，会进行 persist
  * <font color = red>cudaAccessPropertyNormal</font>: 使用这个可以清空前一个kernel使用的L2 cache


  

 **重置L2 persisting**

  * <font color = red>cudaAccessPropertyNormal</font>

  * <font color = red>cudaCtxResetPersistingL2Cache</font>: 会reset所有的persisting L2 cache

  * automatic reset: 不推荐


  

 **对并发线程同时使用L2 persisting**

  如果多个kernel同时运行的话，需要让<font color = red>所有用到persisting的数据求sum让其小于L2 set aside</font>

  








## Local memory



### Basic
 

 **local memory物理特性**

  <font color = red>off-chip</font>

 **什么样的变量会放在local memory上**

  * array
    * <font color = red>array默认</font>都是放在thread private的local memory上

    * array也有可能会被<font color = red>编译器优化</font>后，放到register上。如果满足array大小给定，且array长度小，则可以优化否则会被放在local memory上，因为compiler不知道这个array会有多长，无法把array拆分后放到regsiter中。

  * struct
    * 如果struct <font color = red>占用空间很大的话，也有可能被放在local memory上</font>

 **如何确定变量是在local memory上**

  * 通过<font color = red>PTX可以确定第一轮编译</font>以后是否在local memory上
  * 但是第一轮不在local memory上不代表后面不会放到local memory上
  * 通过<font color = red> `--ptxas-options=-v` 查看总local memory使用用量</font>。

### Coarlesed




#### local memory 特点

  <font color = red>高延迟</font>
  <font color = red>低带宽</font>

  需要global memory一样的<font color = red>内存coarlesed</font>



#### automatic coarlesed layout

  * local memory通过连续的32位组织起来，通过threadid获取。
  * 当获取<font color = red>地址相同时，warp thread获取可以进行合并</font>。 
  * local memory在device上的layout：t0 idx0, t1 idx0, t2 idx0, ... t31 idx0, t0 idx1, t1 idx 1



#### cache behaviour 

  * cc 3.x, local memory accesses are always cached in <font color = red>L1 and L2</font>

  * cc 5.x and 6.x, local memory accesses are always cached in <font color = red>L2</font>











## Register


 **特点**

  * Register带宽大，在 <font color = red>一个时钟周期中，thread可以访问多个register</font>
  *  <font color = red>延迟低 </font>

 **register数据传输**
  * 在每个thread读的过程中，对register 读只能 <font color = red>按顺序</font>

  * ***<font color = purple>register tiling requires thread coarsening</font>***

  * Registers 是  <font color = red>32 bit大小的</font> (same size as int / single precision float)。如果数据类型是double的话，则使用2个register。

  * 可以通过 <font color = red>pack small data into a register</font> (e.g. 2 short) and use bitmask + shift 来读取。从而减少register usage per thread

 **Bank conflict**

  * Register 也会有 <font color = red>bank conflict，只不过这是完全由compiler处理的</font>，programmer对于解决register bank conflict没有任何控制。

  * 并不需要特意把数据pack成vector type从而来避免bank conflict

 **控制per thread max register**

  * 可以通过<font color = red>compiler option来控制max register pre thread</font>

  ```shell
  -maxrregcount=N
  ```


### Bank Conflict


### Register Reuse












## Atomic

 CUDA 提供<font color = red>原子操作函数</font>，用于32/64位读、修改、写。

 **三种主要atomic类型**
  * arithmetic functions
  * bitwise functions
  * swap functions
  
 **常见的function**
  * atomicAdd   加
  * atomicSub   减
  * atomicExch  交换
  * atomicMin   最小
  * atomicMax   最大
  * atomicInc   自增
  * atomicDec   自减
  * atomicCAS   比较并交换
  * atomicAnd   与
  * atomicOr    或
  * atomicXor   位或

  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6fc29b6198c2401ca1ee5f52e0da0844~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


### CAS(compare and swap)



 * CAS(compare and swap) 是一切atomic operation 的基础，全部的<font color = red>atomic 操作都可以使用CAS 实现</font>。

 * CUDA的`atomicCAS()`支持输入参数为`unsigned long long`或`unsigned int`或`int`或`unsigned short int`其他的参数需要进行转换才可使用（详见**PyTorch中的CAS相关实现**）

 **CAS 输入**
  * 某<font color = red>内存地址</font>
  * 内存地址的<font color = red>期望值</font>
  * <font color = red>修改</font>的期望值
  
 **CAS 步骤**
  * 读目标地址并将其与内存期望值<font color = red>比较</font>
    * 如果目标值与期望值<font color = red>相同</font>，则将其改为期望的值
    * 如果目标值与期望值<font color = red>不同</font>，不改变值
  * 在上述情况中，CAS操作总是返回在目标地址找到的值。通过返回目标地址的值，可以判断CAS是否成功，如果改为修改后的期望值，则CAS操作必须成功。

### 实现自己的CAS

 **build float atomic add with float CAS**

  ```cpp
  __device__ int myAtomicAdd(int *address, int incr) 
  {
    // Create an initial guess for the value stored at *address. 
    int expected = *address;
    int oldValue = atomicCAS(address, expected, expected + incr);
    // Loop while expected is incorrect. 
    while (oldValue != expected) {
      expected = oldValue;
      oldValue = atomicCAS(address, expected, expected + incr); 
    }
    return oldValue; 
  }
  ```
  * 使用<font color = red>while 原因</font>：
    * 在做add 的时候，数据可能<font color = red>已经发生改变</font>，使用while 更新未被add的数据之后再做add。

  * <font color = red>读其他线程修改的参数是安全的</font>。
  * 这里的while loop和thread写入产生conflict 的replay很像（see below)


 **PyTorch中的CAS相关实现**


 无**特殊类型**输入情况
  ```c++
  template <typename T>
  __device__ void atomic_add(T* addr, T value){
    atomicAdd(addr, value);
  }
  ```

 重写**其他数据类型**
  ```c++
  __device__ __inline__ void atomic_add(int64_t *addr, int64_t val)
  {
      if (*addr >= val)
          return;
      // int8_t      : typedef signed char;
      // uint8_t    : typedef unsigned char;
      // int16_t    : typedef signed short ;
      // uint16_t  : typedef unsigned short ;
      // int32_t    : typedef signed int;
      // uint32_t  : typedef unsigned int;
      // int64_t    : typedef signed  long long;
      // uint64_t  : typedef unsigned long long

      unsigned long long *const addr_as_ull = (unsigned long long *)addr;
      unsigned long long old                = *addr_as_ull, assumed;
      do {
          assumed = old;
          old     = atomicCAS(addr_as_ull, assumed, reinterpret_cast<int64_t &>(old) + val);
      } while (assumed != old);
  }
  ```


 **使用已有原子类生成不存在的原子函数**

  除了实现已知类型的原子类，还可以实现一些通过<font color = red>隐含类型转换产生的原子函数</font>
  * 以下例子介绍了将address的值转成float，之后increment后，转成unsigned int返回。

  ```cpp
  __device__ float myAtomicAdd(float *address, float incr) 
  {
    // Convert address to point to a supported type of the same size 
    unsigned int *typedAddress = (unsigned int *)address;
    // Stored the expected and desired float values as an unsigned int 
    float currentVal = *address;
    unsigned int expected = __float2uint_rn(currentVal);
    unsigned int desired = __float2uint_rn(currentVal + incr);
    int oldIntValue = atomicCAS(typedAddress, expected, desired); 
    
    while (oldIntValue != expected) 
    {
      expected = oldIntValue;
      /*
      * Convert the value read from typedAddress to a float, increment, 
      * and then convert back 	to an unsigned int
      */
      desired = __float2uint_rn(__uint2float_rn(oldIntValue) + incr);
      oldIntValue = atomicCAS(typedAddress, expected, desired); 
    }
    
    return __uint2float_rn(oldIntValue); 
  }
  ```

### 非原子类处理效果
 * 如果warp内的多个thread non-atomic写入相同地址, 则只有一个thread会进行write，但是是<font color = red>哪个thread是undefined</font>的

### 原子类函数的延迟和带宽




#### 延迟

  **延迟影响**
   * 在原子类操作中，需要<font color = red>先读取**gloabl memory**</font>，把数据<font color = red>传送给SM</font>（这个时候其余的线程和SM不能读写这块内存)，最后把数据<font color = red>传送给global memory</font>
  
  **延迟来源**
   * 原子类操作的延迟 = DRAM 取数据延迟+ internal routing + DRAM 存数据延迟
  
  **各延迟的时间比对**
   * <font color = red>global memory </font>-> few hunderdes cycle

   * last level cache -> few tens cycle

   * shared memocy -> few cycle
   ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1787c40233c04f18afff3088c9fcce33~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



  **延迟改进**
  * 硬件改进
    * 针对global memory读取时间长的问题，现代GPU将支持的原子操作改为L2 cache(cc 3.x)、shared memory(cc 4.x)、warp/block(cc 5.x and later)
    * <font color = red>last level cache上进行</font>，把atomic的latency从few hunderdes cycle变为了few tens cycle. 这个优化不需要任何programmer的更改，是通过使用<font color = red>更先进的hardware来实现的</font>。
  * 软件改进
    * 将global memory保存在**shared memory**中，但可能会带来SMs和并行度的下降(kepler, maxwell)
    * 将global memory保存在**texture memory**中，但对于texture的访问可能会有问题
    * 合并原子操作，可以将多个原子操作合并为一个原子操作，这样可以减少原子操作的数量，从而降低延迟，但需要加锁
    * 将global memory保存在**warp/block**中，存在`Privatization`私有副本合并开销问题与序列问题
      * 开销问题：将数据拷贝到shared memory


#### 带宽

  * GPU的低延迟通过线程并行进行 -> 需要DRAM数据访问并发 -> 当使用原子类，访问被序列化，导致带宽降低

  * <font color = red>带宽和延迟是成反比的</font>



#### 例子

  **配置**

   如果DRAM 配置为8 channels、1GHz、double data rate、word大小为8 bytes。DRAM的延迟为200 cycles。


  **峰值带宽**
  
   $
   8 * 2 * 8 * 1G = 128GB/s
   $
  


  **带atomic的延迟的带宽**

   $
   1/400 * 1G= 2.5M \ atomic/s
   $


  **带uniform的atomic的延迟的带宽**

   如果 uniform为26得到的是
   $
   2.5M *26 \ atomic/s
   $


### 原子类操作的更新



 随着GPU架构的升级，GPU atomic也在更新。


  **GT200**

   global memory



  **Fermi to Kelpler**

   both atomic on L2 cache

   Improve atomic by add more l2 cache buffer 



  **kepler to maxwell**

   improve shared memory atomic through using hardware. 

   Kepler use software for shared memory atomic



  **after maxwell**

   atomic is rouphly the same

   the flexibility of atomic is changed. now have atomic within warp / block.



   **computation capacity 1.1**

   32-bit atomic in global memory



   **computation capacity 1.2**

   32-bit atomic in shared memory

   64 bit atomic in global memory 



  **computation capacity 2.0**

   64 bit atomic in shared memory 

### Replay

 **出现条件**
  * 如果<font color = red>多个thread 对于同一个memory location进行atomic操作</font>，在<font color = red>同一时间只会有一个thread成功，其余的thread会被replay</font>

 **后果**
  * 在warp内如果threads atomic concurrent的写入同一个memory location，则会产生retry。当某一个thread retry的时候，<font color = red>其余的thread会像是branch divergence一样stall</font>

### Warp-aggregated


 
 NVCC在编译的时候，使用warp aggregation，速度比手写的warp aggregation要快。


 **warp aggregation定义**

  * thread的atomic是<font color = red>以warp为单位进行atomic</font>。
  * 首先在<font color = red>warp内部计算出来要atomic的值</font>
  * 选出<font color = red>一个thread执行atomic</font>。
  * 由于atomic执行会产生serial execution，将带宽降低，因此选择<font color = red>只用一个thread执行atomic</font>，减少了atomic操作的执行次数。

 **atomic大小参考**
 
  atomic次数与bandwidth是log的反向相关。
  ![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5fda906fba8b4669a8b8157e12b7f7a3~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




## zero-copy memory

 **零拷贝内存定义与使用条件**
  * 本质上是将主机内存映射到device memory
  * GPU threads 直接获得host的零拷贝内存
  * 如果在host/device修改了内存，需要synchronize来保证一致性。

 **优点**

  * 当device memory不够用，可以使用零拷贝内存
  * 可以避免从device 到host 之间的数据传输

 **缺点**

  * 对于GPU设备连接到PCIs总线，只针对特殊场景效果较好。零拷贝速度比global memory/device memory要慢。
    * 在集成架构中，当CPU和GPU在一个芯片上，且共享内存，零拷贝内存不用走PCIs总线，对性能和可编程性都有帮助。
  
  **应用**
   * 很多时候用zero copy memory只是因为
     * device memory不够用
     * 简化程序

### API

  ```cpp
  cudaError_t cudaHostAlloc(void **pHost, size_t count, unsigned int flags);
  ```

  cudaHostAllocDefault 默认方式makes the behavior of cudaHostAlloc identical to cudaMallocHost

  cudaHostAllocPortable启用后，此块内存对应锁页内存。该内存可以被系统中的所有设备使用（一个系统中有多个CUDA设备时） 

  cudaHostAllocWriteCombined 在device中写不使用L1和L2，其使用PCI总线完成数据传输。读的时候因为没有L1和L2，速度非常慢。一般用于CPU写入和device读。

  cudaHostAllocMapped, 这样，这块存储会有两个地址：一个是从cudaHostAlloc() 或malloc() 返回的在主机内存地址空间上；另一个在设备存储器上，可以通过cudaHostGetDevicePointer() 取得。内核函数可以使用这个指针访问这块存储。













## Unified Virtual Address

 **支持版本**
  cc 2.0以上版本支持Unified Virtual Address。其host memory和device memory共享一块虚拟内存空间。
  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7ce06e8718c746a996f50f9ed913b346~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

 **使用**
  当有UVA时，就不用获得device的指针或管理两个指向相同地址的指针。


  ```cpp
  // allocate zero-copy memory at the host side 
  cudaHostAlloc((void **)&h_A, nBytes, cudaHostAllocMapped); 
  cudaHostAlloc((void **)&h_B, nBytes, cudaHostAllocMapped);
  // initialize data at the host side 
  initialData(h_A, nElem); 
  initialData(h_B, nElem);
  // invoke the kernel with zero-copy memory 
  sumArraysZeroCopy<<<grid, block>>>(h_A, h_B, d_C, nElem);
  ```




## Unified Memory




 * Unified Memory维护了一个内存块。可以同时在device和host中使用，底层对memory的host和device数据进行传输维护。
 * 与UVA的区别：UVA属于零拷贝内存，通过CPU端分配内存，将cuda地址映射上去，通过PCI-E进行每个操作，同时不会进行内存迁移。Unified Memory是对用户来说只有一个memory，由unified memory来负责底层数据的拷贝。

### API

  * static

   ```cpp
   __device__ __managed__ int y;
   ```



  * dynamic

   ```cpp
   cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags=0);
   ```








## Weakly-Ordered Memory Model

### 原因
  * 当同一个地址的值被多个thread修改就导致了inter-thread conflict，所以我们需要同步操作。常用的同步方法包括barriers 和memory fences。
    * barriers：block内所有thread会等待其他thread到达barrier point
    * Memory fence，所有thread会阻塞到所有修改Memory的操作对其他thread可见。
  * 同步的原因
    * 现代内存架构有非常宽松的内存模式，也就是意味着，Memory的读不必按照程序中的顺序来执行。CUDA采用了一种叫做weakly-ordered Memory model来获取更激进的编译器优化。

    * GPU thread写数据到不同的Memory的顺序（比如shared Memory，global Memory，page-locked host memory或者另一个device上的Memory）没必要跟程序里面顺序呢相同。如果两个thread一个read，一个write，没有sync的话，则行为是undefined的

    * 为了显式的强制程序以一个确切的顺序运行，就需要用到fence和barrier。他们也是唯一能保证kernel对Memory有正确的行为的操作。

### Explicit Barriers

  * 任何到达barrier的线程都会等待block内所有线程都到达此处。
  * 使用时候最终要的理解是那些可以到达__syncthreads()的线程需要其他可以到达该点的线程，而不是等待块内所有其他线程。

  ```cpp
  __share__ val[];
  ....
   if(index < n)
   {      
       if(tid condition)      
       {          
           do something  with val;          
           __syncthreads();       
       }      
       do something with val;
   }
  ```



  **错误**

   使用中的注意：当某个thread的 syncthreads 运行以后，需要all threads in threads block都运行这个特定的synchthread方程才可以继续

   ```cpp
   if (threadID % 2 == 0) {
     __syncthreads();
   } else { 
     __syncthreads();
   }
   ```

### Memory Fence

  **意义**
   fence之前写完了，fence之后其它thread就都知道这块Memory写后的值了

  **`void __threadfence_block();`**

   保证同一个block中thread在fence之前写完的值对block中其它的thread可见，不同于barrier，该function不需要所有的thread都执行。

  **`void __threadfence();`**

   应用范围在grid

  **`void __threadfence_system();`**
   其范围针对整个系统，包括device和host

  **例子 1**

   下面这个例子中，不可能得到A=1,B=20。因为X=10一定发生在Y=20之前，如果observe了Y=20的话，则X=10一定运行完了

   ```cpp
   __device__ int X = 1, Y = 2;

   __device__ void writeXY()
   {
      X = 10;
      __threadfence();
      Y = 20;
   }

   __device__ void readXY()
   {
      int B = Y;
      __threadfence();
      int A = X;
   }
   ```

  **例子 2**

   * 一个block写入global memory数据以及用atomic写入flag，另一个block通过flag判断是否可以读取global memory的数据。

   * 如果没有memory fence的话，可能flag会首先被atomic设置了，然后才设置global memory的数据。这样另一个block在读取到flag以后就开始读取global memmory的值可能就是不对的。

   * 通过使用memory fence，确保在fence后面读取memory的数据确实是fence之前写入的数据

### Volatile



  **是什么**

   * 不使用volatile，compiler可以对global memory/shared memory的read write进行优化，例如cache在L1 cache或者register上，只要符合memory fence的要求就可以进行优化。

   * 当声明volatile以后，compiler假设某个thread对内存的操作会any time被其余的thread使用，所以不适用cache进行优化，全部的写入会写入到gloabl memory/shared memory上。这样另一个thread可以读取对应的内存并且得到正确的数值。













## Hardware Implementation

### PCIe

  **用处**
   GPU与CPU通过PCIe链接

  **结构**
   * PCIe由多个link组成
   * 每个link包含多个lanes
   * 每个lane为1-bit width（由4 wires组成，构成16GB/s）

  ![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4cc5cdbb33ae47df88286ca5592999d8~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



  北桥南桥都是用PCIe来链接
  ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/814948f2bc874196a28b8aa6d20cfad0~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


### DMA

  **作用**
   在使用pinned memory做数据拷贝以后，系统使用DMA，可以更充分的利用PCIe的带宽。如果不使用DMA拷贝，则系统无法充分利用PCIe的带宽












## 其他

### restrict

  **作用**

   * 能帮助编译器更好的优化代码，生成更有效率的汇编代码
   * 指针是访问一个数据对象的唯一且初始的方式，即它告诉编译器，所有修改该指针所指向内存中内容的操作都必须通过该指针来修改，而不能通过其它途径（其它变量或指针）来修改。

  **优点**
   * 通过重排序和通用子表达式消除等方式减少内存访问
   * 通过重排序和通用子表达式消除等方式减少数据计算次数

  **缺点**

   使用的寄存器数目增加

  **例子**

   ```cpp
   // 不使用restrict
   void foo(const float* a,
          const float* b, 
          float* c) {
      c[0] = a[0] * b[0];
      c[1] = a[0] * b[0];
      c[2] = a[0] * b[0] * a[1];
      c[3] = a[0] * a[1];
      c[4] = a[0] * b[0];
      c[5] = b[0];
      ...
   }

   // 使用restrict后compilier优化的对应代码
   void foo(const float* __restrict__ a,
          const float* __restrict__ b,
          float* __restrict__ c)
   {
      float t0 = a[0];
      float t1 = b[0];
      float t2 = t0 * t1;
      float t3 = a[1];
      c[0] = t2;
      c[1] = t2;
      c[4] = t2;
      c[2] = t2 * t3;
      c[3] = t0 * t3;
      c[5] = t1;
      .
   }
   ```
