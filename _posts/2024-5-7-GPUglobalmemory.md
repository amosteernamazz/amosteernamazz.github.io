---
layout: article
title: GPU global memory 
key: 100024
tags: GPU GlobalMemory
category: blog
date: 2023-02-08 00:00:00 +08:00
mermaid: true
---


## Global Memory


### 带宽检测方法

带宽有理论带宽和实际带宽，该部分是实际带宽，包括CPU方法和GPU方法

#### 用于带宽检测的计时器

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

##### GPU计时


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




##### Visual profiler 内核性能分析工具



 **<font color=red>requested throughput</font> 和 <font color = red>global throughput</font>** 

   * 系统的global throughput 相当于物理理论带宽（考虑到cache line 要一起传送的带宽）
   * 系统的requested memory 相当于实际带宽（并未考虑 cache line）
   * 应当让requested throughput尽可能接近global throughput。


  


### DRAM硬件与优化方法

 本部分主要讲述global memory的硬件实现<font color = red>DRAM</font>


#### Bit Line & Select Line


 ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_2.png)


  **原理**
   * 一个<font color= red>电容</font>决定一个bit， 通过<font color= red>select线</font>选择读取哪个bit，一个<font color= red>bit line</font>操作只能读取一个bit的数据，即从多个select中选择一个
   * 需要不停<font color= red>检查</font>和<font color= red>电容充放电</font>，因此叫做DRAM



  **特点**

  * <font color= red>bit line的容量大</font>，导致速度慢

  * 每个<font color= red>bit电容</font>，需要信号放大器放大

 ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_3.png)



#### DRAM 的数据传输(Core Array & Burst)

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_4.png)


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

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_5.png)


![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_6.png)



  **引入原因**
   * 从<font color = red>core array到buffer的时间长</font>，单个传输导致实际使用的<font color = red>bus interface数据bandwidth未被完全利用</font>（$T_c:T_b = 20:1$)。如果只使用一个bank导致带宽大部分时间为空闲。
   * 所以需要在一个bus 上使用多个bank，来充分利用bus bandwidth。如果<font color = red>使用多个bank，大家交替使用interface bus</font>，保证bus不会空闲，保证每个时间都有数据从bus传送过来


  **一个bus需要多少个bank？**

   * 如果访问core array与使用bus传输数据的时间比例是20:1，那么一个bus<font color = red>至少需要21个bank</font>才能充分使用bus bandwidth。
   * 一般bus有更多的bank，<font color = red>不仅仅是ratio+1</font>
     * <font color = red>为减少burst的可能性，将数据分布在不同的bank</font>，可以节约总时间
     * ***<font color = purple>每个bank可以存储的诗句有限，否则访问一个bank的latency会很大。</font>***

  
  
  

#### Multiple Channels

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_7.png)



  * <font color = red>一般的GPU processor要求带宽达到128GB/s，HBM2要求带宽为898GB/s，在使用multiple bank后仍未满足要求</font>，因此使用multiple channel的方法。
  * 假设使用DDR DRAM clock为1GHz，每秒能传送8 bytes/ 1 words，得到的传输速率为16GB/s，因此需要多个channel
  


#### 数据分布方法Interleaved（交织） 

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_8.png)



  **数据交织原因**
   * 充分<font color = red>利用多个channels的带宽，实现max bandwidth</font>。
   * 允许在<font color = red>core array -> buffer 的时间进行转化成多个channels获取数据</font>，减少总的时间。
  **如何实现数据交织**
  将数据平均分布<font color = red>在channels对应的banks</font>中
  

 
 
 

### global memory 内存合并和内存对齐及其优化方法


#### global memory 内存合并

  **global memory 内存合并原因**

   * 在GPU中对于<font color = red>内存数据的请求是以wrap 为单位</font>，而不是以thread 为单位。**<font color = purple >- - - - -></font>** *warp 内thread 请求的内存地址会合并为一个<font color = red>warp memory request</font>，然后这个request 由<font color = red>一个或多个memory transaction</font> 组成*。
   * 具体使用几个transaction 取决于request 的个数和transaction 的大小。

  **global memory 数据流向**
   * global memory request<font color = red>一定会经过L2</font>，是否经过<font color = red>L1 </font>取决于<font color = red>cc 和code</font>，是否经过read only texture cache取决于<font color = red>cc 和code </font>。

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_9.png)


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





#### 内存对齐优化方法

 L1 cache line = 128 bytes, L2 cache line 32 bytes，warp的<font color = red>内存请求起始位置位于cache line的偶数倍</font>，为了保证对global memory 的读写不会被拆为多个操作，应保证存储对齐。



  **image cache line**

   为了<font color = red>防止对不同row下的读写产生多个memory segment</font>，导致速度变慢，在<font color = red>每一行末尾加入padding</font>

   padded info叫做 `pitch` 

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_10.png)



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



#### Global Memory 读流程

  注意： GPU L1 cache is designed for spatial but not temporal locality. Frequent access to a cached L1 memory location does not increase the probability that the data will stay in cache. L1 cache是用于spatial（连续读取array）而不是temporal（读取同一个位置的），因为cache line很容易被其余的thread evict。

  

  **内存读性能：global memory load efficency**

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_11.png)



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




     * 请求四个32 bytes的scatter分布，相对<font color= red>128 bytes的方式，缓存利用率高</font>。
      ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_20.png)




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



#### Global Memory 写流程

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
      ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_21.png)




     * 128 bytes 的request，请求内存大小为64 bytes的连续内存空间，则需要2个transactions

      ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_22.png)



#### global memory操作与硬件的关系

  第一次访问，<font color = red>全部4个数据都放到buffer</font>里

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_23.png)


  第二次访问使用后面两个数据（连续内存访问），<font color = red>直接从buffer里读取数据，不用再去core array</font>
  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_24.png)



  **burst 原因** 
   * 在从core array -> buffer 的过程需要的时间长，在每一次从core array 到buffer 的过程中，传输burst数据，在每一次读取中，应让数据充分使用。因为<font color = red>两次core array -> buffer 时间远远大于两次 buffer -> bus的时间</font>。

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_25.png)
  


   * 蓝色的部分是core array到buffer的时间。红色的部分是buffer到pin的时间


 
 
 


### Minimize host to device transfer


#### Batch Small Transfer

  * 为避免多个small memory copy，使用one large memory copy，将<font color = red>多个small memory打包成为一个large memory，然后进行copy</font>

#### Fully utilize transfer

  * 尽量将<font color = red>传输次数减少</font>，如果GPU计算并不方便，也使用，减少数据传输次数。

