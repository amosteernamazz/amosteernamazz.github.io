---
layout: article
title: GPU global memory内存合并和内存对齐
key: 100013
tags: GPU GlobalMemory 软件优化
category: blog
date: 2023-02-08 00:00:00 +08:00
mermaid: true
---


# Global Memory

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

