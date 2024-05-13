---
layout: article
title: GPU global memory内存合并
key: 100013
tags: GPU GlobalMemory 软件优化
category: blog
date: 2023-02-08 00:00:00 +08:00
mermaid: true
---



# global memory 内存合并与优化

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



## 内存合并的优化方法

### 数据分布和内存对齐

 * GPU对于内存数据的请求是以wrap为单位，而不是以thread为单位。如果数据请求的内存空间连续，请求的内存地址会合并为一个warp memory request，然后这个request由一个或多个memory transaction组成。具体使用几个transaction 取决于request 的个数和transaction 的大小。（前文所述）
 * 其中从一个request得到多个memory transactions时，时间消耗集中于多次的memory transactions流程中，因此有必要针对数据的分布和对齐方式进行优化

#### 内存对齐优化

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

#### 数据分布优化

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




### 访问模式

#### 访问合并


#### 模式优化



#### 内存分块


### 线程映射和索引

### 数据重用和缓存



## Global Memory 读流程

  注意： GPU L1 cache is designed for spatial but not temporal locality. Frequent access to a cached L1 memory location does not increase the probability that the data will stay in cache. L1 cache是用于spatial（连续读取array）而不是temporal（读取同一个位置的），因为cache line很容易被其余的thread evict。

  

  **内存读性能：global memory load efficency**

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_11.png)



   nvprof.gld_efficency metrics衡量了此指标



  **内存读模型：Simple model**

   *  在128 bytes/32 bytes的模式下，会产生128 bytes/ 32 bytes / 64 bytes的memory transaction （32 bytes当four segment的时候也会是128 bytes）。在L1中 memory被合并分块为 32-, 64-, or 128- byte memory transactions，在L2中被分块为32 bytes。



### Read-only texture cache

   * 使用条件：CC 3.5+ 可以使用read only texture cach
   * cache line大小：The granularity of loads through the read-only cache is 32 bytes. 



### CC 2.x Fermi 

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
     * 如果每个thread请求的数据大于4 bytes(32 * 4 = 128)，则会被切分为多个128 bytes memory request来进行。
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




### CC 3.x Kepler cache line大小

   * 3.x default 使用 L2 cache，不使用L1 cache

   * 3.5 / 3.7 可以使用read only texture cache

   * 3.5 / 3.7 可以config使用L1 cache

   * L1 cache line size 128 bytes

   * L2 cache line size 32 bytes

   为什么L2 cache 需要以1、2、4倍数传输：为了避免DMA fetch DMA FETCH: The DMA supports an AXI bus width of 128/64 bits. In the case where the source descriptor payload ends at a non-128/64 bit aligned boundary, the DMA channel fetches the last beat as the full-128/64 bit wide bus. This is considered an over fetch.
   当使用L2 cache only的时候，memory transaction是32 bytes. Each memory transaction may be conducted by one, two, or four 32 bytes segments。可以减少over-fetch

  **L1/L2读取顺序**
   * 当使用L1 + L2时候，memory transaction是128 bytes。
   Memory request 首先会去L1，如果L1 miss会去L2，如果L2 miss会去DRAM。




### CC 5.x Maxwell

   * 5.x default使用L2 cache，32 bytes transaction

   * 5.x 可以使用read only texture cache，32 bytes transaction

   * 5.x 可以config使用L1 cache（default不使用）


### CC 6.x Pascal



## Global Memory 写流程

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



## global memory操作与硬件的关系

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




### global memory 内存合并的具体内容

  **global memory 数据流向**

   * global memory作为所有单元可以直接获取的数据源，其速度在所有的GPU内存结构中处于最慢。因此数据流向到GPU处理单元需要其他缓存结构。
   * 下图为早期GPU结构，其中global memory request经过L2，然后再往计算单元靠近经过shared memory、local memory、L1、read only && constant memory，在往上为registers

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_9.png)

