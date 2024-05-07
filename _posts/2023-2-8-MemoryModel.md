---
layout: article
title: GPU Memory Model
key: 100024
tags: GPU架构
category: blog
date: 2023-02-08 00:00:00 +08:00
mermaid: true
---

***replay如何解决？***

***Warp-aggregated当中硬件如何实现atomic***

# Memory Model


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
 ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_36.png)
  
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

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_37.png)

  
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
  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_38.png)


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

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_39.png)


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
   ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_40.png)



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
  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_41.png)




## zero-copy memory

 **硬件支持**
  * 支持Unified Memory或NVLink等技术
  * 延迟在几ms以下，仅仅只快于CPU、GPU数据传输


 **零拷贝内存定义与使用条件**
  * 本质上是将主机内存映射到device memory
  * GPU threads 直接获得host的零拷贝内存
  * 如果在host/device修改了内存，需要**synchronize()**来保证一致性。

 **优点**

  * 当device memory不够用，可以使用零拷贝内存
  * 可以避免从device 到host 之间的数据传输
    * 但是存在数据竞态问题

 **缺点**

  * 对于GPU设备连接到PCIs总线，只针对特殊场景效果较好。零拷贝速度比global memory/device memory要慢。
    * 在集成架构中，当CPU和GPU在一个芯片上，且共享内存，零拷贝内存不用走PCIs总线，对性能和可编程性都有帮助。
  * 潜在的性能问题和数据竞争条件。
  
 **应用**
  * 在需要频繁地传输大量数据时，如在**机器学习**、**图形处理**和**数据分析**等应用中，可以使用GPU零拷贝内存来减少主机和设备之间的数据传输时间
  * 在需要实现低延迟数据传输时，如在**实时音视频处理**、**高性能计算**等领域，可以使用GPU零拷贝内存来避免CPU和GPU之间的数据复制，从而减少传输延迟，提高处理速度
  * 在需要同时使用CPU和GPU时，可以使用GPU零拷贝内存来实现共享内存，避免CPU和GPU之间的数据复制，从而提高数据传输效率和程序性能

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
  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_42.png)

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


**barrier/fence与atomic的关系**
 * atomic保证了共享内存位置上的操作没有**数据竞争**和其他同步问题，但无法解决**数据读取的顺序问题**
   * 针对读写问题，读前值还是读后值

**解决方法**
 * block内的barriers方法
 * fence方法

### Explicit Barriers

**__syncthreads()原语**
  * __syncthreads()同步的是**可以到达**`__syncthreads()`函数的**线程**，而不是所有的线程（应避免部分线程到达，为了避免数据一致性问题）

**推荐方法**

  ```c++
  __shared__ val[];
  ...
  if(index < n){
    if(tid condition){
      do something with val;
      
    }
    __syncthreads();
    do something with val;
    __syncthreads();
  }
  ```


  **错误**

  如果不同线程操作同一块内存的话，对数据竞态与一致性有影响

   ```cpp
   if (threadID % 2 == 0) {
     __syncthreads();
   } else { 
     __syncthreads();
   }
   ```

  如果不同线程操作同一块内存的话，对数据竞态与一致性有影响

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


### Memory Fence

 **Memory Fence与barrier**
  * 区别
    * Memory fence是一种**硬件层面**的同步机制，它可以控制线程对内存的访问顺序和可见性，确保对内存的访问操作完成之前，**先前的访问操作**已经**完成**；Memory fence只影响调用它的线程，而不影响其他线程；Memory fence可以分为Threadfence和Blockfence两种类型
    * Barrier是一种**软件层面**的同步机制，它可以控制线程的执行顺序和同步，确保在某个点上的所有线程**都完成它们的工作**，然后再继续执行下一个任务。Barrier只有在同一个block的线程之间才有效，不同block之间的线程无法互相等待

 **保证**
  * 使用fence保证在下面的内存操作执行前，前面的内存操作都已经完成
  * 保证**内存访问同步**

  **`void __threadfence_block();`**

   * 保证同一个block中thread在fence之前写完的值对block中其它的thread可见，不同于barrier，该function不需要所有的thread都执行。
   * 使用该函数效果类似__syncthreads()函数

  **`void __threadfence();`**

   * 应用范围在grid层面
   * 主要针对不同的block进行同步

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

   * 如果没有memory fence的话，可能flag会首先被atomic设置了，然后才设置global memory的数据。这样另一个block在读取到flag以后就开始读取global memmory的值可能就是不对的

   * 通过使用memory fence，确保在fence后面读取memory的数据确实是fence之前写入的数据

### Volatile

 **目的**
  * 在读取和写入该变量时需要**强制使用内存操作**，而不是将变量缓存到寄存器中


 **应用**
  * 用于共享内存和全局内存中的变量：由于共享内存和全局内存中的数据可能会被其他线程修改，因此在读取和写入这些数据时需要使用Volatile关键字，以确保数据的一致性和正确性。

  ```c++
  __shared__ volatile int shared_data[1024];

  __global__ void kernel() {
      int tid = threadIdx.x;
      shared_data[tid] = tid;

      // 使用Volatile关键字确保其他线程可以立即读取到写入的值
      volatile int value = shared_data[(tid + 1) % blockDim.x];

      // ...
  }
  ```

  * 用于GPU和CPU之间的变量：由于GPU和CPU之间的数据传输可能会受到许多因素的影响，例如数据缓存、数据预取等，因此在进行数据传输时需要使用Volatile关键字来确保数据的正确性和一致性。

  ```c++
  __host__ void transfer_data(int *data, int n) {
      int *dev_data;
      cudaMalloc(&dev_data, n * sizeof(int));
      cudaMemcpy(dev_data, data, n * sizeof(int), cudaMemcpyHostToDevice);

      // 使用Volatile关键字确保CPU可以立即读取到GPU写入的值
      volatile int result;
      cudaMemcpy(&result, dev_data, sizeof(int), cudaMemcpyDeviceToHost);

      // ...
  }
  ```




## Hardware Implementation

### PCIe

  **用处**
   GPU与CPU通过PCIe链接

  **结构**
   * PCIe由多个link组成
   * 每个link包含多个lanes
   * 每个lane为1-bit width（由4 wires组成，构成16GB/s）

  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_43.png)



  北桥南桥都是用PCIe来链接
  ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_44.png)


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
