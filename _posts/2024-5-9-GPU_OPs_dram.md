---
layout: article
title: GPU中的DRAM硬件技术
key: 100011
tags: GPU DRAM 硬件优化
category: blog
date: 2024-05-09 14:31:05 +08:00
mermaid: true
---


# GPU中的DRAM硬件技术


 本部分主要讲述global memory的硬件实现DRAM


## Bit Line & Select Line

 ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_3.png)

 ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_2.png)

  **原理**

   * 一个电容决定一个bit， 通过select线选择读取哪个bit line，然后通过bit line选择对应的bit。
   * 需要不停检查和电容充放电，因此叫做DRAM

  **特点**

  * bit line的容量大，存在电容耦合与信号衰减，同时更容易遇到数据冲突，同时bit line的带宽有限。
  * 每个bit电容，需要信号放大器放大，进一步限制传统DRAM设计

 
## Core Array & Burst的DRAM数据传输

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

  **DRAM 设计思想（局部性原理）**

   * 在DRAM硬件实现中，为了提高DRAM的速度，充分利用局部性思想，通过添加buffer缓存，将core array -> column latches / buffer的时间转换为buffer -> mux pin interface时间，极大地提高速度
     * 该方法从数据存储->数据读取中利用局部性思想


## Multiple Banks技术

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

## Multiple Channels技术

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_7.png)


  * 一般的GPU processor要求带宽达到128GB/s，HBM2要求带宽为898GB/s，在使用multiple banks后仍未满足要求（使用DDR DRAM clock为1GHz，每秒能传送8 bytes/ 1 words，得到的传输速率为16GB/s）因此需要多个channels


## 数据Interleaved（交织）分布技术

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_8.png)


  **数据交织原因**

   * 充分利用channel的带宽，实现max bandwidth。
   * 允许在core array -> buffer 的时间进行转化成多个channels获取数据（并行），减少总的时间。

  **如何实现数据交织**

  将数据平均分布在channels对应的banks中（上图中）

