---
layout: article
title: GPU 内存模型优化方向
key: 100024
tags: GPU 优化 内存
category: blog
date: 2024-05-07 15:27:23 +08:00
mermaid: true
---



## 内存结构

 ![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/gpumemory_1.png)




 **内存延迟** 

  | type                       | clock cycle                             |
  | -------------------------- | --------------------------------------- |
  | register                   | 1                                       |
  | shared memory              | 5                                       |
  | local memory               | 500 (without cache)                     |
  | global memory              | 500                                     |
  | constant memory with cache | 1(same as register)-5(same as L1 cache) |
  | L1 cache                   | 5                                       |


 **内存分类**

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


**内存模型对GPU编程中优化方向**

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


**与锁有关的优化方向**

 * global memory -> shared memory
 * global memory -> texture memory
 * global memory -> warp/block

**解决数据竞争与数据一致性的锁与内存屏障**

 * 数据竞争 -> 锁
 * 数据一致性 -> memory barrier / fence

**其他**

 * 零拷贝内存
 * Unified Memory
 * Weakly-Ordered Memory Model

