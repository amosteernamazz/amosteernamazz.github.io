---
layout: article
title: GPU 各版本
key: 100024
tags: GPU架构
category: blog
date: 2023-02-08 00:00:00 +08:00
mermaid: true
---




# Computation Capacity





## 1.x Tesla



## 2.x Fermi

* each SM contain

1. 32 single-precision CUDA cores
   1. 每个CUDA Core含有一个Integer arithmetic logic unit (ALU)和一个Floating point unit(FPU)，并且提供了对于单精度和双精度浮点数的成绩累加指令
2. 两个Warp Scheduler和两个Dispatch Unit
   1. <font color = red>每个warp会被分发到一个Cuda Core Group(16个CUDA Core), 或者16个load/store单元，或者4个SFU（sin/cos特殊指令）上去真正执行</font>。
   2. 每次分发只执行 一条 指令，而Warp Scheduler维护了多个（比如几十个）的Warp状态。
   3. <font color = red>在出现分支的线程的时候，会发生线程的浪费</font>
3. 64 KB on-chip configureable shared memory / L1 cache (48KB shared memory, 16KB L1 cache)
4. 4 SFU
5. 16 load/store unit (Figure 3.1)
6. 32 KB register file

![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d6b359d36ce746d3a1718faf82accef8~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


<!--more-->


* across SM

1. 16 SM total
2. GigaThread engine : 将blocks分配给SM warp schedulers
3. 768 KB L2 cache


![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/948cef6e599149b488b0ae8755ced503~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

* concurrency

1. 16-way concurrency. 16 kernels to be run on device at same time








## 3.x Kepler GK110

* Each SM contain

1. 192 single-precision CUDA core 12 * 16
2. 64 double-precision unit (DP Unit)
3. 32 special function unit for single precision float (SFU)
4. 32 load/store unit (LD/ST)
5. 4 warp scheduler and 8 dispatch
   1. dynamic scheduling
   2. four warp scheduler select four warp to execute
   3. at every instruction issue time, each scheduler issues two (因为2 dispatch per scheduler) independent instructions for one of its assigned warps that is ready to execute（每次调度器往都会收到2条独立指令，因为调度器有两个warp dispatch）
6. <font color = red>引入GPUDirect技术，支持GPU可以利用DMA直接访问第三方设备，如SSD、NIC等</font>

7. 64 KB on-chip configureable shared memory / L1 cache 
   1. L1 cache for load local memory nd register spill over

   2. cc 3.5 3.7 可以opt-in to cache global memory access on both L1 & L2 通过compiler `-Xptxas -dlcm=ca`，但是默认global memory访问不经过L1 

8. 64 KB register file
9. per SM read-only data cache 
   1. 用于read from constant memory
10. per SM read-only texture cache 48kb
   1. cc 3.5 3.7 可以用于read global memory
   2. 也可以被texture memory使用
   3. 与L1 cache不是unified的


![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/06aa515e59d848039949fa4b284446f8~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


* across SM contain

1. 1.5MB L2 cache 用于 local memory & global memory



* feature

1. <font color = red>32 hardware work queue for hyper-q</font>
2. dynamic parallel


![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0d1f28c1a597490a89cdd5cca9a30bf1~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



## 5.x Maxwell GTX980

* each SM contain

1. 128 CUDA core for arithmetic op
2. 32 special function unit for single precision float
3. 4 warp scheduler
   1. dynamic scheduling
   2. each warp scheduler issue 1 instruction per clock cycle
4. L1 cache/texture cache of 24 KB
   1. 在某些条件下可以通过config来用于访问global memory
   2. default not enable L1 cache for global memory access
   3. L1 cache 与 read only texture cache 是unified的，这点与3.x是不一样的。

5. shared memory 64KB/96KB
   1. 这里shared memory与L1 cache不再是同一个chip了

6. read-only constant cache
   1. 用于constant memory access

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/586cbda331e145488223759c3a569941~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



* across SM

1. L2 cache 用于 local or global memory


![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/900f21e777124e00a9e12ca5eadd8e47~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

## 6.x Pascal

  **第一个考虑deep learning的架构**


* Each SM core

1. 64 (cc 6.0) / 128 (cc 6.1 & 6.2) CUDA core for arithemetic
   1. 每个线程<font color = red>可以使用的register变多，每个SM可以并发更多的线程，同时每个线程使用的shared memory变大，带宽变大</font>。
2. 32(P100) special function unit for double precision float
   1. <font color = red>增加对double的支持，同时可以转为两个float</font>
3. 16 (cc 6.0) / 32 (cc 6.1 & 6.2) special function unit for single precision float 
4. 2 (cc 6.0) / 4 (cc 6.1 & 6.2) warp scheduler 
   1. dynamic assign warp to warp scheduler. When an SM is given warps to execute, it first distributes them among its schedulers
   2. each scheduler issues one instruction for one of its assigned warps that is ready to execute

5. read-only constant cache
   1. read from constant memory space

6. L1/texture cache of size 24 KB (6.0 and 6.2) or 48 KB (6.1),
   1. 用于read global memory
7. <font color = red>采用unified memory 执行GPU代码后不用再进行GPU与CPU之间的同步操作</font>
8. shared memory of size 64 KB (6.0 and 6.2) or 96 KB (6.1).
9.  <font color = red>提供nvlink 用于可以用于CPU与GPU、GPU与GPU之间的点对点通信，其中P100提供4个nvlink接口</font>
10. <font color = red>引入HBM2通信机制</font>

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/bfddc869d57b4f1bbcc4e6fe3f378ba6~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

* across SM

1. L2 cache 用于 local or global memory


![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/cf82eaab8356473e9ebdae7c47d9f958~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


## 7.x Volta    & Turing

  **volta相对pascal来说是版本的大更新**

* each SM

1. 4 processing block, each contain 
   1. 16 FP32, 16 INT32, 8 FP64, 2 tensor core, L0 instruction cache, one warp scheduler, one dispatch unit, and 64KB Register file
      1. 引入L0指令缓存，shared register与上版本一致，提高单线程使用资源，
      2. schdule和dispatch unit变为32threads/clk
      3. 再次将shared memory和L1合并
      4. 将core拆解为int32和float32核，可以同时执行int和float指令
      5. <font color = red>加入tensor core用于矩阵计算（D = AB+C）其中A、B为FP16，C、D为FP16或FP32。</font>
2. <font color = red>nvlink提供6个接口</font>
3. <font color = red>引入独立线程调度</font>
4. <font color = red> 引入合作组</font>
5. read only constant cache 用于 constant memory space
6. unified L1 & shared memory of size 128 KB (volta) / 96 KB (Turing)
   1. can be configued between l1 & shared memory. 
   2. driver automatically configures the shared memory capacity for each kernel to avoid shared memory occupancy bottlenecks while also allowing concurrent execution with already launched kernels where possible. In most cases, the driver's default behavior should provide optimal performance. 自动config shared memory的大小，大多数情况是optimal的
   3. default enable L1 cache for global memory access
   4. smem bank same as Fermi.


![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4d458dbe22314e15929899cc0a60e7f1~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/28d6ae1c0d824e5ca28b93a32a7c231c~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


* Across SM

L2 cache


  **volta相对pascal来说是版本的大更新**
1. <font color = red>将FP算数指令与int指令混合在一起</font>
2. <font color = red>为L1和shared memory引入统一配置</font>
3. <font color = red>tensor core支持int8/int16/binary的支持，加速DL的inference</font>
4. <font color = red>提供RT core</font>

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/497822a434584ae49199e84e356e6192~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b7b680eceafe4232914afe9895f7c9ec~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



## 8.x Ampere

   **大版本**

### Resource

* each SM have 

1. 64 FP32 cores for single-precision arithmetic operations in devices of compute capability 8.0 and 128 FP32 cores in devices of compute capability 8.6,
2. 32 FP64 cores for double-precision arithmetic operations in devices of compute capability 8.0 and 2 FP64 cores in devices of compute capability 8.6
3. 64 INT32 cores for integer math,
4. 4 mixed-precision Third Generation Tensor Cores supporting half-precision (fp16), __nv_bfloat16, tf32, sub-byte and double precision (fp64) matrix arithmetic (see Warp matrix functions for details),
   1. <font color = red>相对于前面支持FP16/INT8/INT4/1，引入FP32/BF16/FP64支持</font>
5. 16 special function units for single-precision floating-point transcendental functions,
6. 4 warp schedulers.
7. <font color = red>引入细粒度的结构化稀疏，用于DL中的reference</font>
   1. 首先使用正常的稠密weight训练，训练到收敛后裁剪到2:4的结构化稀疏Tensor，然后走fine tune继续训练非0的weight, 之后得到的2:4结构化稀疏weight理想情况下具有和稠密weight一样的精确度，然后使用此稀疏化后的weight进行Inference. 而这个版本的TensorCore支持一个2:4的结构化稀疏矩阵与另一个稠密矩阵直接相乘。
8. <font color = red>MIG(Multi-Instance GPU)用于VM中的创建多个虚拟GPU供用户使用</font>
9. read only constant cache
10. unified L1 & shared memory
   1. can be configed 
   2. default enable L1 cache for global memory access

![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/bed507d212904e2192c66dac33f0b24b~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


### schedule

static distribute warp to scheduler

each scheduler issue 1 instruction each clock cycle



### global and shared memory

global same as 5.x

shared memory bank same as 5.x

shared memory configuration same as 7.x


## 9.x Hopper & Lovelace



![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6c2c7fc7bdab4a4db55fca9945abeeeb~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/dde723064b9d4bee9c56f39c0a10e0f1~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

