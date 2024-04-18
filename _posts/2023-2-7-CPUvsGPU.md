---
layout: article
title: CPU VS GPU
key: 100023
tags: GPU架构
category: blog
date: 2023-02-07 00:00:00 +08:00
mermaid: true
---


## CPU vs GPU


* 什么是GPU

GPU是heterogeneous chip. 有负责不同功能的计算模块

![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/059c4df7de5e4b00bb76f43bf7e5a2b1~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




SMs: streaming multiprocessors

SPs: streaming processors : each SM have multiple SP that share control logic and instruction cache



* 为了设么设计

GPU design for high throughput, don't care about throughput so much

CPU design for low latency

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/70174e01e80f466bbc1de536aa69ac9a~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



* CPU GPU

CPU : multicore system : latency oriented 

GPU : manycore / many-thread system : throughput oriented



## Idea to design throuput oriented GPU

* Idea 1 ： 去除CPU中让CPU serialize code运行更快的

CPU中包含out of order execution, branch predictor, memory prefetch等机制让CPU运行serialize code fast，但是这些部分占用很大的memory和chip。

<!--more-->

GPU去除这些部分。

![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7bb6231977b642f5a066ce9898d22574~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)




* Idea 2 ：larger number of smaller simpler core

相比起使用small number of complex core, GPU的工作经常simple core就可以处理。

但也带来了挑战，需要programmer expose large parallel从而充分利用全部的core



* idea 3：让simple core共享instruction stream，减少负责Fetch Decode的芯片面积

因为很多工作都是parallel的，所以多个small simple core共享instruction stream就可以，减少了chip上负责instruction stream的部分。

SIMT single instruction multiple threads. 

SIMT 与 SIMD 有一些不一样。SIMT可以平行thread，而SIMD只可以平行instruction



* idea 4：使用mask来解决branching

在CPU中使用branch prediction

在GPU中，使用mask来解决branching



![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/031601e629bb43ed94c46b6e9ec095e0~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

* idea 5：hide latency instead of reduce latency

*fancy cache 指高级缓存系统，将缓存进行设计，使用高度并行化的存储器结构或数据命中率高*
*prefetch logic 使用预取技术完成对未来数据的预先读取，避免访问慢存储器，prefetch logic指实现预取功能的逻辑电路，常与缓存系统结合使用，实现高效数据预取功能*
CPU通过fancy cache + prefetch logic来avoid stall

GPU通过lots of thread来hide latency。这依赖于fast switch to other threads, 也就需要keep lots of threads alive.


![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1e1ba3f0082c45fa8289da551025c400~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)


* GPU Register 特点

GPU的register通常很大，在V100里与half L1 cahce+shared memory一样大

经常也被叫做inverted memory hierchy

