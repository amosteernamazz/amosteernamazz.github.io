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

GPU有负责不同功能的计算模块

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/cpuvsgpu_1.png)

Shader Core（渲染核/着色器核心）：并行处理多个任务，如着色、贴图处理、几何计算等

Texture Unit（纹理单元/tex）：负责纹理映射和处理。在渲染过程中，Texture Unit会对图形应用纹理，使得物体看起来更真实和细致。

Input Assembly（输入组装）：GPU从内存中读取顶点和索引缓冲区；确定如何连接顶点以形成三角形；将这些数据传递给管线的后续阶段，为进一步的图形处理做准备。

Rasterizer（光栅化器）：将图形从几何描述（如顶点数据）转换为像素数据；确定最终图像中哪些像素应该被涂上颜色，以及这些颜色是什么。

Output Blend（输出混合）：负责将渲染的像素与帧缓冲区中的现有像素进行混合；它可以产生透明、半透明和其他的视觉效果。

Video Decode（视频解码）：负责解码视频数据，以便在GPU上进行进一步的处理或显示；这使得GPU能够支持视频播放和视频游戏等功能。

Work Distributor（工作分配器）：负责将图形处理任务分配给GPU上的不同处理单元；确保任务能够高效、并行地执行，从而提高整体性能。

SMs: GPU中的核心处理单元。每个SM包含多个处理核心，这些核心能够并行执行指令，从而加速图形渲染和计算密集型任务。SMs通常有自己的寄存器文件、指令缓存和共享内存，这使得它们可以独立地执行任务，同时与其他SMs协同工作。

SPs: SMs中的基本处理单元。每个SM包含多个SPs，这些SPs共享控制逻辑和指令缓存。这意味着当SM从指令缓存中取出一条指令时，所有SPs都可以同时执行这条指令。这种设计使得GPU能够并行处理大量数据，从而大大提高了处理速度。



* 为了什么设计

GPU的设计主要是为了实现高吞吐量

CPU的设计则更注重低延迟。CPU需要处理各种不同的数据类型，进行逻辑判断，以及处理分支跳转和中断等复杂情况，这些都需要消耗一定的时间。因此，CPU在设计上需要优化指令的执行流程，减少等待时间，从而实现低延迟。

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/cpuvsgpu_2.png)



## Ideas to design throuput oriented GPU

* Idea 1 ： 去除CPU中让CPU serialize code运行更快的

CPU中包含out of order execution, branch predictor, memory prefetch等机制让CPU运行serialize code fast，但是这些部分占用很大的memory和chip。

<!--more-->

GPU去除这些部分。

![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/cpuvsgpu_3.png)




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



![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/cpuvsgpu_4.png)

* idea 5：hide latency instead of reduce latency

*fancy cache 指高级缓存系统，将缓存进行设计，使用高度并行化的存储器结构或数据命中率高*
*prefetch logic 使用预取技术完成对未来数据的预先读取，避免访问慢存储器，prefetch logic指实现预取功能的逻辑电路，常与缓存系统结合使用，实现高效数据预取功能*
CPU通过fancy cache + prefetch logic来avoid stall

GPU通过lots of thread来hide latency。这依赖于fast switch to other threads, 也就需要keep lots of threads alive.


![](https://github.com/amosteernamazz/amosteernamazz.github.io/raw/master/pictures/cpuvsgpu_5.png)


* GPU Register 特点

GPU的register通常很大，在V100里与half L1 cahce+shared memory一样大

经常也被叫做inverted memory hierchy

