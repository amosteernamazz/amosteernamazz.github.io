---
layout: article
title: 算子库
key: 100030
tags: C++ 算子库
category: blog
date: 2023-02-17 00:00:00 +08:00
mermaid: true
---

***为什么需要定义一种类型为half8_***


# 算子库方法
 算子库包括算数算子（算数运算、逻辑运算、关系运算）、reduce算子（argmax、argmin等）、format算子（类型转换算子）、unary算子（cast、clip、relu、neg、not、zero等）、nn算子（层归一化、conv、lstm、one-hot、pooling-max、topk、gemm等）、数据大小的转变算子（expand、gather、pad、split、tile、reshape等）

## 算数算子

### 算数算子

 * 确定template待确定的变量类型（其中为算数类型与数据类型）
   * `template<ArithmeticOpType op_type, typename T>`
 * 对于int8类型变量（有上下范围时候的差异），为了区分，定义新template函数`ppl_arithmetic_scalar_int8`
 * 对于CUDA内置half类型，重写函数`ppl_arithmetic_scalar`
 * 定义自定义类型half8_的实现

#### static方法

 **计算两个输入张量的形状并在需要时对它们进行填充，以使它们具有相同的最大维度数**
  * 计算输入张量形状的最大维度数`tensor_shape0`、`tensor_shape1`
  * 判断哪个需要填充的张量形状`pad_tensor_shape0`、`pad_tensor_shape1`维度数较少，将其维度进行填充，使得它具有最大维度数
  * 对于需要填充的张量形状，首先将其dim_count属性设置为最大维度数
  * 在它的高维度上添加1，直到达到需要填充的维度

 **将两个输入张量的形状进行广播，使得它们在进行特定操作时具有相同的形状**
  * 获取输出张量的维度数
  * 初始化实际维度数为输出张量的维度数
  * 从输出张量的最后一个维度开始遍历，直到第二个维度
    * 检查两个输入张量在该维度上是否需要广播，即其维度大小是否不同且其中一个维度大小为 1
    * 检查其前一个维度上，是否相同
    * 如果当前维度和前一个维度都需要进行相同的广播操作，则将它们合并为一个维度，同时在输入张量和输出张量的相应位置更新维度大小。实际维度数减 1。
    * 如果当前维度和前一个维度需要进行不同的广播操作，则跳出循环
  * 更新输入张量的维度数为实际维度数

 ****

### 逻辑算子

### 比较算子