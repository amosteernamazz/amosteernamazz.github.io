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

 * 确定template待确定的变量类型（其中为算数类型与数据类型）
   * `template<ArithmeticOpType op_type, typename T>`
 * 对于int8类型变量，为了区分，定义新template函数`ppl_arithmetic_scalar_int8`
 * 对于CUDA内置half类型，重写函数`ppl_arithmetic_scalar`
 * 定义自定义类型half8_的实现

### 算数算子

### 逻辑算子
### 比较算子