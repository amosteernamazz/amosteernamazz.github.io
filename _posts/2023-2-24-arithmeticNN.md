---
layout: article
title: 算子库
key: 100030
tags: C++ 算子库 CUDA GPU算法 算法
category: blog
date: 2023-02-26 00:00:00 +08:00
mermaid: true
---

# 基础

## 快速整数除法

**步骤**

![](https://images0.cnblogs.com/blog/258391/201412/311730547311897.png)

**论文链接**
[快速整数除法论文链接](https://gmplib.org/~tege/divcnst-pldi94.pdf)

**代码实现**
  ```c++
  #ifndef PPL_CUDA_DIVMOD_PAST_H_
  #define PPL_CUDA_DIVMOD_PAST_H_
  #include <stdint.h>
  #include <cuda_runtime.h>
  struct DivModFast{
    DivModFast(int d =1){
      d_ = (d==0) ? 1 : d;
      for(l_ = 0;;++l_){
        if((1U << l_) >= d_){
          break;
        }
      }
      uint64_t one = 1;
      uint64_t m = ((one << 32) * ((one << l_) - d_)) / d_ +1;
      m_ = static_cast<uint32_t> (m);
    }

  __device__ __inline__ int div(int index){
    uint tm = __umulhi(m_, index);
    (tm + index) >> l_;
  }
  __device__ __inline__ int mod(int idx) const
  {
    return idx - d_ * div(idx);
  }
  __divice__ __inline__  void divmod(int index, int& quo, int& rem){
      quo = div(index);
      rem = index - quo * d_;
    }

    uint32_t d_; // divisor
    uint32_t l_; // ceil(log2(d_))
    uint32_t m_; // m' in the paper

  }
  #endif

  ```


# NN 算子

## concat算子

### 


