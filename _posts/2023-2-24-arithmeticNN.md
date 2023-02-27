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

**目的**
 * 用于在GPU上对输入数据进行**并行拼接**

### 实现

**广义concat函数**
 * 广义的concat函数，可以指定对不同轴进行concat
   * `concat_size`表示拼接的 tensor 数量
   * top_axis_width：拼接的轴的宽度
   * axis_offset：拼接轴在输入 tensor 中的偏移量

  ```c++
  template<typename T>
  __global__  void ppl_cukernel_concat(
    int64_t num_elems,
    const T* inputs,
    int64_t concat_size;
    int64_t axis_width;
    int64_t axis_offset;
    DivModFast num_input_fast_div,
    T* output){
      for(int64_t i = blockId.x * blockDim.x + threadId.x; i< num_elems; i +=(int64_t)gridDim.x * blockDim.x){
        int inner_index, outer_index;
        num_input_fast_div.divmod(i, outer_index, inner_index);
        int64_t output_index = inner_index + (outer_index * axis_width+ axis_offset) * concat_size;
        output[output_index] = inputs[i];
      }
    }
  ```


**广义concat函数应用**

 * 数据结构为[batch_size, height, width, channels]
 * 拼接在channels维度上进行
  ```c++
  int batch_size = 2;
  int height = 32;
  int width = 32;
  int channels = 64;
  int concat_size = batch_size * height * width;
  int top_axis_width = channels;
  int axis_offset = batch_size * height * width;
  int64_t num_elems = batch_size * height * width * (2 * channels);

  T* A = new T[num_elems];
  T* B = new T[num_elems];
  T* output = new T[num_elems];

  // 在A和B中填充数据

  const int threadsPerBlock = 256;
  const int blocksPerGrid = (num_elems + threadsPerBlock - 1) / threadsPerBlock;

  DivModFast num_elems_inner_fast(concat_size);
  ppl_cukernel_concat<<<blocksPerGrid, threadsPerBlock>>>(
      num_elems, A, concat_size, top_axis_width, num_elems_inner_fast, 0, output);
  ppl_cukernel_concat<<<blocksPerGrid, threadsPerBlock>>>(
      num_elems, B, concat_size, top_axis_width, num_elems_inner_fast, axis_offset, output);

  // 输出concatenation结果
  ```


**两个input的concat**

  ```c++
  template <typename T1, typename T2>
  __launch_bounds__(256)
  __global__ void ppl_cukernel_concat_two_inputs(
    int64_t num_elems;
    const T1* input0;
    const T1* input1;
    T2* output;){
      for(int64_t i = blockId.x * blockDim.x + threadId.x; i < num_elems ; i += (int64_t)gridDim.x * blockDim.x){
        int threadid = threadId.x;
        __shared__ T1 buffer[512];
        buffer[2 * threadid] = input0[i];
        buffer[2 * threadid + 1] = input1[i];
        T2* buffer_T2 = reinterpret_cast<T2*> buffer;
        output[i] = buffer_T2[threadid];
      }
    }
  ```
  
  * 改进方向
    * `T1`、`T2`转化为向量向量化加载
      * 较难实现
    * loop unrolling，一个线程不止处理两个元素，处理4、8个
      * 与目的不一致
    * 异步内存拷贝，如果输入和输出数据不在设备内存中，我们可以使用异步内存拷贝来将数据传输与内核执行重叠
      * 用于异构编程中的数据无关系的场景，CPU发送指令后，不用等待GPU运行，直接运行
    * warp-shuffle，用于在同一个warp内的线程之间交换数据
      * 与本目的concat没有关系

