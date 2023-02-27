---
layout: article
title: channel shuffle 算子
key: 100030
tags: C++ 算子库 CUDA GPU算法 算法
category: blog
date: 2023-02-27 00:00:00 +08:00
mermaid: true
---

# 基础

## 快速整数除法

[快速整数除法链接](https://amosteernamazz.github.io/blog/2023/02/26/arithmeticNN.html#%E5%BF%AB%E9%80%9F%E6%95%B4%E6%95%B0%E9%99%A4%E6%B3%95)

## array数据结构

[array数据结构链接](https://amosteernamazz.github.io/blog/2023/02/26/arithmeticNN.html#array%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84)
# channel shuffle 算子

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

  * 使用Grid-stride loop解决大数据量的并发性能要求
    * 当为七次array的时候每维达到54左右，数据量可达到10000亿
    * 需要使用Grid-stride loop方法


  ```c++
  template <typename T1, typename T2>
  __launch_bounds__(256)
  __global__ void ppl_cukernel_concat_two_inputs(
    int64_t num_elems;
    const T1* input0;
    const T1* input1;
    T2* output;){
      // 使用Grid-stride loop

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


**nhwc的concat操作**
  * `nhwc_axis`指示拼接操作应在哪个轴上进行的参数，即输入张量的哪个维度需要进行拼接。
  * `axis_offset`指示拼接在指定轴上时需要跳过的元素数。
    * 对于拼接操作，输出张量在拼接轴上的维度大小将等于所有输入张量在该轴上的维度大小之和
    * 表示跳过第一个输入张量中的元素数，以便输出张量中的元素与输入张量中的元素对齐
  * `input_stride_fast`其元素表示输入张量在每个维度上的 stride 值
  * `intput_strides`输入张量在每个维度上的 stride 值
  * `output_strides`输出张量在每个维度上的 stride 值
  * `output_strides`、`intput_strides`和`input_stride_fast`三个参数应该都类似之前算数运算中的数据分布方式，通过不断取余最后到最后一维，得到offset

  ```c++
  template <typename T>
  __global__ void ppl_cukernel_concat_nhwc(
      int64_t num_elems,
      int num_dims,
      int nhwc_axis,
      int axis_offset,
      GArray<DivModFast> input_strides_fast,
      GArray<int64_t> input_padded_strides,
      GArray<int64_t> output_padded_strides,
      const T* input,
      T* output)
  {
      int index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems)
          return;

      int64_t output_offset = 0, input_offset = 0;
      int idx, remain                         = index;
      for (int it = 0; it < num_dims; ++it) {
          input_strides_fast[it].divmod(remain, idx, remain);
          input_offset += idx * input_padded_strides[it];
          idx = (it == nhwc_axis) ? idx + axis_offset : idx;
          output_offset += idx * output_padded_strides[it];
      }
      output[output_offset] = input[input_offset];
  }
  ```

**注意**

 * 使用时，`input_strides`和`output_strides`在主机上分配并填充
   * 因为stride信息在声明空间的时候已经确定，其值不会随着GPU线程变化而变化，没有必要再每个线程中都分配一次

**nhwc两个进行concat**

  ```c++
  template <typename T1, typename T2>
  __launch_bounds__(256)
  __global__ void ppl_cukernel_concat_nhwc_two_inputs(
    int64_t num_elems,
    int inner_dims,
    int axis_width0,
    int axis_width1,
    const T1* input0,
    const T1* input1,
    T2* output){
      for(int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += (int64_t)blockDim.x * gridDim.x){
        int outer_index = i / inner_dims;
        int inner_index = i % inner_dims;
        if(inner_index > axis_width0){
          int index = outer_index * axis_width1 + (inner_index - axis_width0);
          output[i] = input0[index];
        }
        else{
          int index = outer_index * axis_width0 + inner_index;
          output[i] = input1[index];
        }
      }
    }
  ```


**带padding的最后维度的concat**
  * concat是需要对input进行padding
    * 不同输入张量的维度大小不一，可能会导致输出张量的大小超出GPU内存的限制。为了解决这个问题，通常会对输出张量进行Padding，使其大小符合GPU内存的限制，从而能够在GPU上高效地运行。
    * Padding后的输出张量，有一部分无用数据不属于输入张量的有效数据，为输出张量大小符合GPU内存限制，填充这些无用的数据。
      * 这会导致内存浪费，相比于无法在GPU上运行模型，内存浪费的影响相对较小。
      * 同时，在一些情况下，Padding后的输出张量也可以被复用，例如在进行反向传播时，可以将输出张量中的Padding数据设置为0，从而方便计算梯度。


  ```c++
  template<typename T1, typename T2>
  __launch_bounds__(256)
  __global__ void ppl_cukernel_concat_nhwc_two_inputs(
    int64_t num_elems,
    int inner_dims,
    int pad_inner_dims,
    int axis_width0,
    int pad_axis_width0,
    int axis_width1,
    int pad_axis_width1,
    const T1* input0,
    const T1* input1,
    T2* output){
      for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;i < num_elems;i += (int64_t)blockDim.x * gridDim.x){
        int outer_index = i / pad_inner_dims;
        int inner_index = i % pad_inner_dims;
        if(inner_index >= axis_width0){
          int asis_offset = inner_index - axis_width0;
          int input_offset = outer_index * pad_axis_width1 + asis_offset;
          output[i] = asis_offset >= axis_width0 ? 0 : input1[input_offset];
        }else{
          int asis_offset = inner_index - axis_width1;
          int input_offset = outer_index * pad_axis_width0 + asis_offset;
          output[i] = asis_offset >= axis_width1 ? 0 : input0[input_offset];
        }
      }
    }
  ```

**对NHWC格式数据concat**

  ```c++
  template <typename T>
  __global__ void ppl_cukernel_concat_nhwc_nopadding(
      int64_t num_elems,
      const T* inputs,
      int64_t concat_size,
      int64_t top_axis_width,
      DivModFast num_elems_inner_fast,
      int axis_offset,
      T* output)
  {
      for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
          i < num_elems;
          i += (int64_t)blockDim.x * gridDim.x) {
          int outer_idx, inner_idx;
          num_elems_inner_fast.divmod(i, outer_idx, inner_idx);
          int64_t top_idx = inner_idx + (outer_idx * top_axis_width + axis_offset);
          output[top_idx] = inputs[i];
      }
  }
  ```