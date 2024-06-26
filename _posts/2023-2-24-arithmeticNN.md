---
layout: article
title: concat 算子
key: 100030
tags: 算子 CUDA GPU算法 算法
category: blog
date: 2023-02-26 00:00:00 +08:00
mermaid: true
---

# 基础

## 快速整数除法

**步骤**

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e4a1ed43282a40fca58df70aa2d4947b~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)

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

<!--more-->

## array数据结构


**目的**
  * 为NCHW结构中的涉及到concat函数提供GPU数据结构支持

**array的实现**
  * array数据主要是提供GPU端的array实现

  ```c++
  #ifndef PPLCUDA_KERNEL_INCLUDE_MEMORY_UTILS_H_
  #define PPLCUDA_KERNEL_INCLUDE_MEMORY_UTILS_H_
  #define MAX_DIMENSION 7
  #include <vector>
  #include <stdint.h>
  #include <assert.h>

  template <typename T, int32_t capacity = MAX_DIMENSION>
  struct GArray{
    // constructor
    Garray()
    :size_(0)
    ,data_(){
    }

    // constructor
    Garray(int32_t size)
    :size_(size)
    ,data_(){
      assert(size >= 0 && size<= capacity);
    }

    // constructor
    Garray(const std::vector<T>& vec)
    :size_(static_cast<int32_t>vec.size())
    {
      #if !defined(__GNUC__) || __GNUC__ >=5
        static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
      #endif
      memcpy(data_, vec.data(), vec.size()*sizeof(T));
    }


    void SetSize(int32_t size)
    {
      assert(size >= 0 && size <= capacity);
      size_ = size;
    }

    __host__ __device__ int32_t Size() const{
      return size_;
    }

    __host__ __device__ T& operator[] (int32_t index){
      return data_[index];
    }
    
    __host__ __device__ __forceinline__ const T& operator[](int32_t index) const
    {
      return data_[index];
    }

    __host__ __device__ T* Data()
    {
      return data_;
    }

    __host__ __device__ const T* Data() const
    {
      return data_;
    }

    static constexpr int32_t Capacity()
    {
      return capacity;
    };


  private:
    int32_t size_;
    T data_[capacity];
  }

  #endif
  ```


# memory 算子

## concat算子

**目的**
 * 用于在GPU上对输入数据进行**并行拼接**

**concat算子中为什么没有scale变换**

 * Concatenate（concat）操作通常用于将多个张量（tensor）在某个维度上拼接起来。例如，将两个形状为（batch_size，height，width，channels_1）和（batch_size，height，width，channels_2）的张量在第4个维度上拼接起来，得到形状为（batch_size，height，width，channels_1+channels_2）的张量
 * Scale变换通常用于缩放输入数据，以便更好地适应激活函数的范围。在Concatenate操作中，由于拼接的张量具有不同的尺寸和特征数量，缩放可能会引入不必要的复杂性和计算量，并且可能会导致梯度消失或爆炸的问题。
 * 通常使用批量归一化（batch normalization）等技术来控制模型中的尺度问题。这些技术可以通过缩放和平移来标准化每个特征维度，从而避免尺度不一致带来的问题。
 * 在大多数情况下，对于Concatenate操作来说，不需要进行Scale变换。如果需要进行缩放，通常可以在**拼接操作之前**使用其他的方法来调整张量的尺度，例如全局平均池化（global average pooling）或卷积操作等。

### 实现

**广义concat函数**

 * 广义的concat函数，可以指定对不同轴进行concat
   * `concat_size`表示拼接的 tensor 数量
   * top_axis_width：拼接的轴的宽度
   * axis_offset：拼接轴在输入 tensor 中的偏移量

 c++实现

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