---
layout: article
title: 算子库
key: 100030
tags: C++ 算子库 CUDA GPU算法 算法
category: blog
date: 2023-02-24 00:00:00 +08:00
mermaid: true
---

***为什么需要定义一种类型为half8_***

# 基础

 **NHWC和NCHW**
  * 它们在表示图像数据时的存储方式不同。
    * NHWC表示图像的顺序为：（batch_size, height, width, channels），即批次大小、图像高度、图像宽度和通道数的顺序
    * NCHW表示图像的顺序为：（batch_size, channels, height, width），即批次大小、通道数、图像高度和图像宽度的顺序
  * NCHW通常更适合GPU的并行计算方式
    * 更好利用GPU的内存布局和数据传输特性
  * NHWC通常更适合CPU的计算方式
    * 符合CPU的内存层次结构和缓存策略

 **什么是带stride的算子方法**
  * 在计算机视觉和深度学习中，带有stride的算子方法通常用于处理图像或其他类型的多维数据
  * Stride是指在多维数组中，沿着每个轴或维度的间隔（或跳过）的元素数
    * 带有stride的算子方法会根据输入和输出的内存布局以及stride参数，计算每个元素的位置，以便在处理多维数据时能够正确地访问每个元素
    * 例如，如果一个算子需要处理一个四维张量（例如图像），则它需要知道每个维度的步长，以便能够正确地遍历整个张量
    * 使用带有stride的算子方法可以大大简化计算多维数据的代码，并且可以加速算法的执行速度

 **带stride的算子方法为什么能够加速**
  * 通过对输入和输出数据的内存地址进行计算，从而在一个线程块中对多个数据进行计算。
  * 带stride的算子方法可以将输入数据中每个数据的访问间隔设置为固定值，从而能够更好地利用CPU或GPU的cache机制，提高数据的缓存命中率，进一步提高计算效率。


# 算子库方法

 算子库包括算数算子（算数运算、逻辑运算、关系运算）、reduce算子（argmax、argmin等）、format算子（类型转换算子）、unary算子（cast、clip、relu、neg、not、zero等）、nn算子（层归一化、conv、lstm、one-hot、pooling-max、topk、gemm等）、数据大小的转变算子（expand、gather、pad、split、tile、reshape等）

## 算数算子

  首先确定算数算子方法使用template方法，其中包括待确定的**算数类型**和算数元素的**数据类型**
  `template<ArithmeticOpType op_type, typename T>`

### 算数算子宏与基本方法

#### 算数算子宏
 * 算数方法枚举类

  ```c++
  enum ArithmeticOpType {
      Arithmetic_Unknown = 0,
      Arithmetic_Add,
      Arithmetic_Sub,
      Arithmetic_Mul,
      Arithmetic_Div,
      Arithmetic_Max,
      Arithmetic_Min,
      Arithmetic_Pow,
      Arithmetic_PRelu, // similar to arithmetic
      Arithmetic_Mod,
      Arithmetic_OpNum,
      Arithmetic_ForceWord = INT_MAX,
  };
  ```
 * 新建数据类型`half8_`

  ```c++
  struct half8_ {
      half x0;
      half y0;
      half z0;
      half w0;
      half x1;
      half y1;
      half z1;
      half w1;
  };
  ```

#### 维度结构宏

  ```c++
  #define MAXDIMENSIONS 7

  struct ArithmeticParam {
      uint32_t stride_in0[MAXDIMENSIONS];
      uint32_t stride_in1[MAXDIMENSIONS];
      uint32_t stride_out[MAXDIMENSIONS];
  };

  ```

#### 标量算数方法

**算数方法template**

  ```c++
  template<ArithmeticOpType op_type, typename T>
  __device__ inline T ppl_arithmetic_scalar(T a, T b);
  ```

根据不同**数据类型**，不同**算数方法**，进行实现

**标量算数方法实现注意点**
 * 对于int8类型变量（有上下范围时候的差异），为了区分，定义新template函数`ppl_arithmetic_scalar_int8`
 * 对于CUDA内置half类型，重写函数`ppl_arithmetic_scalar`
 * 定义自定义类型half8_的实现

### 算数算子static方法

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

### 算数算子方法
 **在算子方法中，针对不同的数据结构，定义了不同的算子方法，包括双向量算子(无数据结构算子方法、带维度结构的算子方法、图像算子)、单向量算子（元素算子、某维度算子、带广播的算子）**

#### 无数据结构的算子方法

 **ppl_cukernel_arithmetic_nobroadcast**
  ```c++
  template<ArithmeticOpType op_type, typename T>
  __global__ void ppl_cukernel_arithmetic_nobroadcast(
      const uint64_t num_elems,
      const T *input0,
      const T* input1,
      T *output) {
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems) return;
      output[index] = ppl_arithmetic_scalar<op_type, T>(input0[index], input1[index]);
  }
  ```

 **ppl_cukernel_arithmetic_nobroadcast_int8**
  
  ```c++
  template<ArithmeticOpType op_type, typename T>
  __global__ void ppl_cukernel_arithmetic_nobroadcast_int8(
      const uint64_t num_elems,
      const T *input0,
      const T* input1,
      T *output,
      float in_scale0 = 0,
      float in_scale1 = 0,
      float out_scale = 0) {
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems) return;
      output[index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[index], input1[index], in_scale0, in_scale1, out_scale);
  }
  ```

#### 带维度结构算子（stride方法）

 **ppl_cukernel_arithmetic_fp16算子**
  * 实现两个输入张量的逐元素算术运算，输出结果保存在输出张量中
  * 参数定义
    * num_elems：输出张量中元素的总数
    * dim_count：输入张量和输出张量的维度数
    * param：一个包含输入张量和输出张量的尺寸、步长等信息的结构体
    * input0：第一个输入张量的指针
    * input1：第二个输入张量的指针
    * output：输出张量的指针
  * 实现流程
    * 根据线程索引计算出当前线程需要处理的元素索引
    * 利用输入张量的步长信息，计算出当前元素在输入张量中的索引
    * 把当前元素所在位置的两个输入张量元素加载到共享内存中
    * 对共享内存中的两个元素执行逐元素算术运算，并将结果保存在共享内存中
    * 把共享内存中的运算结果写回到输出张量中

  ```c++

  template <ArithmeticOpType op_type, typename T1, typename T2>
  __global__ void ppl_cukernel_arithmetic_fp16(
      const uint64_t num_elems,
      const int dim_count,
      ArithmeticParam param,
      const T1 *input0,
      const T1 *input1,
      T1 *output)
  {
      // index tid out_index
      // 根据线程索引计算出当前线程需要处理的元素索引
  #if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems)
          return;
      int tid = threadIdx.x;
      __shared__ T2 transm[512];
      T1 *transm_half      = reinterpret_cast<T1 *>(transm);
      const T2 *input0_ptr = reinterpret_cast<const T2 *>(input0);
      const T2 *input1_ptr = reinterpret_cast<const T2 *>(input1);
      T2 *output_ptr       = reinterpret_cast<T2 *>(output);

      uint64_t out_index = index;
      uint64_t offset0   = 0;
      uint64_t offset1   = 0;
      for (int i = 0; i < dim_count; i++) {
          // 根据输入张量的步长信息，计算当前元素在输入张量中的索引
          uint64_t dim_off = index / param.stride_out[i];
          offset0 += dim_off * param.stride_in0[i];
          offset1 += dim_off * param.stride_in1[i];
          index = index % param.stride_out[i];
      }
      // 将当前元素所在位置的元素输入到共享内存中
      transm[tid + 0]   = input0_ptr[offset0];
      transm[tid + 256] = input1_ptr[offset1];
      // 对共享内存中的两个元素执行逐元素算术运算，并将结果保存在共享内存中
      // 解决了shared memory bank conflict问题
      transm_half[tid] = ppl_arithmetic_vector_fp16<op_type>(transm_half[tid + 0], transm_half[tid + 256]);
      // 把共享内存中的运算结果写回到输出张量中
      output_ptr[out_index] = transm[tid];
  #endif
  }
  ```

 **ppl_cukernel_arithmetic**
  * 执行了对两个输入数组进行逐元素算术操作的操作
    * 通过索引计算输出元素的索引，然后计算输入数组的偏移量
    * 在循环中遍历输入数组的每个维度，计算偏移量的公式是将索引除以输出数据在该维度的步幅
    * 将其乘以相应的输入数据步幅
    * 将输入数组中的值进行算术操作，将结果存储到输出数组中


  ```c++
  template<ArithmeticOpType op_type, typename T>
  __global__ void ppl_cukernel_arithmetic(
      const uint64_t num_elems,
      const int dim_count, 
      ArithmeticParam param,
      const T *input0,
      const T* input1,
      T *output) {
      // 通过索引计算输出元素的索引，然后计算输入数组的偏移量
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems) return;

      uint64_t out_index = index;
      uint64_t offset0 = 0;
      uint64_t offset1 = 0;
      // 在循环中遍历输入数组的每个维度，计算偏移量的公式是将索引除以输出数据在该维度的步幅
      for (int i = 0; i < dim_count; i++) {
          uint64_t dim_off = index / param.stride_out[i];
          // 将其乘以相应的输入数据步幅
          offset0 += dim_off * param.stride_in0[i];
          offset1 += dim_off * param.stride_in1[i];
          index = index % param.stride_out[i]; 
      }
      // 将输入数组中的值进行算术操作，将结果存储到输出数组中
      output[out_index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
  }
  ```

 **ppl_cukernel_arithmetic_int8**

  * 不同与第一种fp16的算子方法，本方法是带张量缩放因子的int8方法

  ```c++

  template<ArithmeticOpType op_type, typename T>
  __global__ void ppl_cukernel_arithmetic_int8(
      const uint64_t num_elems,
      const int dim_count, 
      ArithmeticParam param,
      const T *input0,
      const T* input1,
      T *output,
      float in_scale0 = 0,
      float in_scale1 = 0,
      float out_scale = 0) {
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems) return;

      uint64_t out_index = index;
      uint64_t offset0 = 0;
      uint64_t offset1 = 0;
      for (int i = 0; i < dim_count; i++) {
          uint64_t dim_off = index / param.stride_out[i];
          offset0 += dim_off * param.stride_in0[i];
          offset1 += dim_off * param.stride_in1[i];
          index = index % param.stride_out[i]; 
      }
      
      output[out_index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[offset0], input1[offset1], in_scale0, in_scale1, out_scale);
  }

  ```

#### 图像算子

 **ppl_cukernel_arithmetic_limit_nhwc**
  * 与ppl_cukernel_arithmetic类似，不同的是对output进行重排

  ```c++
  template<ArithmeticOpType op_type, typename T>
  __global__ void ppl_cukernel_arithmetic_limit_nhwc(
      const uint64_t num_elems,
      const int dim_count, 
      ArithmeticParam param_ndarray,
      ArithmeticParam param_nhwc,
      const T *input0,
      const T* input1,
      T *output) {
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems) return;

      uint64_t offset0 = 0;
      uint64_t offset1 = 0;
      uint64_t out_offset = 0;
      for (int i = 0; i < dim_count; i++) {
          uint64_t dim_off = index / param_ndarray.stride_out[i];
          offset0 += dim_off * param_nhwc.stride_in0[i];
          offset1 += dim_off * param_nhwc.stride_in1[i];
          out_offset += dim_off * param_nhwc.stride_out[i];
          index = index % param_ndarray.stride_out[i]; 
      }
      
      output[out_offset] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
  }
  ```

 **ppl_cukernel_arithmetic_limit_nhwc_int8**
  * 不同于前面的方法，int8方法引入了张量缩放因子

  ```c++

  template<ArithmeticOpType op_type, typename T>
  __global__ void ppl_cukernel_arithmetic_limit_nhwc_int8(
      const uint64_t num_elems,
      const int dim_count, 
      ArithmeticParam param_ndarray,
      ArithmeticParam param_nhwc,
      const T *input0,
      const T* input1,
      T *output,
      float in_scale0,
      float in_scale1,
      float out_scale) {
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems) return;

      uint64_t offset0 = 0;
      uint64_t offset1 = 0;
      uint64_t out_offset = 0;
      for (int i = 0; i < dim_count; i++) {
          uint64_t dim_off = index / param_ndarray.stride_out[i];
          offset0 += dim_off * param_nhwc.stride_in0[i];
          offset1 += dim_off * param_nhwc.stride_in1[i];
          out_offset += dim_off * param_nhwc.stride_out[i];
          index = index % param_ndarray.stride_out[i]; 
      }
      // 引入张量缩放因子
      // (input0[offset0] * in_scale0 +input1[offset1] * in_scale1) / out_scale;
      output[out_offset] = ppl_arithmetic_scalar_int8<op_type, T>(input0[offset0], input1[offset1],
              in_scale0, in_scale1, out_scale);
  }
  ```

#### 单向量算子

 包括标量与向量的算数算子方法、多维标量数据算数方法（NCHW结构中的C不同）、带广播机制的算子方法

 **ppl_cukernel_arithmetic_one_scalar**

  ```c++

  template<ArithmeticOpType op_type, typename T>
  __global__ void ppl_cukernel_arithmetic_one_scalar(
      const uint64_t num_elems,
      const bool first_shorter, 
      const T *input0,
      const T* input1,
      T *output) {
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems) return;
      int calc_index = 0;
      uint64_t offset0 = first_shorter ? calc_index : index;
      uint64_t offset1 = first_shorter ? index : calc_index;
      output[index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
  }
  ```

 **ppl_cukernel_arithmetic_one_scalar_int8**
  * 其中`inner_dim`表示向量的内部算数操作的单位
    * 对(512,3,1024,1024)和(512,4,1024,1024)来说`inner_dim`应设置为1024*1024

  ```c++
  template<ArithmeticOpType op_type, typename T>
  __global__ void ppl_cukernel_arithmetic_one_dimension(
      const uint64_t num_elems,
      const int32_t inner_dim,
      const bool first_shorter, 
      const T *input0,
      const T* input1,
      T *output) {
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems) return;
      int calc_index = index % inner_dim;
      uint64_t offset0 = first_shorter ? calc_index : index;
      uint64_t offset1 = first_shorter ? index : calc_index;
      output[index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
  }
  ```

 **ppl_cukernel_arithmetic_one_not_broadcast**
  主要针对HW两个维度进行非广播算数运算，`axis_lgt`为C维度的大小，`inner_dim`则是H*W
  * 在图像领域，`axis_lgt`（即轴长度）通常对应于图像的通道数量（例如，RGB图像有3个通道，RGBA图像有4个通道等）`inner_dim`为`H*W`
  * 如果在对多个通道进行算术运算的过程中，需要使用该函数，其中一个输入张量在通道数量上与另一个不匹配，可以使用此函数进行处理。

  ```c++
  template<ArithmeticOpType op_type, typename T>
  __global__ void ppl_cukernel_arithmetic_one_not_broadcast(
      const uint64_t num_elems,
      const int32_t axis_lgt,
      const int32_t inner_dim,
      const bool first_shorter, 
      const T *input0,
      const T* input1,
      T *output) {
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems) return;
      int calc_index = (index / inner_dim) % axis_lgt;
      uint64_t offset0 = first_shorter ? calc_index : index;
      uint64_t offset1 = first_shorter ? index : calc_index;
      output[index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
  }

  ```

 **ppl_cukernel_arithmetic_one_dimension**
  * 一般用于元素级别的运算，在图像中为常见的像素级算数运算

  ```c++
  template<ArithmeticOpType op_type, typename T>
  __global__ void ppl_cukernel_arithmetic_one_dimension(
      const uint64_t num_elems,
      const int32_t inner_dim,
      const bool first_shorter, 
      const T *input0,
      const T* input1,
      T *output) {
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems) return;
      int calc_index = index % inner_dim;
      uint64_t offset0 = first_shorter ? calc_index : index;
      uint64_t offset1 = first_shorter ? index : calc_index;
      output[index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
  }
  ```

 **ppl_cukernel_arithmetic_one_dimension_int8**

  ```c++

  template<ArithmeticOpType op_type, typename T>
  __global__ void ppl_cukernel_arithmetic_one_dimension_int8(
      const uint64_t num_elems,
      const int32_t inner_dim,
      const bool first_shorter, 
      const T *input0,
      const T* input1,
      T *output,
      float in_scale0 = 0,
      float in_scale1 = 0,
      float out_scale = 0) {
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems) return;
      int calc_index = index % inner_dim;
      uint64_t offset0 = first_shorter ? calc_index : index;
      uint64_t offset1 = first_shorter ? index : calc_index;
      output[index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[offset0], input1[offset1], in_scale0, in_scale1, out_scale);
  }
  ```


 **ppl_cukernel_arithmetic_one_broadcast**
  * 假设有两个形状分别为 (N, C, H, W) 和 (1, C, 1, 1) 的输入张量 input0 和 input1，我们要对它们进行广播操作并进行 element-wise 的二元运算。其中 N, C, H, W 分别表示 batch size, channel 数量，高度和宽度。
  * 在这种情况下，我们将 `outer_stride` 设为 CHW，也就是 outermost dimension 的大小，因为在这个 dimension(N) 上进行广播。而 inner_dim 则是 1，因为 input1 在这个 dimension(N) 上只有一个元素。

  ```c++
  template<ArithmeticOpType op_type, typename T>
  __global__ void ppl_cukernel_arithmetic_one_broadcast(
      const uint64_t num_elems,
      const int outer_stride, 
      const int inner_dim, 
      const bool first_shorter, 
      const T *input0,
      const T* input1,
      T *output) {
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems) return;
      int inner_idx = index % inner_dim;
      int outer_idx = index / outer_stride;
      uint64_t calc_index = outer_idx * inner_dim + inner_idx;
      uint64_t offset0 = first_shorter ? calc_index : index;
      uint64_t offset1 = first_shorter ? index : calc_index;
      output[index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
  }
  ```


 **ppl_cukernel_arithmetic_one_broadcast_int8**

  ```c++
  template<ArithmeticOpType op_type, typename T>
  __global__ void ppl_cukernel_arithmetic_one_broadcast_int8(
      const uint64_t num_elems,
      const int outer_stride, 
      const int inner_dim, 
      const bool first_shorter, 
      const T *input0,
      const T* input1,
      T *output,
      float in_scale0 = 0,
      float in_scale1 = 0,
      float out_scale = 0) {
      uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      if (index >= num_elems) return;
      int inner_idx = index % inner_dim;
      int outer_idx = index / outer_stride;
      uint64_t calc_index = outer_idx * inner_dim + inner_idx;
      uint64_t offset0 = first_shorter ? calc_index : index;
      uint64_t offset1 = first_shorter ? index : calc_index;
      output[index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[offset0], input1[offset1], in_scale0, in_scale1, out_scale);
  }
  ```

## 逻辑算子

 首先明确逻辑算子的**逻辑运算**方法和**数据类型**，数据类型为bool类型
 `template <LogicalOpType op_type>`

### 逻辑算子宏与基本方法
 

#### 逻辑算子宏

**逻辑运算方法enum**
  ```c++

  enum LogicalOpType {
      Logical_Unknown = 0,
      Logical_And,
      Logical_Xor,
      Logical_OpNum,
      Logical_ForceWord = INT_MAX,
  };
  ```

**新建数据结构bool8_**
  ```c++
  struct bool8_ {
      bool x0;
      bool y0;
      bool z0;
      bool w0;
      bool x1;
      bool y1;
      bool z1;
      bool w1;
  };
  ```

#### 维度结构宏

  ```c++
  #define MAXDIMENSIONS 7

  struct LogicalParam {
      uint32_t stride_in0[MAXDIMENSIONS];
      uint32_t stride_in1[MAXDIMENSIONS];
      uint32_t stride_out[MAXDIMENSIONS];
  };

  ```



#### 标量逻辑方法

**template结构**
  ```c++
  template<LogicalOpType op_type>
  __device__ inline bool ppl_logical_scalar(bool a, bool b);
  ```

**标量方法实现**

根据**标量逻辑方法**的不同，对bool类型标量方法进行实现


  ```c++
  template<>
  __device__ inline bool ppl_logical_scalar<Logical_And>(bool a, bool b){
    return a && b;
  }
  ```
提供上层**不同数据类型**的**static方法**

  ```c++
  template<LogicalOpType op_type>
  static __device__ inline bool ppl_logical_vector(bool a, bool b){
    bool ans;
    ans = ppl_logical_scalar<op_type>(a, b);
    return ans;
  }

  // 还有bool8_数据结构的实现

  ```


#### 逻辑算子方法

 逻辑算子方法根据数据结构的不同有两种方法

**没有数据结构的逻辑算子方法**
  ```c++
  template <LogicalOpType op_type, typename T1, typename T2>
  __global__ void ppl_cukernel_logical_naive(
    const uint64_t num_elems,
    const T1* input0,
    const T1* input1,
    const T2* output){
      #if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
        uint64_t index = blockId.x * blockDim.x + threadId.x;
        if(index >= num_elems){
          return;
        }
        output[index] = ppl_logical_vector<op_type>(input0[index], input1[index]);
      #endif
    }
  ```
**带数据结构的逻辑算子方法**
 * 使用`stride_in`和`stride_out`可能会有不同，主要是为了输出的时候进行**shape的更改**
  ```c++
  template <LogicalOpType op_type, typename T1, typename T2>
  __global__ void ppl_cukernel_logical(
    uint64_t num_elems,
    int dim_count,
    LogicalParam param,
    const T1* input0,
    const T1* input1,
    const T2* output){
      #if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR >= 9
        uint64_t index = blockId.x * blockDim.x + threadId.x;
        if(index >num_elems){
          return;
        }
        uint64_t out_index = index;
        uint64_t offset0 = 0;
        uint64_t offset1 = 0;
        for(int i = 0; i <dim_count; i++){
            uint64_t dim_off = index / param.stride_out[i];
            offset0 += dim_off * param.stride_in0[i];
            offset1 += dim_off * param.stride_in1[i];
            index = index % param.stride_out[i];
        }
        output[out_index] = ppl_logical_vector<op_type>(input0[offset0], input1[offset1]);
      #endif
    }
  ```







## 比较算子


 首先明确比较算子的**逻辑运算**方法和**数据类型**
 `template <RelationOpType op_type, typename T>`

### 比较算子宏与基本方法
 

#### 比较算子宏

**比较运算方法enum**
  ```c++

  enum RelationOpType {
    Relation_Unknown = 0,
    Relation_Equal,
    Relation_Greater,
    Relation_Greater_Or_Equal,
    Relation_Less,
    Relation_OpNum,
    Relation_ForceWord = INT_MAX,
  };

  ```

**新建数据结构bool8_和 half8_**
  ```c++
  struct half8_ {
    half x0;
    half y0;
    half z0;
    half w0;
    half x1;
    half y1;
    half z1;
    half w1;
  };

  struct bool8_ {
    bool x0;
    bool y0;
    bool z0;
    bool w0;
    bool x1;
    bool y1;
    bool z1;
    bool w1;
  };
  ```

#### 维度结构宏

  ```c++
  #define MAXDIMENSIONS 7

  struct RelationParam {
      uint32_t stride_in0[MAXDIMENSIONS];
      uint32_t stride_in1[MAXDIMENSIONS];
      uint32_t stride_out[MAXDIMENSIONS];
  };


  ```



#### 标量比较方法

**template结构**
  ```c++
  template<LogicalOpType op_type, typename T>
  __device__ inline bool ppl_relation_scalar(T a, T b);
  ```

**标量方法实现**

根据**标量比较方法**的不同和**数据类型**的不同，对标量方法进行**实现**

 * Equal方法使用`fabsf(a-b) <1e-6`
 * 
提供上层**不同数据类型**的**static方法**

  ```c++
  template<LogicalOpType op_type>
  static __device__ inline bool ppl_logical_vector(bool a, bool b){
    bool ans;
    ans = ppl_logical_scalar<op_type>(a, b);
    return ans;
  }

  // 还有bool8_数据结构的实现

  ```


#### 比较算子方法

 比较算子方法根据数据结构的不同有两种方法

**没有数据结构的比较算子方法**
  ```c++
  template <LogicalOpType op_type, typename T1, typename T2>
  __global__ void ppl_cukernel_logical_naive(
    const uint64_t num_elems,
    const T1* input0,
    const T1* input1,
    const T2* output){
      #if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
        uint64_t index = blockId.x * blockDim.x + threadId.x;
        if(index >= num_elems){
          return;
        }
        output[index] = ppl_logical_vector<op_type>(input0[index], input1[index]);
      #endif
    }
  ```
**带数据结构的比较算子方法**
 * 使用`stride_in`和`stride_out`可能会有不同，主要是为了输出的时候进行**shape的更改**
  ```c++
  template <LogicalOpType op_type, typename T1, typename T2>
  __global__ void ppl_cukernel_logical(
    uint64_t num_elems,
    int dim_count,
    LogicalParam param,
    const T1* input0,
    const T1* input1,
    const T2* output){
      #if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR >= 9
        uint64_t index = blockId.x * blockDim.x + threadId.x;
        if(index >num_elems){
          return;
        }
        uint64_t out_index = index;
        uint64_t offset0 = 0;
        uint64_t offset1 = 0;
        for(int i = 0; i <dim_count; i++){
            uint64_t dim_off = index / param.stride_out[i];
            offset0 += dim_off * param.stride_in0[i];
            offset1 += dim_off * param.stride_in1[i];
            index = index % param.stride_out[i];
        }
        output[out_index] = ppl_logical_vector<op_type>(input0[offset0], input1[offset1]);
      #endif
    }
  ```

