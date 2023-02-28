---
layout: article
title: GPU锁实现
key: 100030
tags: 操作系统 CUDA 线程
category: blog
date: 2023-02-28 00:00:00 +08:00
mermaid: true
---

# 锁机制


## CUDA锁函数`atomicCAS()`

**接受参数**
 * int / unsinged int
 * unsigned long long int
 * unsigned short int

## 锁机制模板类

```c++
template <typename T>
__device__ void atomic_max(T* addr, T val){
  atomicMax(addr, val);
}

template <typename T>
__device__ void atomic_add(T *addr, T val)
{
    atomicAdd(addr, val);
}
template <typename T>
__device__ void atomic_min(T *addr, T val)
{
    atomicMin(addr, val);
}
```

## 锁机制实现

```c++
__device__ __inline__ void atomic_max(int8_t* addr, int8_t val){
  if(*addr >= val)
    return;

  unsigned int* const addr_as_ull = (unsigned int*) addr;
  unsigned int old = *addr_as_ull;
  unsigned int assumed;
  do{
      assumed = old;
      if(reinterpret_cast<int8_t&>(assumed) >= val)
        break;
      old = atomicCAS(addr_as_ull, assumed, val);
  } while (assumed != old);
}

__device__ __inline__ void atomic_add(int8_t* addr, int8_t val){
  if(*addr >=val){
    return;
  }
  unsigned int* const addr_as_ull = (unsigned int*) addr;
  unsigned int old = *addr_as_ull;
  unsigned int assumed;
  do{
    assumed = old;
    old = atomicCAS(addr_as_ull, assumed, reinterpret_cast<int8_t&> (old) + val);
  }while(asuumed != old)
}


```