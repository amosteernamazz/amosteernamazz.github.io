---
layout: article
title: C++ 链接
key: 100004
tags: C++ 链接
category: blog
date: 2024-04-19 15:20:13 +08:00
mermaid: true
---

### extern 链接
 * extern：说明是在别处定义的，在此处引用，相对于#include方法，加速编译过程
   * 用于支持c或c++函数调用规范，当在c++程序中调用c的库，需要extern c的声明引用，主要因为c++和c编译完成后，目标代码的命名规则不同，用来解决名字匹配

### this指针的赋值