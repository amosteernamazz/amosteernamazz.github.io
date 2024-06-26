---
layout: article
title: C++ 预编译关键字
key: 100001
tags: C++ 预编译 关键字
category: blog
date: 2024-04-23 17:47:33 +08:00
mermaid: true
---

# C++ 预编译

本部分主要介绍两部分：**C++文件预编译处理策略**与**常用预编译命令**。在C++预编译文件处理策略中，将介绍**文件处理的策略**和**对C++预编译的循环依赖产生问题的解决方案**。在常用预编译命令中，将介绍**常用的预编译命令**。

<!--more-->
  | field | 关键词 |
  |---|---|
  | C++预编译文件处理策略与循环依赖注意事项 | 预编译顺序 |
  | C++预编译文件处理策略与循环依赖注意事项 | 循环依赖注意事项 |
  | 常用预编译命令 | #include<> 和 #include "" |
  | 常用预编译命令 | #define #undef |
  | 常用预编译命令 | #ifdef, #ifndef, #if, #elif, #else, #endif |
  | 常用预编译命令 | #pragma once |
  | 常用预编译命令 | #error |
  | 常用预编译命令 | #defined |
  | 常用预编译命令 | #warning |

## C++预编译文件处理策略与循环依赖注意事项

### C++预编译文件处理策略

 * 文件内：按照行顺序自上而下
 * 文件间：这主要涉及到头文件（.h 或 .hpp）和源文件（.cpp）的包含关系。一般现代的集成开发环境（IDE）和构建系统（如Make、CMake、GNU Autotools等）会自动为你处理这些步骤。
   * 首先处理所有被引用的头文件（递归处理）。
   * 然后处理源文件，用头文件内容替换#include指令


### 循环依赖注意事项

 * 应该确保头文件是自包含的，并且尽量避免循环依赖

  ```c++
  // my_class.h
  #ifndef MY_CLASS_H
  #define MY_CLASS_H

  // 如果MyClass类依赖于其他类型或函数，确保它们在这里被定义或声明
  // 例如，如果MyClass有一个成员变量是另一个类型AnotherType，
  // 那么AnotherType应该在这个头文件中被定义或至少被前向声明
  class AnotherType; // 前向声明：只是告诉编译器某个名称（如变量、函数或类型）的存在，但并未提供其完整的实现或内存布局。

  class MyClass {
  public:
      MyClass();
      ~MyClass();
      void doSomething();

  private:
      AnotherType* myMember; // 使用前向声明的类型
  };

  // 如果MyClass的成员函数实现在这个头文件中，它们也应该被包含
  inline MyClass::MyClass() {
      // 初始化代码
  }

  inline MyClass::~MyClass() {
      // 析构代码
  }

  inline void MyClass::doSomething() {
      // 功能实现
  }

  #endif // MY_CLASS_H
  ```

## 常用预编译命令

### #include<> 和 #include ""

  * <>特点：标准库文件所在目录，编译器设置的include路径内（Linux：/usr/include，Windows：C:\Program Files\Microsoft Visual Studio\2022\VC\Tools\MSVC\<version>\include\）
  * ""特点：先从当前源文件所在目录找，找不到再去标准库目录找，因此可以设置与标准库同名的函数

### #define #undef

 #define：定义一个宏。宏可以是无参数的（称为对象式宏）或带参数的（称为函数式宏）

 ```c++
 #define PI 3.14159  
 #define MAX(a, b) ((a) > (b) ? (a) : (b))
 #undef PI
 ```

### #ifdef, #ifndef, #if, #elif, #else, #endif
 #ifdef, #ifndef, #if, #elif, #else, #endif：这些指令用于条件编译。它们允许你根据某些条件包含或排除代码块。

 ```c++
 #ifdef DEBUG  
    // 调试代码  
 #endif  
  
 #if SOME_CONDITION  
    // 一些条件代码  
 #elif SOME_OTHER_CONDITION  
    // 另一个条件代码  
 #else  
    // 默认代码  
 #endif
 ```

### #pragma once 

  确保包含#progma once的文件只会被包含一次

  并不是c++标准的一部分。因此，如果需要在所有编译器上都能正确编译，需要确保可移植性，那么使用传统的 #ifndef/#define/#endif 方法更稳妥

```c++
// my_class.h

#pragma once

class MyClass {
public:
    MyClass();
    ~MyClass();
    void doSomething();
};

// main.cpp

#include "my_class.h" // 无论包含my_class.h多少次，只会包含一次

int main() {
    MyClass obj;
    obj.doSomething();
    return 0;
}
```


### #error

生成一个编译时错误，通常用于在编译时检查某些条件是否满足。

```c++
#if !defined(SOME_FEATURE)
#error "SOME_FEATURE is not defined!"
#endif
```

### #defined

#defined是一个预处理运算符，它用于检查一个宏是否已经被定义。这在条件编译中非常有用，允许你根据宏的定义状态来包含或排除代码块。

```c++

#if defined(MY_MACRO)
    // 代码块仅当MY_MACRO被定义时执行
#endif
```


### #warning

#warning 这个预处理指令用于在编译时生成警告消息。它通常用于提醒开发者注意某些潜在的问题或即将发生的变化。

```c++
#warning "This code is deprecated and will be removed in a future version."
```
