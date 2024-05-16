---
layout: article
title: C++ 流程
key: 100006
tags: C++
category: blog
date: 2019-09-07 00:00:00 +08:00
mermaid: true
---



## C++代码的编译流程

 * 步骤：预处理器→编译器→汇编器→链接器
 * 预处理阶段：在这个阶段，编译器会处理预处理指令（以 '#' 开头），如 #include、#define 等。预处理器会将这些指令替换为相应的内容，生成一个被称为 "translation unit" 的中间文件。
 * 编译阶段：在这个阶段，预处理后的代码会被编译成汇编语言。编译器会将源代码转换成汇编语言的表示形式，这个过程包括了语法和语义分析，以及生成中间表示（如抽象语法树）编译器可能会进行一系列的优化操作，以提高程序的性能和效率。这些优化包括但不限于循环展开、内联函数、常量折叠等。该阶段会构建继承关系。编译工具包括GCC、Clang、MSVC等
 * 汇编阶段：汇编器将汇编代码转换为机器代码，也称为目标代码或对象代码。
目标代码是特定于体系结构的，这意味着为x86架构生成的代码不能直接在ARM架构上运行。
 * 链接阶段：如果程序由多个源文件组成，那么编译后会生成多个目标文件（Object Files）。链接器将这些目标文件以及所需的库文件链接在一起，生成最终的可执行文件。这个过程包括解析符号引用、地址重定向和符号重定向等步骤。链接器可以生成两种类型的输出：可执行文件（可以直接运行）和库文件（包含可以在其他程序中使用的代码和数据）。



## 静态链接和动态链接

### 静态链接
 
 .a
 **定义**

  * 静态链接是将程序的所有模块在编译时链接到一个单独的可执行文件中的过程。
  * 在静态链接中，目标文件中的所有模块和库都会被复制到最终的可执行文件中，使得可执行文件独立于系统环境。

 **过程**
  * 在编译和链接程序时，链接器将目标文件中的所有符号（如函数和变量）解析并合并到一个单独的可执行文件中。
  * 静态链接器将所有依赖项（如库文件）的代码和数据都复制到可执行文件中，因此生成的可执行文件比较大。

 **优点**

  * 独立性：生成的可执行文件可以在没有外部依赖的情况下在任何环境中运行。
  * 性能：静态链接可以避免运行时的库加载和解析过程，因此可以提高程序的启动速度。

 **缺点**
  * 可执行文件较大：每个可执行文件都包含了所有依赖项的代码和数据，因此可能会占用较多的磁盘空间。
  * 更新困难：如果库更新了，所有依赖该库的程序都需要重新编译和链接以使用新版本。

### 动态链接

 .so，动态链接库的地址：LD_LIBRARY_PATH，GCC默认为动态链接
 **定义**

  * 动态链接是在程序运行时将程序的模块和所需的库链接到一起的过程。
  * 在动态链接中，程序运行时所需的库被加载到内存中，而不是在编译时被合并到可执行文件中。

 **过程**

  * 在编译和链接程序时，只生成包含程序自身代码和对外部库的引用的可执行文件。
  * 当程序运行时，操作系统的动态链接器（如Linux中的ld.so）将程序所需的库加载到内存中，并将程序中的引用解析为动态库中的符号。

 **优点**

  * 节省空间：由于可执行文件只包含程序自身的代码和数据，因此通常较小。
  * 管理便捷：多个程序可以共享同一个动态库，节省了系统资源并方便了库的更新和维护。

 **缺点**

  * 运行时开销：程序运行时需要加载和解析动态库，可能会稍微降低启动速度。
  * 系统依赖：程序运行需要确保所依赖的动态库在系统中可用，否则可能会导致运行时错误。




## 加载

  * 在加载阶段，操作系统负责将可执行文件的内容加载到内存中，并为程序分配必要的资源。这个过程包括以下几个关键步骤：
    * 内存分配：操作系统会为程序分配内存空间，这个空间通常包括代码段、数据段、堆和栈等区域。代码段用于存储程序的指令，数据段用于存储静态变量和全局变量，堆用于动态分配内存，栈用于存储函数调用和局部变量。
    * 装载程序：操作系统会将可执行文件的内容从存储设备（如硬盘）中读取到内存中。这包括将程序的指令、全局变量的初始值等加载到相应的内存区域。
    * 地址重定位：在加载阶段，操作系统还会执行地址重定位的操作。由于程序可能会被加载到内存的不同位置，其中的地址引用需要被调整，确保程序能够正确地访问内存中的数据和指令。
    * 动态链接库加载：如果程序依赖于动态链接库（DLL），那么这些库也会在加载阶段被加载到内存中，并与程序进行链接。
  * 一旦加载完成，操作系统会将程序的控制权交给程序的入口点，即 main() 函数


### 加载&运行时的对象构建与析构顺序

**顺序原因**

 * 在编译阶段会产生继承树

**构建顺序**

 * 基类对象构建
 * 基类对象成员按照声明顺序构建，不按照初始化列表
 * 对象构建

**析构顺序**

 * 析构顺序与构建顺序相反

```c++
#include <iostream>

class Member1 {
public:
    Member1() {
        std::cout << "Member1 constructor\n";
    }
    ~Member1() {
        std::cout << "Member1 destructor\n";
    }
};

class Member2 {
public:
    Member2() {
        std::cout << "Member2 constructor\n";
    }
    ~Member2() {
        std::cout << "Member2 destructor\n";
    }
};

class Base1 {
public:
    Base1() : member111(), member222() {
        std::cout << "base1 constructor\n";
    }
    ~Base1() {
        std::cout << "base1 destructor\n";
    }
private:
    Member1 member111;
    Member2 member222;
};

class Base: public Base1 {
public:
    Base() {
        std::cout << "Base constructor\n";
    }
    ~Base() {
        std::cout << "Base destructor\n";
    }
};

class Derived : public Base {
public:
    Derived() : member1(), member2() {
        std::cout << "Derived constructor\n";
    }
    ~Derived() {
        std::cout << "Derived destructor\n";
    }
private:
    Member2 member2;
    Member1 member1;

};
Derived d;
int main() {
    std::cout << "main\n";
    return 0;
}

// Member1 constructor
// Member2 constructor
// base1 constructor
// Base constructor
// Member2 constructor
// Member1 constructor
// Derived constructor
// main
// Derived destructor
// Member1 destructor
// Member2 destructor
// Base destructor
// base1 destructor
// Member2 destructor
// Member1 destructor
```


### C++加载的内存分配模型

 **代码段**

  * 可执行文件中的代码被复制到代码段中
    * 只读，防止程序意外修改自身的代码
    * 共享，多个进程同时执行相同的程序，可以共享同一个代码段，节省内存

 **数据段**

  * 包含程序中的全局变量和静态变量，以及一些常量数据。
  * 在加载过程中，数据段被初始化并分配内存空间，以存储这些变量的值。
  * 数据段通常分为两部分：初始化的数据段（Initialized Data Segment）和未初始化的数据段（Uninitialized Data Segment，也称为BSS段）。
    * 初始化的数据段存储已经初始化的全局变量和静态变量的值
    * 未初始化的数据段则存储全局变量和静态变量的声明，但尚未被初始化的值。程序开始执行时将其初始化为零或空值。

 **堆**

  * 动态内存分配的区域，用于存放程序运行时动态分配的内存。
  * 在加载过程中，操作系统会为堆分配一块初始大小的内存空间，通常称为堆的起始地址。
  * 程序可以通过调用new和delete等动态内存管理函数来在堆上分配和释放内存，堆的大小可以根据程序的需要动态增长或缩小。

 **栈**

  * 栈是用于函数调用和局部变量存储的内存区域。
  * 在加载过程中，操作系统会为每个线程分配一块栈空间，用于存储函数调用时的参数、局部变量和函数调用的返回地址等信息。栈是一种先进后出（LIFO）的数据结构，函数调用时会将调用函数的参数和局部变量压入栈中，函数返回时会从栈中弹出这些数据。


### 加载中的程序装载

 * 在C++中，加载时初始化主要涉及全局变量（数组或对象）和静态变量（数组或对象）。这些变量在程序加载到内存时就会被初始化。
   * 全局变量（数组或对象）在程序的任何部分都可见，它们的初始化在程序启动时完成，在main函数执行之前进行。

```c++
// 变量
int globalVar = 42; // 加载时初始化

// 数组
int arr[4] = {1,2,3,4}; // 加载时初始化

// 对象
class MyClass {
public:
    MyClass() {
        std::cout << "MyClass Constructor" << std::endl;
    }
};

MyClass globalObject; // 加载时初始化，构造函数在main之前调用
```

   * 静态变量（数组或对象）
     * 全局静态变量、命名空间作用域中的静态变量和类的静态成员变量：通常在加载时初始化。这些变量的作用域仅限于声明它们的文件。
     * 静态局部变量：
       * 如果初始化表达式是常量表达式（带const关键字），则在加载时初始化。
       * 否则，在第一次使用时初始化。
     * 静态变量的初始化顺序不确定，特别是针对不同的单元时，会导致使用未定义的静态变量。解决办法是使用函数的本地静态变量，在第一次使用时才进行初始化，同时线程安全

```c++
// 全局静态变量
static int staticGlobalVar = 10; // 加载时初始化

// 命名空间下的全局静态变量
namespace MyNamespace {
    static int staticVar = 10; // 命名空间作用域中的静态变量，在加载时初始化
}

// 类的静态成员变量
class MyClass {
public:
    static int staticMemberVar;
};

int MyClass::staticMemberVar = 300; // 加载时初始化

// 局部静态变量
void exampleFunction() {
    static const int staticLocalVar = 20; // 如果是常量表达式，加载时初始化
    static int staticconst = 10; // 加载时初始化
}

void func() {
    static int localStaticVar = someFunction(); // 第一次调用func时初始化
}

// 静态全局对象
class MyClass {
public:
    MyClass() {
        std::cout << "MyClass Constructor" << std::endl;
    }
};

static MyClass staticGlobalObject; // 加载时初始化，构造函数在main之前调用
```

   * 初始化列表或联合体：使用初始化列表初始化的全局或静态变量在加载时初始化，联合体同理

```c++
// struct
struct Data {
    int x;
    int y;
};

Data data = {1, 2}; // 加载时初始化

// union
union MyUnion {
    int a;
    float b;
};

MyUnion myUnion = { 1 }; // 加载时初始化
```

   * 外部库与自定义初始化
     * 有些外部库会在加载时初始化全局状态。这些初始化通常通过库的构造函数或初始化函数进行。
     * 使用自定义初始化函数：通过自定义初始化函数，程序员可以确保某些初始化在加载时完成。比如使用GCC的构造函数属性

```c++
// 外部库初始化
extern "C" void __attribute__((constructor)) myInitFunction() {
    // 加载时初始化代码
}

// 自定义初始化
void __attribute__((constructor)) initFunction() {
    // 加载时初始化代码
}
```

   * 常量表达式的初始化：
     * 对于局部静态常量来说，在加载时初始化
     * 对于编译时能确定的初始值的常量，会在编译期间进行初始化，是直接插入exe文件中
     * 对于某些需要运算得到的值，初始化在运行时进行



## 执行

在C++程序的执行阶段，程序将按照main()函数开始执行，然后根据程序中的逻辑顺序执行相应的语句和函数调用。
