---
layout: article
title: C++ 编译关键字
key: 100002
tags: C++ 编译 关键字
category: blog
date: 2024-04-27 22:23:34 +08:00
mermaid: true
---

  | 类别 | 问题 | 
  |---|---|
  | ++i与i++ | ++i |
  | ++i与i++ | i++ |
  | NULL和nullptr | NULL定义 |
  | NULL和nullptr | nullptr定义 |
  | inline | inline优缺点 |
  | inline | inline使用建议 |
  | inline | 虚函数是否可以为inline |
  | auto类型的确定 | auto优势 |
  | auto类型的确定 | auto注意事项 |
  | auto类型的确定 | auto可以推断类型 |
  | auto类型的确定 | auto不可以推断类型 |
  | friend | friend友元函数 |
  | friend | friend友元类 |
  | friend | friend注意事项 |
  | explicit | explicit应用 |
  | final | final防止类被继承 |
  | final | final禁止虚函数被重写 |
  | override | override优势 |
  | override | override注意事项 |
  | volatile | volatile定义 |
  | volatile | volatile应用 |
  | enum | C++11之前的enum |
  | enum | C++11及之后的enum |
  | 字节对齐 | 字节对齐原因 |
  | 字节对齐 | 对齐原则、常见字节数 |
  | 字节对齐 | 控制字节对齐 |
  | struct | C++11之前的struct |
  | struct | C++11之后的struct |
  | union | union注意事项 |
  | union | union使用场景 |
  | typedef | union使用场景 |
  | 静态this指针的存在和处理 | this指针的处理逻辑 |
  | 静态this指针的存在和处理 | this用途 |
  | 静态this指针的存在和处理 | this优点 |
  | 静态this指针的存在和处理 | 谨慎使用delete this |
  | 静态this指针的存在和处理 | 什么情况可以使用delete this |
  | 泛化常数constexpr | constexpr注意事项 |
  | decltype | decltype可以应用的类型 |
  | decltype | decltype不可引用的类型 |
  | decltype | decltype与左值右值 |
  | extern定义 | extern处理逻辑 |
  | extern定义 | extern举例 |


### ++i与i++

 **++i**

  ++i 不会产生临时对象

  ```c++
  int& int::operator++ (){
  *this +=1;
  return *this;
  }
  ```

 **i++**
  
  i++ 产生临时对象，会导致效率降低

  ```c++
  const int int::operator(int){
  int oldValue = *this;
  ++(*this);
  return oldValue;
  }

  ```



### NULL和nullptr

 **NULL**
  在c++与c中，使用NULL代表0或void*，但可能会产生类型不匹配的警告或错误

  ```c++
  #ifdef __cplusplus
  #define NULL 0
  #else
  #define NULL ((void*)0)
  #endif
  ```

 **nullptr**
  * 为解决NULL的二义性，使用nullptr替代空指针，对应参数为 void*

<!--more-->

### inline

 * 建议编译器将函数体插入到每个调用点，而不是进行常规的函数调用。用于优化那些体积小、调用频繁的函数，减少函数调用开销。
 * 只是建议给编译器的提示，并不保证一定会被内联。编译器会根据优化策略来决定是否内联一个函数。

 **inline优点**
  * 减少函数调用开销，提高速度：内联函数在编译时将函数体直接插入到每个调用点，从而避免了函数调用的开销，如参数传递、栈帧创建和销毁等。这有助于提高程序的执行效率，特别是在函数体较小且调用频繁的情况下。
  * 类型检查：与宏相比，内联函数在展开时会进行类型检查和语法分析，从而提供了更好的类型安全性。这有助于减少因类型不匹配或语法错误而导致的运行时错误。
  * 内联函数与普通函数一样具有函数名和参数列表，这使得代码更具可读性。同时，它还可以访问类的私有和保护成员，提供了更灵活的使用方式。

 **inline缺点**
  * 代码量增大：由于内联函数的代码会在每个调用点展开，因此可能导致代码体积增大，这可能导致可执行文件的大小增加。对于大型程序或库来说，这可能成为一个问题。
  * 编译增加：由于内联函数的代码需要在每个调用点展开，这可能导致编译时间增加。特别是在大型项目中，这种影响可能更加明显
  * 内联函数不当设计：对于代码较长或包含循环的函数，内联可能导致性能下降，因为函数体展开后的代码可能比函数调用本身更耗时
  * 编译增加：函数频繁修改，需要重新编译所有调用该函数的地方
  * 滥用可能导致优化失败：不当使用内联可能会导致编译器无法进行其他有效的优化。


 **使用建议**
  * 小函数优先：对于只有几行代码的小函数，内联通常是有益的。但对于大函数，内联可能会导致代码膨胀和性能下降。
  * 不要过度使用：不要仅仅因为你想让函数更快就盲目地使用内联。应该基于性能分析和代码审查来决定是否使用内联。
  * 注意编译器优化：现代编译器通常能自动地识别并内联那些适合内联的小函数，所以有时候你不需要显式地使用 inline 关键字。

 **虚函数是否可以为inline**
  * 无多态：虚函数只有在编译器知道所调用的对象是哪个类才可，但将虚函数设置为inline并不会提高执行效率；
  * 有多态：表现多态是不可内联

### auto类型的确定
 * auto 用于自动推断变量的类型。它让编译器根据初始化表达式的类型自动为变量选择正确的类型，主要适用于局部变量和临时变量，这些变量的类型通常可以在编译时根据初始化表达式明确推断出来，但auto变量的赋值需要开辟内存，是在运行期确定的
 
 **auto优势**
  * 代码简洁性：避免显式地写出变量的类型，代码更加简洁易读。
  * 泛型编程：在模板编程或处理容器、迭代器等泛型结构时，auto 可以自动推断出正确的类型，使代码更加灵活和易于编写。
  * 推断发生在编译期，所以使用auto并不会造成程序运行时效率的降低。

 **auto注意事项**
  * 必须初始化：使用 auto 声明的变量在声明时必须立即初始化，因为编译器是根据初始化表达式的类型来推断变量类型的。
  * 不要过度使用：虽然 auto 可以简化代码，但过度使用可能会使代码的可读性降低。在某些情况下，显式地写出变量的类型可能更有助于理解代码。

 **auto可以推断类型**
  * 基本类型，如int、double、char等
  * 引用、指针类型
  * 复合类型：例如数组、结构体、类实例等。
    * 对于数组，auto会推断出数组第一个元素的指针类型，而不是数组本身。
  * 模板类型：在模板编程中，auto可以推断出模板参数的类型。

 **auto不可以推断的类型**
  * 函数参数类型：函数参数的类型必须显式指定，auto不能用于推导函数参数的类型。
  * 非静态成员变量类型：auto是在编译时期进行推导的，因此不能用于推导类的非静态成员变量的类型，但一般静态成员变量也不用auto。
  * 顶层const：auto一般会忽略掉顶层const（即指针本身的常量性），但底层const（即指针指向的对象的常量性）会保留下来。
  * 数组类型：auto不能用于声明数组，因为数组类型不能出现在顶级类型中。
  * 模板参数实例化：auto不能作为模板参数进行实例化。


  ```c++
  // 引用
  int x = 10;
  auto& ref = x;  // 运行时确定
  ref = 20;

  // 指针
  int w = 10;
  auto ptr = &w;  // 运行时确定
  const int ci = 10;
  const auto c2 = ci; // 运行时确定

  // const类型
  const int a = 10;
  auto b = a; // b被推导为int
  const auto c = a; // c被推导为const int

  const int x = 20;  
  const int* ptr1 = &x;  // ptr1 为const int* 类型
  auto ptr2 = ptr1; // ptr2 为const int* 类型
  const auto ptr3 = ptr1; // ptr1 为const int* 类型
  ```


### friend

 * 允许一个函数或类访问另一个类的私有（private）或保护（protected）成员，即使这些成员在通常情况下是不可访问的

 **friend友元函数**
  * 当一个函数被声明为另一个类的友元（friend），那么这个函数就可以访问该类的所有成员，包括私有和保护成员。
 
 ```c++
 class MyClass {
private:
    int secret;

public:
    MyClass(int val) : secret(val) {}

    // 声明 friend 函数
    friend void revealSecret(const MyClass& obj);
};

// friend 函数实现
void revealSecret(const MyClass& obj) {
    std::cout << "The secret is: " << obj.secret << std::endl;
}

int main() {
    MyClass myObj(42);
    revealSecret(myObj);  // 输出: The secret is: 42
    return 0;
}
 ```

**friend友元类**

```c++
class MyClass {
private:
    int secret;

public:
    MyClass(int val) : secret(val) {}

    // 声明另一个类为 friend
    friend class MyFriendClass;
};

class MyFriendClass {
public:
    void revealSecret(const MyClass& obj) {
        std::cout << "The secret is: " << obj.secret << std::endl;
    }  
};  
  
int main() {
    MyClass myObj(42);
    MyFriendClass friendObj;
    friendObj.revealSecret(myObj);  // 输出: The secret is: 42
    return 0;
}
```

 **friend使用注意**
  * 谨慎使用：friend关键字破坏了封装性，使得原本不应该被外部访问的私有成员变得可访问。因此，应该谨慎使用
  * 单向关系：友元关系是单向的。如果类A将类B声明为友元，那么类B可以访问类A的私有成员，但类A不能访问类B的私有成员，除非类B也将类A声明为友元。
  * 非传递性：友元关系不是传递的。如果类A是类B的友元，类B是类C的友元，那么类A不一定是类C的友元。
  * 非继承性：友元关系不会被子类继承。如果类A是类B的友元，而类C是类B的子类，那么类A不一定是类C的友元。





### explicit

 * 主要用于修饰只有一个参数的类构造函数，目的是防止该类对象进行不期望的隐式转换。

 **explicit应用**
  * 当定义了只有一个参数的类构造函数时，构造函数除了构造对象外，还被编译器用于进行隐式类型转换。可能会导致一些不期望的行为或错误。使用 explicit 关键字可以明确地告诉编译器，我们不希望进行这种隐式转换。

```c++
class Foo {  
public:  
    Foo(int value) {  
        // ... 构造函数的实现 ...  
    }  
};  
  
void takeFoo(Foo foo) {  
    // ... 函数实现 ...  
}  
  
int main() {  
    int x = 10;  
    takeFoo(x);  // 这里会隐式调用 Foo(int value) 构造函数  
    return 0;  
}
```

```c++
class Foo {  
public:  
    explicit Foo(int value) {  
        // ... 构造函数的实现 ...  
    }  
};  
  
void takeFoo(Foo foo) {  
    // ... 函数实现 ...  
}  
  
int main() {  
    int x = 10;  
    takeFoo(x);  // 这里会编译错误，因为 Foo(int value) 构造函数是 explicit 的  
    return 0;  
}
```

### final

 **final防止类被继承**
  
  ```c++
  class Base final {  
public:  
    void foo() {  
        // ...  
    }  
};  
  
class Derived : public Base { // 错误！不能从final类继承
public:
    void bar() {
        // ...
    }
};
  ```

 **final禁止虚函数被重写**

```c++
class Base {  
public:  
    virtual void foo() final {
        // ...  
    }  
};  
  
class Derived : public Base {
public:  
    void foo() override { // 错误！不能重写final虚函数  
        // ...  
    }  
};
```

### override

* 类的成员函数声明中明确函数是为了重写基类中的虚函数。
* override 可以帮助编译器在编译时检查是否正确地重写了基类中的虚函数，如果基类中没有相应的虚函数，编译器会报错。

**override优势**
  * 提高代码可读性：通过在派生类中使用 override 关键字，可以清晰地表明该成员函数是为了重写基类中的虚函数。
  * 编译时检查：编译器会检查基类是否确实有一个虚函数与派生类中的函数具有相同的签名。如果没有，编译器会报错，这有助于早期发现错误。
  * 防止意外重载：有时候，由于函数签名的不匹配（例如参数类型或数量的不同），我们可能无意中重载了基类的虚函数而不是重写它。使用 override 可以避免这种情况。

```c++
class Base {  
public:  
    virtual void func() {  
        std::cout << "Base::func()" << std::endl;  
    }  
};  
  
class Derived : public Base {  
public:  
    void func() override { // 使用 override 关键字  
        std::cout << "Derived::func()" << std::endl;  
    }  
};  
  
int main() {  
    Derived d;  
    d.func(); // 输出 "Derived::func()"  
    return 0;  
}
```

 **override注意事项**
  * 即使不使用 override 关键字，只要函数签名匹配，也可以重写基类的虚函数。但是，使用 override 可以提供更好的安全性和可读性。
  * 如果基类中的虚函数被声明为 final，那么它不能在派生类中被重写。在这种情况下，尝试使用 override 关键字会导致编译错误。
  * 只能用于非静态成员函数，且必须是类的成员函数（不能是友元函数）。



### volatile

 **volatile定义**
 * volatile告诉编译器，某个特定的变量可能在程序的外部被改变，因此编译器不应将其视为优化的一部分。
   * 在大多数情况下，编译器会尝试优化代码以提高性能，例如，它会假设变量在两次读取之间不会被其他线程或硬件事件修改。
   * 然而，在某些特殊情况下，例如访问硬件寄存器、多线程编程中的共享变量，或者与某些信号处理函数交互时，这种假设可能不成立。
   * 在这些情况下，使用volatile可以确保编译器不会做出这些假设，并且每次使用变量时都会从内存中读取其值。

 **volatile应用**
  * 阻止编译器优化：编译器可能会进行某些优化，比如将变量的值存储在寄存器中，而不是每次都从内存中读取。如果变量被声明为 volatile，那么编译器就不会进行这样的优化，每次使用这个变量时都会直接从内存中读取。
  * 硬件寄存器访问：在嵌入式系统或操作系统内核中，经常需要直接访问硬件寄存器。由于这些寄存器的值可能随时由硬件改变，因此应该使用 volatile 来确保每次访问都是直接从硬件读取的。
  * 多线程编程：在多线程编程中，一个线程可能修改了一个全局变量的值，而另一个线程正在读取这个变量的值。如果这个变量没有被声明为 volatile，那么编译器可能会缓存这个变量的值，导致第二个线程读取到的是旧值。使用 volatile 可以防止这种情况。

```c++

// 基本类型变量：每次使用a时，编译器都会直接从内存中读取其值，而不是从寄存器或缓存中读取。
volatile int a = 0;

// 指针：它们指向的变量的值可能随时被改变，因此编译器不会对其进行优化。
volatile int *p = &a;
int *volatile q = &b;
volatile int *volatile r = &c;

// 多线程：
volatile bool flag = false;  
  
void thread1() {  
    while (!flag) {  
        // do something  
    }  
    // do something else  
}  
  
void thread2() {  
    // do some work  
    flag = true;  
}

// 硬件寄存器访问：直接读取或修改硬件寄存器的值，编译器不会对这些操作进行优化，确保了直接和硬件的交互。
#define HARDWARE_REGISTER_ADDRESS 0x12345678  
volatile uint32_t* hardware_register = (volatile uint32_t*) HARDWARE_REGISTER_ADDRESS;
  
// 读取寄存器值  
uint32_t value = *hardware_register;
  
// 修改寄存器值  
*hardware_register = new_value;
```

### enum

 C++的枚举是一种用户定义类型，用于为一组相关值命名。它创建具有特定含义的符号常量，使得代码更具可读性和可维护性。

 **C++11之前的enum**

  ```c++
enum Color { RED, GREEN, BLUE };
  ```
  枚举常量默认从0开始，每个后续的枚举常量值比前一个增加1。例如，在上面的Color枚举中，RED的值为0，GREEN的值为1，BLUE的值为2。

  ```c++
enum Weekday { SUNDAY = 1, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY }; // SUNDAY的值为1，MONDAY的值为2
  ```

  * 枚举值会隐式地转换为整数：可以直接将枚举值赋给整数变量，或者将整数赋给枚举变量，而编译器不会报错。这种隐式转换有时会导致意外的错误。
  * 枚举没有作用域限制，这意味着枚举常量可以在定义它们的枚举类型之外被直接访问。这可能会导致命名冲突

 **C++11及之后的enum**
  * 枚举类型的作用域与封装性
    * enum class（也被称为enum struct），这是一种强类型枚举。enum class定义的枚举类型具有更强的作用域控制和类型安全性。与传统的enum不同，enum class的枚举值的作用域被限制在枚举类内部，这避免了命名冲突，提高了代码的可读性和安全性。要访问enum class中的枚举值，需要使用枚举类名作为作用域前缀。

```c++
  #include <iostream>  
  
enum class Color {  
    RED,  
    GREEN,  
    BLUE  
};  
  
int main() {  
    Color myColor = Color::RED; // 正确的使用方式，通过作用域运算符访问枚举值  
    // Color anotherColor = RED; // 错误！RED不在当前作用域内  
      
    if (myColor == Color::RED) {  
        std::cout << "The color is red." << std::endl;  
    }  
      
    // int colorValue = myColor; // 错误！不能将Color类型的枚举值直接赋值给int类型  
      
    return 0;  
}
```

  * 底层类型的指定
    * 可以为枚举类型指定底层类型。这意味着可以明确指定枚举值所对应的整数类型，例如unsigned int或char等。这提供了更大的灵活性，并允许程序员更好地控制枚举值的底层表示。

```c++
enum Color: unsigned int { RED, GREEN, BLUE };
```

  * 枚举值的显式初始化
    * 可以显式地为枚举值指定整数值，这允许程序员更精细地控制枚举值的表示。如果未显式指定，枚举值将从0开始，依次递增
  * 类型安全
    * 在C++11及以后的版本中，通过使用enum class，我们可以实现更强的类型安全。

```c++
enum class TrafficLight { RED, YELLOW, GREEN };
TrafficLight light = 1; // 错误：不能隐式地将整数转换为 TrafficLight 类型
TrafficLight light = static_cast<TrafficLight>(1); // 正确，但假设 1 对应 TrafficLight 的某个有效值
```

  * 位掩码与标志
    * 枚举也可以用来表示一组标志（flags），每个枚举值对应一个位。这对于处理需要同时设置多个选项的情况非常有用。

```c++
enum class Weekday {  
    SUNDAY = 1 << 0,  
    MONDAY = 1 << 1,  
    TUESDAY = 1 << 2,  
    WEDNESDAY = 1 << 3,  
    THURSDAY = 1 << 4,  
    FRIDAY = 1 << 5,  
    SATURDAY = 1 << 6  
};

Weekday weekends = Weekday::SATURDAY | Weekday::SUNDAY;
if (weekends & Weekday::SATURDAY) {  
    // 周末包含星期六  
}
```


  * 枚举与常量
    * 枚举常常被用来定义一组相关的常量，以代替直接使用字面量或魔法数字（magic numbers）。这样做的好处是提高代码的可读性和可维护性。

```c++
enum class HttpStatusCode {  
    OK = 200,  
    NOT_FOUND = 404,  
    INTERNAL_SERVER_ERROR = 500  
    // ... 其他状态码  
};

HttpStatusCode responseCode = HttpStatusCode::OK;  
// 使用枚举而不是直接的数字常量，使得代码更加清晰易懂，并且如果需要更改某个状态码的值，只需要在枚举定义中修改一处即可，无需在整个代码库中搜索和替换数字。
if (responseCode == HttpStatusCode::NOT_FOUND) {  
    // 处理未找到的情况  
}
```


### 字节对齐

 字节对齐（Byte Alignment）是一个重要的概念，它涉及到数据在内存中的布局和访问效率。字节对齐有助于优化程序性能，避免潜在的性能瓶颈。

 **字节对齐原因**
  * 硬件访问效率：许多处理器在访问未对齐的内存地址时效率较低。为了充分利用处理器的性能，通常需要将数据按照特定的对齐方式进行存储。
  * 内存利用率：虽然字节对齐可能会增加内存使用，但它有助于减少内存碎片，提高内存管理的效率。
  * 数据完整性：某些硬件平台在访问未对齐的内存时可能会导致数据损坏或产生异常。

 **对齐规则**
  * 基本对齐：每个数据类型的对象都按照其大小进行对齐。例如，char 类型通常占用1个字节，int 类型通常占用4个字节（这取决于平台和编译器），因此 int 类型的对象通常会从4的倍数地址开始。
  * 结构体对齐：结构体的对齐通常由其成员中最大的对齐要求决定。结构体的大小通常是其成员对齐要求的整数倍。


 **常见变量字节数**
  
  | type | bytes |
  | ----- | ----- |
  | char | 1 |
  | short | 2 |
  | int | 4 |
  | double | 8 |
  | int b[0] | 0 |

 **控制字节对齐**
  * 编译器指令：不同的编译器提供了不同的指令来控制对齐。例如，GCC编译器提供了__attribute__((aligned(n))，其中n是所需的对齐大小（以字节为单位）。C++11标准提供了alignas(n)方法
    * 标准与扩展：alignas()是C++11标准引入的一个关键字，用于提供标准的对齐控制机制。它是跨编译器和平台的，因此使用alignas()的代码更有可能在不同的编译器和操作系统上保持一致的行为。相对地，__attribute__((aligned(n)))是GCC编译器的一个特定扩展，它并非C++标准的一部分。这意味着使用__attribute__((aligned(n)))的代码可能无法在非GCC编译器上正确编译或运行。
    * 可移植性：由于alignas()是标准的一部分，因此它提供了更好的可移植性。使用alignas()的代码更有可能在不同的编译器和平台上无需修改即可运行。而__attribute__((aligned(n)))则可能需要在不同的编译器或平台上进行特定的调整或替换。
    * 未来兼容性：由于alignas()是C++标准的一部分，它更有可能在未来得到持续的支持和改进。而GCC的特定扩展可能会随着编译器版本的变化而发生变化或不再支持。
    * 语法与用法：在语法上，alignas()通常作为类型说明符使用，可以放在类型定义之前，而__attribute__((aligned(n)))则作为GCC的特定属性使用，通常放在变量或类型声明的末尾

```c++
struct MyStruct __attribute__((aligned(16))) {  
    int a;  
    double b;  
};

alignas(16) struct MyStruct {  
    int a;  
    double b;  
};
```

```c++
struct alignas(16) MyStruct {  
    int a;  
    double b;  
};
```
  * #pragma指令：某些编译器还支持使用#pragma指令来控制对齐。例如，MSVC编译器提供了#pragma pack指令。

```c++
#pragma pack(push, 1)  
struct MyStruct {  
    char a;  
    int b;  
};  
#pragma pack(pop)
```



### struct

 将多个不同类型的数据项组合成一个单一的单元。

 **C++11之前的struct**

  * 用于封装一组数据成员，并可以包含成员函数来操作这些数据。结构体通常用于创建复合数据类型，以简化代码和提高代码的可读性。
  * 访问权限：默认public，结构体所有成员均可以从外部直接访问

```c++
struct MyStruct {  
    int x;  
    double y;  
    char z;  
  
    void print() {  
        std::cout << "x: " << x << ", y: " << y << ", z: " << z << std::endl;  
    }  
};
```

 **C++11之后的struct**
  * 成员初始化列表；C++11引入了成员初始化列表的语法，允许在结构体构造函数中直接初始化成员变量。

```c++
struct MyStruct {  
    int x;  
    double y;  
  
    MyStruct(int a, double b) : x(a), y(b) {}  
};
```

  * 列表初始化：C++11引入了列表初始化的语法，使得结构体的初始化更加简洁和直观

```c++
MyStruct s = {10, 20.5}; // 使用列表初始化
```

  * 默认构造函数和析构函数：如果结构体没有定义任何构造函数，编译器会自动为其生成一个默认构造函数。同样地，编译器也会为结构体生成一个默认析构函数（如果需要的话）。

```c++
struct MyStruct {  
    int x;  
    double y;  
  
    MyStruct(int a) : MyStruct(a, 0.0) {} // 委托给另一个构造函数  
    MyStruct(int a, double b) : x(a), y(b) {}  
};
```

  * 删除和默认构造函数/析构函数：可以使用 =delete来显式删除构造函数、析构函数或成员函数，以防止它们被调用。同样地，可以使用 =default来显式请求编译器生成默认的构造函数、析构函数或成员函数。

```c++
struct MyStruct {  
    MyStruct() = delete; // 删除默认构造函数  
    ~MyStruct() = default; // 请求编译器生成默认析构函数  
};
```



### union

 * 允许在相同的内存位置存储不同的数据类型，但每次只能使用其中一种类型。union 提供了一种有效的方式来节省内存，特别是在你不需要同时访问多个类型的情况下。

```c++
#include <iostream>  
  
union Data {  
    int i;  
    float f;  
    char str[20];  
};  
  
int main() {  
    Data data;  
    data.i = 10;  
    std::cout << "data.i : " << data.i << std::endl;  
  
    data.f = 220.5;  // 每次当你为一个成员赋值时，其他成员的值都会变得不可预测，因为它们共享同一块内存。
    std::cout << "data.f : " << data.f << std::endl;  
  
    strcpy(data.str, "Hello");  
    std::cout << "data.str : " << data.str << std::endl;  
  
    return 0;  
}
```


 **union注意事项**
  * 内存共享：联合体的所有成员共享同一块内存。这意味着当你为其中一个成员赋值时，其他成员的值也会受到影响。
  * 大小：联合体的大小是其最大成员的大小。这是因为所有成员都需要能够适应最大的数据类型的空间。
  * 对齐：编译器可能会因为内存对齐的原因在联合体的成员之间插入填充字节。
  * 初始化：在 C++11 之前，联合体不能直接初始化。从 C++11 开始，你可以使用列表初始化来初始化联合体的第一个成员。

 **union使用场景**
  * 尽管 union 可以节省内存，但也带来了复杂性，特别是当涉及到类型安全和内存对齐时。因此，在大多数情况下，更推荐使用struct 或class 来组织数据。
  * 然而，在某些特定的硬件编程或低级编程场景中，union 可能会非常有用
    * 例如需要直接操作硬件寄存器或处理特定格式的二进制数据时。在这些情况下，union 可以提供一种灵活且高效的方式来访问和操作数据。

### typedef
 * 为现有的数据类型创建别名。通过使用 typedef，使代码更易于理解和维护，特别是当处理复杂的数据类型或需要多次使用相同的类型定义时。
   * 增强代码可读性：使用 typedef 可以使代码更易读。特别是当数据类型的定义很复杂或者需要多次使用时，为其定义一个有意义的别名可以显著提高代码的可读性。
   * 跨平台编程：在某些情况下，不同的平台或编译器可能使用不同的数据类型来表示相同的概念（例如，字的大小）。通过使用 typedef 和条件编译指令，可以创建跨平台兼容的数据类型。
   * 隐藏实现细节：通过 typedef，可以隐藏数据的内部表示，只暴露其接口。这有助于封装和抽象，使得代码更加模块化。

```c++

// 基本用法
typedef int Integer;
typedef float Float;
  
Integer a = 10;
Float b = 3.14;


// 结构体
typedef struct {
    int x;
    int y;
} Point;

Point p1;
p1.x = 10;
p1.y = 20;


// 指针
typedef int* IntPtr;  
  
IntPtr p = new int(10);



// 数组
typedef int IntArray10[10];  
  
IntArray10 arr;

// 函数指针
typedef void (*Callback)(int);

void myFunction(int value) {
    printf("%d\n", value);
}

int main() {
    Callback callback = myFunction;
    callback(42); // 输出: 42
    return 0;
}
```




### 静态this指针的存在和处理

 **this指针的处理逻辑**
  * 在编译期间，编译器处理类的定义和成员函数的声明。
    * 对于每个非静态成员函数，编译器知道这个函数是属于某个类的成员，并且这个函数在调用时需要访问对象。
    * 为了能够在函数体内访问该对象的成员，编译器在内部为这个函数添加了一个隐含的this指针参数。
  * this指针的值是在运行时确定的。
    * 当创建一个对象并调用其成员函数时，编译器生成的代码会确保将当前对象的地址作为this指针的值传递给该函数。这样，在函数内部，你就可以通过this指针来访问和修改该对象的成员。
  * 编译期间的this指针只是一个占位符或隐含参数，用于表示在运行时需要传递一个指向当前对象的指针。而this指针的实际值（即指向哪个对象的地址）是在运行时根据具体的对象实例来确定的。这种设计允许每个对象在调用其成员函数时都能通过它自己的this指针来访问和修改自己的成员变量。

 通过 this 指针，成员函数可以访问调用它的对象的所有成员（包括数据成员和成员函数）。

 **this用途**
  * 区分成员变量和局部变量：当成员变量和局部变量同名时，可以使用 this 指针来区分它们。
  * 返回调用对象本身：在需要返回调用对象本身的函数中，可以使用 return *this，返回对象的引用;
  * 构造函数中初始化列表中使用：当类的数据成员是另一个类的对象时，构造函数的初始化列表需要使用 this 指针。

 **this优点**
  * 当对象调用成员函数时，编译器会自动将对象的地址传递给 this 指针
  * this 指针是隐含于每个类的非静态成员函数中的，不需要程序员显式定义。
  * this 指针的类型是类的指针，即 类名* const
  * this 指针的值（即它所指向的地址）在成员函数执行期间是常量，不能改变。

 **谨慎使用delete this**
  * 释放当前对象所占用的内存。这通常意味着对象的析构函数会被调用，然后其占用的内存会被释放。
    * 多次删除：如果你已经删除了一个对象，再次调用 delete this 会导致未定义行为，因为你会尝试释放已经被释放的内存。
    * 在析构函数中调用：在对象的析构函数中调用 delete this 是不安全的。因为当你显式或隐式地删除一个对象时，其析构函数会被调用。如果析构函数内部再次调用 delete this，那么当析构函数返回时，对象的内存可能已经被释放，这会导致问题。
    * 一旦你调用了 delete this，对象的内存就被释放了，但是对象本身可能仍然存在于作用域中。这会导致对象的状态变得不确定，任何对对象的进一步操作都可能导致未定义行为。
    * 继承与多态：在涉及继承和多态的复杂场景中，delete this 可能会导致意想不到的行为，特别是当基类或派生类有不同的内存管理策略时。

 **什么情况可以使用delete this**
  * 在某些特定的、受控的上下文中，它可能是有意义的。例如，在某些自定义的内存管理策略中，或者在实现某些特定的设计模式（如对象池）时，你可能需要更精细地控制对象的生命周期。在这些情况下，你应该确保你完全理解 delete this 的行为，并确保它不会导致任何问题。


```c++
#include <iostream>
#include <list>
#include <memory>

class SelfDeletingObject {
public:
    static std::list<SelfDeletingObject*> objectPool;
    static std::shared_ptr<SelfDeletingObject> Create() {
        SelfDeletingObject* obj = new SelfDeletingObject();
        objectPool.push_back(obj);
        return std::shared_ptr<SelfDeletingObject>(obj, [](SelfDeletingObject* ptr) {
            // 当最后一个shared_ptr引用被释放时，这个lambda会被调用
            ptr->SelfDelete();
        });
    }
  
    void DoWork() {
        std::cout << "Object is doing work." << std::endl;
        // 模拟某些条件下对象决定自我删除
        if (/* some condition */) {
            SelfDelete();
        }
    }

private:
    SelfDeletingObject() {
        std::cout << "Object created." << std::endl;
    }

    ~SelfDeletingObject() {
        std::cout << "Object destroyed." << std::endl;
        // 从对象池中移除自己
        objectPool.remove(this);
    }

    void SelfDelete() {
        // 确保对象池中确实包含当前对象
        auto it = std::find(objectPool.begin(), objectPool.end(), this);
        if (it != objectPool.end()) {
            // 安全地删除自己
            delete this;
        } else {
            std::cerr << "Error: Attempting to delete object not in pool." << std::endl;
        }
    }
};

std::list<SelfDeletingObject*> SelfDeletingObject::objectPool;

int main() {
    // 创建对象，并使其通过shared_ptr管理
    auto obj = SelfDeletingObject::Create();
    obj->DoWork(); // 模拟工作，可能会触发SelfDelete()

    // 当obj离开作用域时，shared_ptr的析构函数会调用自定义删除器，
    // 这可能会再次调用SelfDelete()，但此时对象应该已经从池中移除了。
    return 0;
}
```


### 泛化常数constexpr

 * 在编译时计算表达式的值。它的主要用途是提供对常量的编译时计算，使得这些常量可以在编译时就确定其值，而不是在运行时。constexpr 可以用于变量、函数和类的构造函数。
   * constexpr 变量必须是常量表达式，并且必须在编译时就能确定其值。它们通常用于定义编译时常量，如数组的大小或模板参数。
   * constexpr 函数必须在编译时就能计算其结果，而且它的所有参数都必须是 constexpr 类型。这意味着函数内部不能包含非确定性的操作，如动态内存分配、文件I/O或运行时函数调用。
   * 如果一个类的构造函数被声明为 constexpr，那么这个类的对象就可以在编译时初始化。这对于那些需要在编译时就能确定其状态的类非常有用，如模板元编程或编译时计算的复杂数据结构。


```c++
// constexpr 变量
constexpr int arraySize = 10; // 编译时常量  
int myArray[arraySize];       // 使用编译时常量定义数组大小

// constexpr 函数
constexpr int add(int a, int b) {  
    return a + b;  
}  
  
constexpr int sum = add(2, 3); // sum 在编译时就确定了，值为 5


// constexpr构造函数
class MyClass {  
public:  
    constexpr MyClass(int value) : m_value(value) {}  
    int getValue() const { return m_value; }  
  
private:  
    int m_value;  
};  
  
constexpr MyClass myObject(5); // 编译时初始化
```

 **constexpr注意事项**
  * constexpr 并不保证函数或表达式在运行时不会执行，它只保证如果可能的话，这些操作会在编译时完成。
  * constexpr 并不要求所有的函数或表达式都在编译时计算，它只是一个建议或提示给编译器。编译器可以选择忽略 constexpr，并在运行时计算表达式的值。
  * 在某些情况下，即使使用了 constexpr，编译器也可能因为某些原因（如优化设置、函数复杂性等）选择在运行时计算表达式的值。




### decltype

  * 用于在编译时检查实体的类型，并返回该实体的类型。
  * 模板元编程：decltype 提供了一种在模板元编程和类型推导中非常有用的机制，因为它可以自动推断出表达式的类型，而无需显式地指定。
 
 **decltype可以应用的类型**
  * 变量
  * 表达式
  * 函数
  * 指针和数组

 **decltype不可引用的类型**
  * 未定义的标识符：如果尝试对未定义的标识符使用decltype，编译器会报错，因为它无法确定未定义标识符的类型。
  * 非常量表达式：虽然decltype可以用于表达式，但对于那些在执行时可能会改变其值的表达式，使用decltype可能并不总是有意义的。decltype主要用于在编译时确定类型，而不是在运行时。
  * lambda表达式：虽然lambda表达式在C++中是一种非常有用的特性，但直接使用decltype来获取lambda表达式的类型可能并不是很有用，因为lambda表达式的类型通常是唯一的、匿名的闭包类型。它们的类型对于大多数用途来说是不透明的，并且是不可移植的。

```c++
const int ci = 0;  
decltype(ci) x = 0; // x的类型是const int

int a = 5, b = 10;  
decltype(a + b) sum = a + b; // sum的类型是int

int foo() { return 0; }  
decltype(foo) func_ptr; // func_ptr的类型是int(*)()：一个指向不接受任何参数并返回int的函数的指针类型。
func_ptr = &foo; // 将foo的地址赋值给func_ptr  
int result = func_ptr(); // 通过func_ptr调用foo函数

int* ptr;  
decltype(ptr) another_ptr; // another_ptr的类型是int*

```


 **decltype与左值右值**

  * 左值右值
    * 左值是指那些有持久状态，可以取地址的对象。例如，变量就是左值。
    * 右值则是那些临时的、不可取地址的对象，如字面量或者表达式的临时结果。
  * decltype与左值
    * 当decltype用于左值时，它通常返回该左值的类型或该类型的引用。具体返回哪种类型取决于是否对表达式使用了括号。
    * 如果不使用括号，且表达式是一个左值，decltype会返回该左值的类型。
    * 如果使用了括号，即使表达式是一个左值，decltype也会返回该左值的引用类型。

```c++
int x = 42;  
decltype(x) a = x; // a的类型是int  
decltype((x)) b = x; // b的类型是int&，因为(x)是左值表达式
```

  * 当decltype用于右值时，它返回该右值的类型（非引用）。右值通常包括字面量、临时对象或表达式的结果。
  * 如果表达式是一个右值引用表达式，decltype会返回右值引用类型。

```c++
int&& rvalueRef = int(); // rvalueRef是一个右值引用  
  
decltype(42) d = 50; // d的类型是int，因为42是右值  
decltype(std::move(rvalueRef)) e = int(); // e的类型是int&&，因为std::move(rvalueRef)是右值引用表达式
```

### extern定义


 **extern处理逻辑**
  * 声明全局变量或函数在其他文件中定义。它在编译阶段和连接阶段各自扮演着重要的角色
    * 编译阶段让编译器知道某个变量或函数是在其他文件中定义的，并在当前文件中只作为引用存在。
    * 连接阶段，链接器则负责解决这些引用，确保它们正确地链接到实际的定义。这样，多个源文件就可以共享同一个全局变量或函数，实现跨文件的代码重用和交互。


 **extern举例**
```c++
// state.h
// 声明一个全局变量，但不定义它  
extern int programState;


// state.c
#include "state.h"  
  
// 定义全局变量  
int programState = 0; // 初始状态为 0


// file1.c
#include "state.h"  
  
void functionInFile1() {  
    // 使用全局变量  
    programState = 1; // 改变状态  
}

// file2.c
#include "state.h"  
#include <stdio.h>  
  
void functionInFile2() {  
    // 使用全局变量  
    printf("The program state is: %d\n", programState); // 输出状态  
}
```

 * 编译阶段，每个源文件都被单独编译成一个目标文件（如 .o 文件）。
 * 编译器会检查每个源文件中的引用是否合法，包括检查 extern 变量的引用。在 file1.c 和 file2.c 中，编译器知道 programState 是在其他地方定义的，因此它不会在这些文件中为 programState 分配内存空间。
 * 然后，在链接阶段，链接器将所有目标文件合并成一个可执行文件。在这个过程中，链接器会解析所有 extern 变量的引用，并查找这些变量在何处定义。在这种情况下，链接器会在 state.o（由 state.c 编译生成的目标文件）中找到 programState 的定义，并将其与 file1.o 和 file2.o 中的引用进行匹配。
 * 最终生成的可执行文件将包含 programState 变量的一个实例，这个实例在 state.c 中被初始化为0，并可以在 file1.c 和 file2.c 中通过 extern 声明来访问和修改。
 * 其中要求extern的变量声明和定义在同一名下的.c和.h




## lambda 表达式

lambda表达式创建函数对象的过程都是在编译期间进行的，而lambda表达式的计算过程则是在运行期间进行的

定义

* lambda表达式是一种匿名函数对象，捕获所在作用域中的变量，并执行特定操作。
* 提供了一种匿名函数的特性，可以编写内嵌的匿名函数，用于替换独立函数，而且更可读
* 本质上来讲， lambda 表达式只是一种语法糖，因为所有其能完成的工作都可以⽤其它稍微复杂的代码来实现。

lambda表达式

 * 从闭包[]开始，结束于{}，{}内定义的是lambda表达式体

```c++
auto basicLambda = [] { cout << "Hello, world!" << endl; };
basicLambda();
```

 * 带返回值类型的lambda表达式

```c++
auto add[](int a, int b) -> int{return a+b;};

auto multiply = [](int a, int b)-> {return a*b;};

int sum = add(2,3);
int product = multiply(2, 5);
```

 * []闭包

[]：默认不捕获任何变量
[=]：默认以值捕获所有变量；
[&]：默认以引用捕获所有变量；
[x]：仅以值捕获x，其它变量不捕获；
[&x]：仅以引用捕获x，其它变量不捕获；
[=, &x]：默认以值捕获所有变量，但是x是例外，通过引用捕获；
[&, x]：默认以引用捕获所有变量，但是x是例外，通过值捕获；
[this]：通过引用捕获当前对象（其实是复制指针）；
[*this]：通过传值方式捕获当前对象



```c++
int main() {
 int x = 10;
 
 auto add_x = [x](int a) { return a + x; }; 
 auto multiply_x = [&x](int a) { return a * x; }; 
 
 cout << add_x(10) << " " << multiply_x(10) << endl;
 // 输出：20 100
 return 0;
}
```

 * 声明为mutable，让lambda表达式表示为可以更改参数值

```c++
#include <iostream>
#include <functional> // 包含 std::function 的头文件

class MyClass {
public:
    int x = 10;

    // 使用 std::function 来定义返回一个可调用的对象
    std::function<int(int)> add() mutable  {
        return [this](int y) {
            x += y;
            return x + y; };
    }
};

int main() {
    MyClass obj;

    // 获取 add 方法返回的 lambda 表达式（现在被封装在 std::function 中）
    std::function<int(int)> adder = obj.add();

    // 使用这个 lambda 表达式（现在通过 adder 变量）来计算结果
    int result = adder(20);

    std::cout << "result: " << result << std::endl; // 50
    std::cout << "obj.x: " << obj.x << std::endl;  // 30

    return 0;
}
```

```c++

// 利用多态调用
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
class Base {
public:
    virtual int add(int y) = 0;
    virtual ~Base() {} // 通常，当类中有虚函数时，建议添加虚析构函数
};

class Derived1 : public Base {
public:
    int add(int y) override {
        return y + 1;
    }
};

class Derived2 : public Base {
public:
    int add(int y) override {
        return y + 2;
    }
};

int main() {
    std::vector<Base*> v = { new Derived1(), new Derived2() };

    // 注意 lambda 表达式的参数顺序和用法  
    int result = std::accumulate(v.begin(), v.end(), 0, [](int sum, Base* obj) {
        return sum + obj->add(10); // 通过多态调用add函数
        });

    std::cout << "result: " << result << std::endl; // 输出应该是 23 而不是 33

    // 清理动态分配的内存
    for (auto ptr : v) {
        delete ptr;
    }

    return 0;
}
```

应用于函数的参数，实现回调

```c++
int val = 3;
vector<int> v{1,8,3,4,7,3};
int count = std::count_if(v.begin(), v.end(), [val](int x) {return x >3;});
```

