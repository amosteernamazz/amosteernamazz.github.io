---
layout: article
title: C++基础
key: 100020
tags: C++
category: blog
date: 2023-02-07 00:00:00 +08:00
mermaid: true
---

### 问题

***把异常完全封装在析构函数内部，决不让异常抛出函数之外***

## **C++特性**

<br>

***<font color = purple>封装：原因</font>***
<br>

***<font color = purple>继承：原因、类型、缺点、class与struct的继承、什么不能继承</font>***
<br>

***<font color = purple>多态：原因、形式、实现</font>***
<br>


<br>


### 封装
 **原因**
  * 结合性：属性与方法结合
  * 信息隐蔽性：利用接口机制隐藏内部实现细节，只留下接口供外部调用
  * 实现代码复用

<!--more-->

### 继承
 **原因**
  * 可以使用父类的所有非私有方法
  * 继承父类中定义的成员方法以及成员变量，使得子类可以减少代码的书写
  * 重写父类的方法以增加子类的功能


 **类型**
  * 单一继承：继承一个父类，最多使用
  * 多重继承：一个类有多个基类，类之间使用逗号隔开，如果都含有
  * 菱形继承：BC继承自A，D继承自BC


 **缺点**
  * 耦合性太大
  * 破坏了类的封装性
  * 一般多用于抽象方法的继承和接口的实现


 **class 与struct**
  * class -> private继承
  * struct ->  public继承


 **什么不能继承**
  * 构造函数
    * 派生类构造函数通常使用成员列表初始化来调用基类构造函数以创建派生类中的基类部分，如果派生类没有使用成员列表初始化语法，则将使用默认的基类构造函数，如果基类没有默认的构造函数就会报错。
    * 在设计派生类时，对继承过来的成员变量的初始化工作也要由派生类的构造函数完成，但是大部分基类都有private属性的成员变量，它们在派生类中无法访问，更不能使用派生类的构造函数来初始化。
    * 派生类调用基类构造
  * 析构函数
  * 赋值运算符=
  * final
  * 自身构造和析构在private作用域，单例模式
  * friend + 虚继承

![](https://img-blog.csdn.net/20180626002635328?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1dhbl9zaGlidWdvbmc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

 **默认继承方式**
  * 类 -> private
  * 结构体 -> public

### 多态

 **原因**
  * 运行时类型决定了编译时类型修饰的变量名所调用的方法在程序最终执行过程中真实调用的方法


 **形式**
  * 静态多态：编译期间确定，对于相关的对象类型，直接实现各自的定义，不需要共有基类，只需要在各个具体类的视线中要求相同的接口声明
    * 重载、模板函数
  * 动态多态：程序运行时确定，对于对象，确定之间的共同功能，然后在基类中，将共同功能声明为多个公共的虚函数接口，子类重写虚函数，完成具体功能
    * 虚函数、基类引用指向子类对象


 **多态的实现**
  * 重载：编译期实现
  * 类、函数模板：编译期
  * 虚函数：运行期





## **C++关键字**

### NULL和nullptr

 **NULL**
  C++中，NULL实际上是0.因为C++中不能把void*类型的指针隐式转换成其他类型的指针，所以为了结果空指针的表示问题，C++引入了0来表示空指针
  ```c++
  #ifdef __cplusplus
  #define NULL 0
  #else
  #define NULL ((void*)0)
  #endif
  ```
  * 问题
    * 在函数重载的时候会出现问题，不确定参数是 void* 还是 int


 **nullptr**
  * 为解决NULL的二义性，使用nullptr替代空指针，对应参数为 void* 版本

### include<> 和 include ""

 **不同**
  * <>：标准库文件所在目录，编译器设置的include路径内
  * ""：当前源文件所在目录


### enum
  ```c++

  enum class B: unsigned char /** 每个枚举都是unsigned char类型的 */
  {
      my_enum3 = 0,
  }
  ```
 **特点**
  * 与整型之间不会发生隐式类型转换，除非用static_cast强制转换
  * 可以指定底层的数据类型，默认是int
  * 需要通过域运算符来访问枚举成员


### auto

 **目的**
  * 避免太长，影响代码可读性
  * 类型不是我们所关注的，也会用

 **应用**
  * 可以推断基本类型
  * 可以推断引用类型
  * **推断发生在编译期**，所以使用auto并不会造成程序运行时效率的降低。

### inline

 **优点**
  * 在调用处进行代码展开，不用参数压栈等
  * 可以进行安全检查
  * 可调试

 **缺点**
  * 代码膨胀
  * inline函数的改变需要**重新编译**

 **虚函数是否可以为inline**
  * 虚函数只有在编译器知道所调用的对象是哪个类才可，表现多态是不可内联


### decltype
 **目的**
  * 为了解决复杂的类型声明而使用的关键字
 
 **应用**
  * 多出现在泛型编程，编译期间确定


### explicit

 **目的**
  * 防止隐式转换

 **应用**
  * 应用于仅含有一个参数，或除第一个参数外，其他参数都有默认值的构造函数


### friend

 **友元函数**
  * 使得普通函数直接访问类的保护数据和私有数据成员，避免了类成员函数的频繁调用，可以节约处理器开销，提高程序的效率
  * 编译期间确定
 
 **使用注意**
  * 当某类函数需要另外一个类的私有成员，在另外那个类声明friend，并引入需要私有成员的类`friend int B::func(A& a)`
  * 在需要类成员的类中声明`int func(A& a)`
  * 最后在类外实现方法`int B::func(A& a)`

 **友元类**
  * 友元类的所有成员函数都是另一个类的友元函数，都可以访问另一个类中的隐藏信息




### 结构体
 **定义**
  * **不同类型**的数据组成整体


 **特点**
  * 每个成员都有自己的独立地址
  * **sizeof**之后是内存对齐之后所有成员的长度和


 **struct与数组**
  * 原因：数组空间有限
    * 在结构体和类中定义空数组，其数组内容不占用struct空间，可以在最后分配malloc中直接分配动态buffer，`malloc(sizeof(struct()+buff.size()))`


### 共同体
 **定义**
  * 各成员共享一段内存空间，大小为成员中最长成员的长度


 **特点**
  * 共享内存可以将变量付给任一成员，但每次只能赋一种值，会覆盖



### 字节对齐
 **引入原因**
  * CPU访问数据效率问题，对于0-7存储的变量读取只需要一次，否则需要两次


 **对齐规则**
  * x86 -> GCC默认4字节对齐
    * 可以使用`__attribute__`选项改变对齐规则
  * vs中使用`#pragma pack (n)`改变


 **常见变量字节数**
  
  | type                       | bytes                             |
  | -------------------------- | --------------------------------------- |
  | char                   | 1                                       |
  | short               | 2                                       |
  | int                | 4                     |
  | double               | 8                                     |
  | int b[0]                     | 0                                       |


 **struct对齐**
  * 对char、char、int类型有8个bytes
  * 对char[5]、short、int类型有12个bytes
  * 对char、double、char类型有24个bytes
  * 数组`int[]`或`int[0]`指向前一个位置，本身不占空间
    * char、int、int[]有8个bytes
    * int[]占用4个bytes
    * 空结构体占用1 bytes



### arr & arr[0]、&arr区别

 **print**
  * 都是首元素地址


 **+1 print**
  * `arr[0] + 1`和 `arr + 1`在数组内移动
  * `&arr + 1`按照数组单位移动


### char、char a[]、char* a、char* a[]、char** a

  | type                       | 含义                             |
  | -------------------------- | --------------------------------------- |
  | char a                   | 定义存储空间，存储char类型变量                                       |
  | char a[]               | 字符数组，每个元素都是char类型数据                                       |
  | char *a                | 字符串首地址                     |
  | char *a[]               | 表示char数组，数组元素为指针，指针指向char类型                                     |
  | char **a                     | 与char *a[]相同                                       |

### 一维数组名和二维数组名的区别

 **相同点**
  * 存储都是一维的
  * 一维数组名与二维数组名都指向数组的指针

 **不同点**
  * 二维数组名不能赋给二级指针
    * 二级指针：要求指向的是指针，而二维数组确定一维后指向数组
      * 想要获得 a[i] 中第 x 个元素，可以直接使用 `*(a+x)`
      * 想要获得 `b[i][j] `中第 x 行第 y 个元素，则需用 `*(*(b+x)+y)`
  * 一维数组+1跳过对应值，二维数组跳过行或列





### static

#### static与局部变量

 **特点**
  * 存储位置data区
  * 生命周期保持不变
  * 局部作用域退出时，数据仍然暂存data区


#### static与全局变量

 **特点**
  * 加入static后，源程序的其他源文件不能再使用该变量（不使用static可以用extern扩展）


#### static与函数

 **特点**
  * 跟全局变量相同，限制作用域，只能在该文件中使用（与全局变量用法也相同）


#### static与类对象成员变量

 **特点**
  * 变量会变成类的全局变量，只能在类外初始化
    * 但如果加入const修饰，则可以在类内初始化


#### static与类对象成员函数

 **特点**
  * 类只存在一份函数，所有对象共享，不含this指针，无需创建实例即可访问
    * 不可同时用const和static修饰对象的成员函数





### const

#### const与变量

 **特点**
  * 限定不可更改


#### const与指针

 **特点**
  * `int const * a`与`int * const a`


 **指向常量的指针**
  * `const int * a`
  * `int const * a`


 **指针常量**
  * `int * const a`


#### const与函数
  * `const int& func(int& a)`：修饰返回值为const
  * `int& func(const int& a)`：修饰形参
  * `int& func(int& a) const{}`：const成员函数
    * 不允许修改类的成员的值


#### const与类
  **const修饰类成员变量**
   * 在对象的声明周期内是常量，对整个类而言是可以改变的。
   * 不能在类内初始化const成员变量，在初始化列表中初始化。


  **const类对象成员函数**
   * 不允许修改类的成员的值


  **const对象**
   * 只能调用const函数


### static与const总结

**static的作用是表示该函数只作用在类型的静态变量上，与类的实例没有关系**

**const的作用是确保函数不能修改类的实例的状态**
static和const不可同时修饰成员函数



### 引用与指针
  ```c++
  int i = 5;
  // 引用
  int &a = i;
  a = 8;
  ```


#### 引用的本质
 * `& = T * const a`
 * 本质是常量指针


#### 引用与常量指针相同点
 * 都占用4/8字节
 * 都必须初始化


#### 引用与常量指针不同点
 **是否可寻址**
  * 指针常量允许寻址 -> &p返回指针常量的地址 *p返回被指向对象
  * 引用不允许寻址  -> &r返回指向对象的地址

 **是否可空**
  * 指针常量 -> 可NULL
  * 引用 -> 不允许NULL

 **是否支持数组**
  * 指针常量 -> 支持
  * 引用 -> 不支持

 **参数传递**
  * 指针常量 -> 值传递，局部变量
  * 引用  ->引用传递，实参传递

 **sizeof()的不同**
  * 指针常量 -> 指针的大小
  * 引用 -> 得到指向对象的大小

### const -> #define
 **不同**
  * 类型
    * 宏定义是字符替换，没有数据类型的区别
    * const常量是常量的声明，有类型区别
  * 安全检查
    * 可能产生边际效应等错误
    * 在编译阶段进行类型检查
  * 编译器处理
    * 宏定义是一个"编译时"概念
    * const常量是一个"运行时"概念
  * 存储方式
    * 宏定义：代码段
    * const常量：data区
  * 是否可以做函数参数
    * 可以在函数的参数列表中出现

### #define -> typedef

 **不同**
  * 编译器处理
    * typedef在编译阶段，有类型检查的功能
    * define则是宏定义，发生在预处理阶段
  * 作用域的限制
    * define没有作用域的限制
    * typedef有自己的作用域
  * 指针操作不同
    * typedef int * pint; const pint p1 = &i1;  指针常量
    * #define PINT int * const PINT p2 = &i2; 常量指针



### this指针

 **作用**
  * 指向非静态成员函数所作用的对象

 **什么时候创建**
  * 调用非静态函数时才会使用的

 **delete this**
  * 为将被释放的内存调用一个或多个析构函数（因此不能在析构中调用delete this），类对象的内存空间被释放，之后不能涉及this指针，如操作数据成员，调用虚函数等





### 强制类型转换
 **static_cast**
  * 基本数据类型之间的转换
    * void*和其他类型指针之间的转换
    * 子类对象的指针转换成父类对象指针
  * 最好所有隐式转换都用static_cast代替

 **dynamic_cast**
  * 用于安全的向下转型
    * 转换成功会返回引用或者指针，失败返回null，否则会抛出一个`bad_cast`的异常类型

 **const_cast**
  * 用于移除指针和引用的常量性，但是不能改变原来常量的常量性
    * 指向常量的指针被转化成非常量指针
    * 常量引用被转换成非常量引用
    * 常量对象被转换成非常量对象

 **reinterpret_cast**
  `reinpreter_cast<type-id> (expression)`
  * 可以将任意类型指针转换为其他类型的指针。所以他的type-id必须是一个指针、引用、算术类型。
  * 能够在非相关的类型之间转换。它可以把一个指针转换成一个整数，也可以把一个整数转换成一个指针。

 **应用**
  * 一般不要使用dynamic_cast



## **c++运行时特征**


### 阶段

 * 预编译
   * #define、#if、#ifdef、#ifndef、#undef、#endif、#else
 * 编译期
   * enum、struct、union、auto、decltype、friend、声明、全局static、constexpr、inline、全局变量、vtable
   * 第二步：编译期创建的对象
     * 需要使用动态初始化
     * 执行顺序：基类局部static -> 子类局部static -> 非静态代码 -> 派生类的基类构造(执行基类普通成员的初始化，按照基类对象声明顺序执行) -> 派生类的成员构造(按照成员声明顺序) -> 派生类构造(派生类普通成员初始化)（包括vptr）
   * 编译期析构
     * 执行顺序：派生类析构 -> 派生类成员类对象析构 -> 基类析构
 * 运行期
   * 运行期创建的对象
     * 执行顺序：基类局部static -> 子类局部static -> 非静态代码 -> 派生类的基类构造(执行基类普通成员的初始化，按照基类对象声明顺序执行) -> 派生类的成员构造(按照成员声明顺序) -> 派生类构造(派生类普通成员初始化)（包括vptr）
   * 运行期析构
     * 执行顺序：派生类析构 -> 派生类成员类对象析构 -> 基类析构
   * 运行期virtual

### C++初始化

 **类型**
  * 编译初始化
  * 动态初始化

#### 编译初始化

 * 静态初始化在程序加载的过程中完成
 * 包括全局变量初始化和constexpr类型的初始化
   * zero initialization 的变量会被保存在 bss 段
   * constexpr initialization 的变量则放在 data 段内
   * 其次全局类对象也是在编译器初始化。

#### 动态初始化
 出现时机：出现在编译期和运行期的局部位置初始化
 * 动态初始化也叫运行时初始化
 * 需要经过函数调用才能完成的初始化、类初始化
   * 局部静态类对象的初始化
   * 局部静态变量的初始化
 * 动态初始化一般出现在

### 动态初始化中静态局部变量2个问题

 **线程安全问题**

 实现方法
  * 一个线程在初始化 m 的时候，其他线程执行到 m 的初始化这一行的时候，就会挂起而不是跳过
    * 局部静态变量在编译时，编译器的实现是和全局变量类似的，均存储在bss段中。
    * 然后编译器会生成一个保证线程安全和一次性初始化的整型变量，是编译器生成的，存储在 bss 段。
      * 它的最低的一个字节被用作相应静态变量是否已被初始化的标志
        * 若为 0 表示还未被初始化，否则表示已被初始化(if ((guard_for_bar & 0xff) == 0)判断)。 
      * __cxa_guard_acquire 实际上是一个加锁的过程，
        *  相应的 __cxa_guard_abort 和__cxa_guard_release 释放锁。

 **内存泄漏问题**

 原因
  * 在局部作用域消失时，data区仍然保存其内存空间
  * 执行路径不明
    * 对于局部静态变量，构造和析构都取决于程序的执行顺序。程序的实际执行路径不可预知的
  * 关系不明
    * 局部静态变量分布在程序代码各处，彼此直接没有明显的关联，很容易让开发者忽略它们之间的这种关系

 建议
  * 减少使用局部静态变量



### 对象构建顺序

  1. 先执行base的static，再执行派生类static（按出现顺序）
  2. 非静态代码（成员方法，成员变量，成员代码块等）如果有类对象则按照顺序构建
  3. 构造函数
     1. 基类普通成员初始化、基类构造（按照基类在派生类中出现的顺序，而不是成员初始化顺序）
     2. 成员类对象构造函数（如果有多个成员类构造函数，调用顺序是对象在类中被声明的顺序）
     3. 派生类普通成员初始化、派生类构造

### 析构顺序

  1. 派生类析构
  2. 成员类对象析构
  3. 基类析构








### 构造函数


 构造类型

 **默认构造**
  * 无参构造


 **一般构造**
  * 包含各种参数，参数顺序个数不同可以有不同构造


 **拷贝构造**
  * 函数参数必须为引用
    * 如果是值传递，则会递归调用拷贝构造
    * 如果是指针类型，则为值传递
  * 浅拷贝，存在问题，进行重写


 **移动构造**
  * 避免分配新空间，将原来的对象直接拿过来使用


 **赋值构造**
  * 
 **类型转换构造**
  * 
 


 构造中的问题
 
 **构造中的内存泄漏问题**
  * 原因：c++只会析构已经完成的对象。
  * 出现：如果构造函数中发生异常，不会调用析构函数。如果在构造函数中申请了内存操作，则会造成内存泄漏。
  * 派生类有问题：如果有继承关系，派生类中的构造函数抛出异常，那么基类的构造函数和析构函数可以照常执行的。
  * 解决办法：用智能指针来管理内存

### 拷贝构造中的浅拷贝和深拷贝

 **使用深拷贝的场景**
 * 在copy构造中，copy的对象是否存在指针，如果有需要重写copy构造，因为浅拷贝不会存储数据，相同指针指向同一对象，**当数据成员中有指针时，必须要用深拷贝。**

 **系统默认**

  系统默认的拷贝函数——即浅拷贝。当数据成员中没有指针时，浅拷贝是可行的；

 **原因**
  如果没有自定义拷贝构造函数，会调用默认拷贝构造函数，这样就会调用两次析构函数。**第一次析构函数delete了内存，第二次的就指针悬挂了。**所以，此时，必须采用深拷贝。

 **操作**
  * 深拷贝在堆内存中另外申请空间来储存数据，从而也就解决了指针悬挂的问题。
  简而言之，**当数据成员中有指针时，必须要用深拷贝**。

### 析构函数
 **析构中的问题**
  * 析构函数不能、也不应该抛出异常
    * 析构函数抛出异常，则异常点之后的程序不会执行，造成资源泄露
    * 异常发生时，异的传播过程中会进行栈展开。调用已经在栈构造好的对象的析构函数来释放资源，此时若其他析构函数本身也抛出异常，则前一个异常尚未处理，又有新的异常，会造成程序崩溃。
      * 解决办法：把异常完全封装在析构函数内部，决不让异常抛出函数之外


### 实例化
 **不可实例化的类**
  * 抽象类（本身是抽象）
  * 工具类（直接通过static调用函数）

 **如何阻止实例化**
  * 包含纯虚函数
  * 构造函数私有

#### 实例化中的变量初始化时机与顺序

 时机
 * 类中const初始化必须在构造函数初始化列表中初始化
 * 类中static初始化必须在类外初始化
 * 成员变量初始化顺序按照类中声明顺序，而构造函数初始化顺序按照成员变量在构造函数中位置决定

 顺序
 * 初始化base类中的static部分（按程序出现顺序初始化）
 * 初始化派生类中的static部分（按程序出现顺序初始化）
 * 初始化base类的普通成员变量和代码块，再执行父类的构造方法；
 * 初始化派生的普通成员变量和代码块，在执行子类的构造方法；




### 运行期virtual
![](https://img-blog.csdnimg.cn/20201005162842163.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlc2lnag==,size_16,color_FFFFFF,t_70#pic_center)

 **原理**
  * 对象中有**虚函数的指向的指向**（编译期间**创建对象**或运行时创建对象时创建）和虚函数表（每个**类的虚函数入口地址**，为**编译期**创建）

  * 管理对象的空间中有vptr地址（随对象创建而创建），vptr指针对应的vtable（在编译期确定，是针对类的）中保存该对象的虚函数成员，其保存函数的入口地址


 **多继承**
  * 在多继承中，vtable会有多个vptr地址，对应不同基函数的vptr


 **运行时virtual**
  * 为了多态，编译器会给每个包含虚函数或继承了虚函数的类自动建立一个虚函数表，当子类继承父类的虚函数时，子类会有自己的vtable
    * 如果存在**大量的子类继承**，且重写父类的虚函数接口只占总数的一小部分的情况下，会造成大量地址空间浪费


 **多态的实现原理（为什么构造和析构需要按顺序）**

  [虚表的写入时机、多态的实现原理、构造析构顺序的原因](https://blog.csdn.net/weixin_43919932/article/details/104356460)

 ![](https://img-blog.csdnimg.cn/20200217142930458.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzkxOTkzMg==,size_16,color_FFFFFF,t_70)



 **构造函数与virtual**

  构造函数是否可以为virtual
  * 不能
  * 虚函数调用需要对象构建得到虚表调用，而对象还没有构造。

  构造函数中调用virtual
  * 破坏C++动态联编
    * 父类对象构造中在子类之前，子类的成员还没有初始化


 **析构函数与virtual**
 
  析构函数的子类应该声明为virtual
  * 为了确保析构的时候，释放派生类对象，需要基类析构函数声明为虚函数，否则只会析构对应的父类对象，而不会析构子类对象。


 **哪些函数不能是虚函数**
  * 构造函数
  * 某些析构函数
  * 友元函数（原因不是类成员）
  * 静态成员函数（原因：不属于任何对象或实例）
  * 内联函数（原因：需要在编译期间展开，同时需要类对象有vptr，但没有地址）
  * 成员函数模板（原因：成员模板函数需要在调用的时候才能确定，而虚函数需要解析时候确定vtable大小）


### 纯虚函数
 **区别**
  * 纯虚函数用于如果生成基类对象则不合理的场景
  * 其使得纯虚函数的类为抽象基类，本身成为了接口


 **使用**
  `virtual void exit()=0`=0表示为纯虚函数

### 堆上和栈上的对象

 **区别**
  * 声明周期
    * 需要生命周期比上下文长的生命周期，则只能在堆上创建
      * 只要能在栈上创建对象，就在栈上创建；否则的话，如果你不得不需要更长的生命周期，只能选择堆上创建
    * 某些情况如果是在栈上创建，但数据仍然在堆上`std::vector v`对象v创建在栈，但其数据在栈上
  * 性能
    * 栈性能更快，栈有专门的寄存器，压栈出栈指令效率更高，堆是由OS动态调度，堆内存可能被OS调度在非物理内存中，或是申请内存不连续，造成碎片过多等问题；
    * 堆都是动态分配的，栈是编译器完成的。栈的分配和堆是不同的，他的动态分配是由编译器进行释放，无需我们手工实现

 **只在堆上生成对象的类**
  ```c++
  class A {
    // A a; 创建对象是利用编译器确定，需要public的构造和析构，因此使用private或protected构造和析构可以取消静态创建，但针对需要继承的类型有进一步限制为protected
  protected:
    A(){}
    ~A(){}
  public:
    static A* create(){
      return new A();
    }
    // 在析构中因为无法调用，使用单独的delete()函数
    void delete(){
      delete this;
    }

  };
  ```

 **只在栈上生成对象的类**
  ```c++
  class A{
  private:
    void operator delete(void* ptr){}
    void * operator new (size_t t){}
  public:
    A(){}
    ~A(){}
  };
  ```


### C++内存布局


#### 栈
 **特点**
  * 向下生长
  * 保存函数的局部变量，参数以及返回值


#### 堆
 **特点**
  * 向上生长
  * 用户分配的动态内存区域，存在内存泄漏问题，需要及时释放内存，否则需要等程序退出


#### bss区
 **特点**
  <!-- * global 初始化在此区，初始化后的非0放在data
  * 未初始化的static全局/局部变量 -->
  * 编译时：对程序全局变量载入时，由内核置为0
  * 运行时：未初始化的static全局/局部变量


#### data区
 **特点**
  <!-- * const初始化在此区
  * 初始化后非0的global、初始化后的static全局/局部变量 -->
  * 初始化后的全局变量
  * 初始化后的static全局/局部变量
  * 编译时：const初始化后的const


#### text段
 **特点**
  * 只读，一般为二进制文件









## c++内存管理

### new/delete与malloc/free
 **相同**
  * 申请动态内存和释放动态内存


 **不同**
  * 返回类型安全性 （new返回安全，malloc返回`void *`）
  * 返回失败后返回值 （new失败后要捕获异常`bad_alloc`，malloc返回nullptr）
  * 是否指定内存大小（new不，malloc需要）
  * 后续内存分配（new 没有配备，malloc如果不够，使用realloc进行扩充）


 **应用上共存**
  * 对于需要初始化的场景，使用new更合适
  * 对于c程序需要使用malloc/free管理内存


 **配对**
 new和delete、malloc和free、new[]和delete[]要配对使用

### free原理
 * glibc中的free，空间的大小记录在参数指针指向地址的前面，free的时候通过这个记录即可知道要释放的内存有多大。
 * 同时free(p)表示释放p对应的空间，但p这个pointer仍然存在，只是不能操作
 * free后的内存会使用双链表保存，供下次使用，避免频繁系统调用，同时有合并功能，避免内存碎片

 **使用**
  * `char* p = (char*) malloc(10);`
  * `free(p);`
  * `p = NULL;`

### 栈上分配内存
 **alloca**
  * 不需要手动释放，超出作用域自动释放


 **问题**
  * 会爆栈


### 内存泄漏
 **原因**
  * malloc/new和delete/free没有匹配
  * new[] 和 delete[]没有匹配
  * 没有将父类的析构函数定义为虚函数


 **监测手段**
  * 把new封装在构造函数中，将delete封装到析构函数中
  * 智能指针
  * valgrind ，这个可以打印出发生内存泄露的部分代码
  * linux使用swap命令观察还有多少可以用的交换空间，两分钟内执行三四次，肉眼看看交换区是不是变小了
  * 使用/usr/bin/stat工具如netstat、vmstat等。如果发现有内存被分配且没有释放，有可能进程出现了内存泄漏。

### 智能指针 shared_ptr
 * 是RAII类模型，用来动态分配内存
   * 将指针用类封装，然后实例化为对象，当对象过期，让析构函数删除指向的内存

 **shared_ptr**
  * 多个指针可以指向一个相同的对象，当最后一个shared_ptr离开作用域的时候才会释放掉内存。


 **实现原理**
   * 在shared_ptr内部有一个共享引用计数器来自动管理，计数器实际上就是指向该资源指针的个数
   * 每当复制一个 shared_ptr引用计数会 + 1
     * `shared_ptr<A> sp2(sp1);`
     * `shared_ptr <A> sp3; sp3 = sp2;`
   * 当一个 shared_ptr 离开作用域时，引用计数会 - 1
     * `sp3.reset(new A(3));`
   * 当引用计数为 0 的时候，则delete 内存。
   * 这样相比auto来说就好很多，当计数器为0的时候指针才会彻底释放掉这个资源。
   * 注意**不能**将两个shared_ptr托管同一个指针
     * `shared_ptr <A> sp1(p), sp2(p); //error!!!`


 **线程安全**
  * 同一个shared_ptr，多个线程读是安全的
  * 同一个shared_ptr，多个线程写是不安全的
    * 线程在指向修改到计数器变化两个过程中并非原子操作，中间可能被打断
    * 方法：加锁
  * 不同的shared_ptr，多个线程写是安全的
  * shared_ptr管理数据的安全性不明

 **线程不安全例子**

  ```c++
  shared_ptr<foo> o1;
  shared_ptr<foo> p2(new foo);
  shared_ptr<foo> p3(new foo);
  p1 = p2;
  p2 = p3; // 可能出现p1悬空指针
  ```
  ![](https://img-blog.csdnimg.cn/20200525124202197.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg1MDQ3NA==,size_16,color_FFFFFF,t_70)

  ![](https://img-blog.csdnimg.cn/20200525124656242.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg1MDQ3NA==,size_16,color_FFFFFF,t_70)

  ![](https://img-blog.csdnimg.cn/2020052513010518.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg1MDQ3NA==,size_16,color_FFFFFF,t_70)

  ![](https://img-blog.csdnimg.cn/20200525130206633.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg1MDQ3NA==,size_16,color_FFFFFF,t_70)


 **方法**

 ![](https://www.cnblogs.com/JCpeng/p/15031742.html)


  * 构造函数
  ```c++
  std::shared_ptr<int> p1;  // p1.use_count() = 0
  std::shared_ptr<int> p2(nullptr);  // p1.use_count() = 0
  std::shared_ptr<int> p3(new int);
  std::shared_ptr<int> p4(new int, std::default_delete<int>());
  ```

  * reset
    * reset()会释放并摧毁原生指针
    * reset(param)会管理这个新指针
  * make_shared()
    * 缺点
      * 构造函数为protected或private时，无法使用make_shared
      * 对象的内存可能无法及时回收
  * swap方法
    * 交换两个shared_ptr所拥有的对象，即指向的对象交换
  * shared_from_this
    * 如果当前对象需要交给某个对象来管理，则当前对象生命周期需要晚于某个对象，为实现上述目标，类继承`std::enable_sharded_from_this<T>`
  
  ```c++
    class Widget: public std::enable_shared_from_this<Widget>{
      public:
        void do_something(A& a){
          a.widget = shared_from_this();
        }
    }
  ```

 **shared_ptr和make_shared区别**

  性能
   * shared_ptr初始化需要分配两次内存
   * make_shared初始化只需要分配一次内存


![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNDM2ODIwMS0xNzlhNWU3ZWZlYzM0ZWIyLnBuZz9pbWFnZU1vZ3IyL2F1dG8tb3JpZW50L3N0cmlwJTdDaW1hZ2VWaWV3Mi8yL3cvNDgwL2Zvcm1hdC93ZWJw?x-oss-process=image/format,png)

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNDM2ODIwMS0zNDRhOWY3N2E0MGU3ZDQ0LnBuZz9pbWFnZU1vZ3IyL2F1dG8tb3JpZW50L3N0cmlwJTdDaW1hZ2VWaWV3Mi8yL3cvNDgwL2Zvcm1hdC93ZWJw?x-oss-process=image/format,png)

  内存泄漏问题

   * shared_ptr可能会产生异常问题，而maked_shared可以避免问题
  ```c++
  processWidget(std::shared_ptr<Widget>(new Widget), computePriority());
  ```
   * 上述代码可能会有资源泄露，如果先执行`new Widget`、执行`computePriority()`、执行`std::shared_ptr`的构造，因此如果`computeProirity()`中如果产生异常，则Widget会泄漏


  ```c++
  processWidget(std::make_shared<Widget>(),computePriority());
  ```
   * 不存在内存泄漏问题

### 智能指针 weak_ptr

 **目的**
  * 解决shared_ptr指针循环引用出现内存泄漏问题

 **方法**
  * `ptr.expired()`判断是否被释放
  * `ptr.use_count()`返回原生指针引用次数
  * `std::shared_ptr<CTxxx>ptr2 = ptr.lock()`返回为空的shared_ptr或转化为强指针`shared_ptr`
  * `reset()`将本身置为空

 **应用场景**
  * 观察者功能

  ```c++
  class CTxxx{
  public:
    CTxxx(){printf("CTxxx cst\n");}
    ~CTxxx(){printf("CTxxx dst\n");}

  };
  int main(){
    std::shared_ptr<CTxxx> sp_ct(new CTxxx);
    std::weak_ptr<CTxxx> wk_ct = sp_ct;
    std::weak_ptr<CTxxx> wka1;
    {
      std::cout << "wk_ct.expired()=" << wk_ct.expired() << std::endl;
      std::shared_ptr<CTxxx> tmpP = wk_ct.lock();
      if (tmpP) {
        std::cout << "tmpP usecount=" << tmpP.use_count() << std::endl;
      } else {
        std::cout << "tmpP invalid" << std::endl;
      }
      std::shared_ptr<CTxxx> a1(new CTxxx);
      wka1 = (a1);
    }
    std::cout << "wka1.expired()=" << wka1.expired() << std::endl;
      std::cout << "wka1.lock()=" << wka1.lock() << std::endl;
  
      std::shared_ptr<CTxxx> cpySp = wka1.lock();
      if (cpySp) std::cout << "cpySp is ok" << std::endl;
      else std::cout << "cpySp is destroyed" << std::endl;
      return 1;
  }
  ```

  * 解决循环引用

![](https://img-blog.csdnimg.cn/583726c18d114a8aafaa2bff06b96fe5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAT2NlYW5TdGFy55qE5a2m5Lmg56yU6K6w,size_14,color_FFFFFF,t_70,g_se,x_16)

  * A、B、C三个对象的数据结构中，A和C共享B的所有权，因此各持有一个指向B的std::shared_ptr
  * 假设有一个指针从B指回A（即上图中的红色箭头），则该指针的类型应为weak_ptr，而不能是裸指针或shared_ptr
    * 裸指针，当A被析构时，由于C仍指向B，所以B会被保留。但B中保存着指向A的空悬指针（野指针），而B却检测不出来，但解引用该指针时会产生未定义行为
    * shared_ptr时。由于A和B相互保存着指向对方的shared_ptr，此时会形成循环引用，从而阻止了A和B的析构。
    * weak_ptr，这可以避免循环引用。假设A被析构，那么B的回指指针会空悬，但B可以检测到这一点，同时由于该指针是weak_ptr，不会影响A的强引用计数，因此当shared_ptr不再指向A时，不会阻止A的析构

  ```c++
  #include <iostream>
  #include <memory>
  using namespace std;
  class A {
  public:
    std::weak_ptr<B> bptr; // 修改为weak_ptr
    ~A() {
      cout << "A is deleted" << endl;
    }
  };
  class B {
  public:
    std::shared_ptr<A> aptr;
    ~B() {
      cout << "B is deleted" << endl;
    }
  };
  int main()
  {
    {//设定一个作用域
      std::shared_ptr<A> ap(new A);
      std::shared_ptr<B> bp(new B);
      ap->bptr = bp;
      bp->aptr = ap;
    }
    cout<< "main leave" << endl; 
    return 0;
  }

  ```

  * 缓存对象
    * 对工厂函数loadWidget（基于唯一ID来创建一些指向**只读对象**的智能指针）
      * 只读对象需要频繁使用
      * 需要从文件或数据库中加载
      * 可以考虑将对象缓存。当不再使用时，则将该对象删除。
    * 由于除了工厂函数还有缓存管理，unique_ptr不合适
    * 当用户用完工厂函数的对象后，对象会析构，缓存条目悬空。
      * 考虑将工厂函数的返回值设定为shared_ptr类型
      * 缓存类型为weak_ptr类型
  * 线程安全的对象回调与析构 —— 弱回调
    * 如果对象还在，则调用其成员函数，否则忽略
    * 线程A和线程B访问一个共享的对象，如果线程A正在析构这个对象的时候，线程B又要调用该共享对象的成员方法，此时可能线程A已经把对象析构完了，线程B再去访问该对象，就会发生不可预期的错误。
  * 做法：在开启新线程时，传入共享对象的弱指针

  ```c++
  class Test
  {
  public:
    // 构造Test对象，_ptr指向一块int堆内存，初始值是20
    Test() :_ptr(new int(20)) 
    {
      cout << "Test()" << endl;
    }
    // 析构Test对象，释放_ptr指向的堆内存
    ~Test()
    {
      delete _ptr;
      _ptr = nullptr;
      cout << "~Test()" << endl;
    }
    // 该show会在另外一个线程中被执行
    void show()
    {
      cout << *_ptr << endl;
    }
  private:
    int *volatile _ptr;
  };
  void threadProc(weak_ptr<Test> pw) // 通过弱智能指针观察强智能指针
  {
    // 睡眠两秒
    std::this_thread::sleep_for(std::chrono::seconds(2));
    /* 
    如果想访问对象的方法，先通过pw的lock方法进行提升操作，把weak_ptr提升
    为shared_ptr强智能指针，提升过程中，是通过检测它所观察的强智能指针保存
    的Test对象的引用计数，来判定Test对象是否存活，ps如果为nullptr，说明Test对象
    已经析构，不能再访问；如果ps!=nullptr，则可以正常访问Test对象的方法。
    */
    shared_ptr<Test> ps = pw.lock();
    if (ps != nullptr)
    {
      ps->show();
    }
  }
  int main()
  {
    // 在堆上定义共享对象
    shared_ptr<Test> p(new Test);
    // 使用C++11的线程，开启一个新线程，并传入共享对象的弱智能指针
    std::thread t1(threadProc, weak_ptr<Test>(p));
    // 在main线程中析构Test共享对象
    // 等待子线程运行结束
    t1.join();

    return 0;
  }
  ```
  
  * 当将`t1.join()`换为`t1.detach()`时候，让`main`主线程结束，`p`智能指针析构，`Test`对象析构，此时`show()`不会被调用
    * threadProc方法中，`pw`提升到`ps`时，`lock`方法判定`Test`对象已经析构，提升失败



### unique_ptr

 拥有对持有对象的唯一所有权，两个`unique_ptr`不能同时指向同一个对象

 **特点**
  * 不能复制
  * 只能通过转移语义将所有权转移到另外一个


  ```c++
  std::unique_ptr<A> a1(new A());
  std::unique_ptr<A> a2 = a1;//编译报错，不允许复制
  std::unique_ptr<A> a3 = std::move(a1);//可以转移所有权，所有权转义后a1不再拥有任何指针
  ```


 **方法**
  * `get()` 获取其保存的原生指针
  * `bool()` 判断是否拥有指针
  * `release()` 释放所管理指针的所有权，返回原生指针。但并不销毁原生指针。
  * `reset()`释放并销毁原生指针。如果参数为一个新指针，将管理这个新指针

  ```c++
  std::unique_ptr<A> a1(new A());
  A *origin_a = a1.get();//尽量不要暴露原生指针
  std::unique_ptr<A> a2(a1.release());//常见用法，转义拥有权
  a2.reset(new A());//释放并销毁原有对象，持有一个新对象
  a2.reset();//释放并销毁原有对象，等同于下面的写法
  a2 = nullptr;//释放并销毁原有对象
  ```


## 模板元编程

  * 函数名相同，参数类型不同要重新写函数。模板出现就是提高了程序的复用性，提高效率
  * 当刚上手的时候肯定是根据具体的数据类型来组织代码。随着越来越熟，用一种广泛的表达去取代具体数据类型，在c++中就叫做模板编程。


 **类型**
  * 函数模板
  * 类模板

 **格式**
 `template <template T>`或`template <class T>`

 **底层实现**
  * 编译器将函数模板通过具体类型产生不同的函数
    * 对模板代码声明处进行编译
    * 在调用地方对替换后代码编译


 **模板和继承**
  * 使用目的
    * 模板用于生成一组类或函数，这些类和函数的实现是一样的
    * 继承是事物之间的联系，从父类到子类是从普遍到特殊，从共性到特性
  * 多态的不同
    * 模板是编译时多态
    * 继承是运行时多态
  * 复制内容
    * 模板是对代码的复制，编译完成后，会生成对应的函数或类
    * 继承是对数据的复制，复制虚表、数据


### 函数模板

 **类型**

  * 成员函数模板
  * 普通函数模板

 **调用方式**

  * 自动推导，隐式调用
    * `myswap(a, b)`
    * 参数类型和模板定义的一致才可以
    * 模板必须确定出T的类型
  * 显式调用
    * `myswap<int>(a, b)`

 **普通函数和模板函数**

 区别
  * 普通函数调用时可以发生自动类型转换（隐式类型转换）
  * 如果使用函数模板，自动类型推导的话，则不会发生隐式转换
  * 如果使用函数模板，显式指定类型，则可以发生隐式转换

 调用规则
  * 优先调用普通函数
  * 可以使用空模板参数来强制调用模板函数
  * 函数模板也可以重载
  * 如果函数模板可以产生更好的匹配，优先调用函数模板

### 类模板

 **调用方式**

  只有显式指定参数类型

 **普通类和模板类**

 成员函数
  * 普通类在编译时创建
  * 模板类在调用时创建
 
 类模板对象作函数参数
  * 指定传入类型，直接显示对象的数据类型

  ```c++
  void print(Person<string, int>& p);
  ```

  * 参数模板化，将对象中的参数变为模板进行传递
  ```c++
  template <class T1, class T2>
  void print(Person<T1, T2>& p);
  ```

  * 整个类模板化，将整个对象类型模板化进行传递

  ```c++
  template <class T>
  void print(T& t);
  ```


 **类模板与继承**

  * 当派生类继承基类的一个类模板时，子类在声明时，要指定出分类中的T类型

  ```c++
  template <class T>
  class father{
    T t;
  };

  // 子类在声明时，要指定出分类中的T类型
  class son : public father<int>{

  }
  ```

  * 如果还需要灵活，则子类需要变为类模板

  ```c++
  template <class T>
  class father{
    T t;
  };

  template <class T1, class T2>
  class son : public father <T2>{
    T1 obj;
  }
  ```

  * 类模板成员的类外实现

  ```c++
  // 构造函数类外实现
  template<class T1, class T2>
  Person<T1, T2>::Person(T1 name, T2 age){}

  // 成员函数类外实现
  template<class T1, class T2>
  void Person<T1, T2>::show(){}
  ```

 **文件要求**
  * 要求模板和实现在一个文件内
