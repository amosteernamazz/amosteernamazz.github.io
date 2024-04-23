---
layout: article
title: C++基础与关键字
key: 100032
tags: C++
category: blog
date: 2019-05-14 00:00:00 +08:00
mermaid: true
---


### 问题

***把异常完全封装在析构函数内部，决不让异常抛出函数之外***

***栈上的指针什么时候析构***

***.h .cpp .hpp 关系***

### 函数指针
 * 指向函数的指针变量。函数指针本身是一个指针变量，变量指向一个函数。
   * 有了这个指针，可以用这个指针变量调用函数
   * 除了调用外还可以做函数的参数


```c++
char * fun(char * p) {…}  // 指向char的指针
char * (*pf)(char * p);   // pf函数指针
pf = fun;                 // 函数指针指向函数
pf(p);                    // 调用
```

### bool、int、float、指针类型变量a与0的比较语句

  ```c++
  if(!a) or if(a)

  if (a ==0)

  if(a <= 0.000001 && a >=-0.000001)

  if(a != NULL ) or if(a == NULL)
  ```
<!--more-->


# **C++关键字**

## printf实现原理

函数参数通过压入堆栈的方式来传递参数
而栈是从内存高地址向低地址生长，因此最后压栈的在堆栈指针的上方，printf第一个被找到的参数就是字符指针。函数通过判断字符串控制参数的个数来判断参数个数与数据类型，进而算出需要的堆栈指针偏移量



## enum

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

## arr & arr[0]、&arr区别

 **print**
  * 都是首元素地址


 **+1 print**
  * `arr[0] + 1`和 `arr + 1`在数组内移动
  * `&arr + 1`按照数组单位移动


## char、char a[]、char* a、char* a[]、char** a

  | type                       | 含义                             |
  | -------------------------- | --------------------------------------- |
  | char a                   | 定义存储空间，存储char类型变量                                       |
  | char a[]               | 字符数组，每个元素都是char类型数据                                       |
  | char *a                | 字符串首地址                     |
  | char *a[]               | 表示char数组，数组元素为指针，指针指向char类型                                     |
  | char **a                     | 与char *a[]相同                                       |

## 一维数组名和二维数组名的区别

 **相同点**
  * 存储都是一维的
  * 一维数组名与二维数组名都指向数组的指针

 **不同点**
  * 二维数组名不能赋给二级指针
    * 二级指针：要求指向的是指针，而二维数组确定一维后指向数组
      * 想要获得 a[i] 中第 x 个元素，可以直接使用 `*(a+x)`
      * 想要获得 `b[i][j] `中第 x 行第 y 个元素，则需用 `*(*(b+x)+y)`
  * 一维数组+1跳过对应值，二维数组跳过行或列



## c++如何定义常量

 * 局部常量：栈
 * 全局常量：编译期不分配内存，放在符号表
 * 字面值常量： 字符串放在常量区

## static

### static与局部变量

 **特点**
  * 存储位置data区
  * 生命周期保持不变
  * 局部作用域退出时，数据仍然暂存data区


### static与全局变量

 **特点**
  * 加入static后，源程序的其他源文件不能再使用该变量（不使用static可以用extern扩展）


### static与函数

 **特点**
  * 跟全局变量相同，限制作用域，只能在该文件中使用（与全局变量用法也相同）


### static与类对象成员变量

 **特点**
  * 变量会变成类的全局变量，只能在类外初始化
    * 但如果加入const修饰，则可以在类内初始化


### static与类对象成员函数

 **特点**
  * 类只存在一份函数，所有对象共享，不含this指针，无需创建实例即可访问
    * 不可同时用const和static修饰对象的成员函数





## const

### const与变量

 **特点**
  * 限定不可更改


### const与指针

 **特点**
  * `int const * a`与`int * const a`


 **指向常量的指针**
  * `const int * a`
  * `int const * a`


 **指针常量**
  * `int * const a`


### const与函数
  * `const int& func(int& a)`：修饰返回值为const
  * `int& func(const int& a)`：修饰形参
  * `int& func(int& a) const{}`：const成员函数
    * 不允许修改类的成员的值


### const与类
  **const修饰类成员变量**
   * 在对象的声明周期内是常量，对整个类而言是可以改变的。
   * 不能在类内初始化const成员变量，在初始化列表中初始化。


  **const类对象成员函数**
   * 不允许修改类的成员的值


  **const对象**
   * 只能调用const函数


## static与const总结

**static的作用是表示该函数只作用在类型的静态变量上，与类的实例没有关系**

**const的作用是确保函数不能修改类的实例的状态**
static和const不可同时修饰成员函数



## 引用与指针
  ```c++
  int i = 5;
  // 引用
  int &a = i;
  a = 8;
  ```


### 引用的本质
 * `& = T * const a`
 * 本质是常量指针


### 引用与常量指针相同点
 * 都占用4/8字节
 * 都必须初始化


### 引用与常量指针不同点
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
  * 指针常量 -> 值传递。编译的时候：会将指针和变量存放符号表（变量名和对应地址）地址为指针变量的地址值。参数传递时候：会在开辟空间的时候形成实参的副本，因此对此参数的操作最终只是对局部变量的操作，并不会影响实参指针对应的值。如果想要使用，需要使用指向指针的指针或指针引用
  * 引用  ->引用传递。编译的时候：存放的地址是引用对象的地址。参数传递时候：是主调函数放进来的实参变量值，被调函数任何操作都会影响主调函数的实参变量

 **sizeof()的不同**
  * 指针常量 -> 指针的大小
  * 引用 -> 得到指向对象的大小

## const -> #define
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

## #define -> typedef

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



## this指针

 **作用**
  * 指向非静态成员函数所作用的对象

 **什么时候创建**
  * 调用非静态函数时才会使用的

 **delete this**
  * 为将被释放的内存调用一个或多个析构函数（因此不能在析构中调用delete this），类对象的内存空间被释放，之后不能涉及this指针，如操作数据成员，调用虚函数等


## volatile 和extern
 * voltile：不可优化（保证一定执行）、顺序性（不会进行乱序优化）、易变性（下一个语句不会直接使用上一个语句volatile变量的寄存器内容，选择从内存中重新读取）
 * extern：说明是在别处定义的，在此处引用，相对于#include方法，加速编译过程
   * 用于支持c或c++函数调用规范，当在c++程序中调用c的库，需要extern c的声明引用，主要因为c++和c编译完成后，目标代码的命名规则不同，用来解决名字匹配


## 强制类型转换
 **static_cast**
 派生->基类安全，反向不安全
  * 基本数据类型之间的转换
    * void*和其他类型指针之间的转换
    * 子类对象的指针转换成父类对象指针
  * 最好所有隐式转换都用static_cast代替

 **dynamic_cast**
  * 用于安全的向下转型
    * 转换成功会返回引用或者指针，失败返回null，否则会抛出一个`bad_cast`的异常类型

 **const_cast**
  * 用于**移除指针和引用的常量性**，但是不能改变原来常量的常量性
    * 指向常量的指针被转化成非常量指针
    * 常量引用被转换成非常量引用
    * 常量对象被转换成非常量对象

 **reinterpret_cast**
  **高危险性操作**
  `reinpreter_cast<type-id> (expression)`
  * 可以将任意类型指针转换为其他类型的指针。所以他的type-id必须是一个指针、引用、算术类型。
  * 能够在非相关的类型之间转换。它可以把一个指针转换成一个整数，也可以把一个整数转换成一个指针。

 **应用**
  * 一般不要使用dynamic_cast、reinterpret_cast


## fork、wait和exec函数

在早期unix系统中，当调⽤ fork 时，内核会把所有的内部数据结构复制⼀份，复制进程的⻚表项，然后把⽗进程的地址空间中的内容逐⻚的复制到⼦进程的地址空间中。但从内核⻆度来说，逐⻚的复制⽅式是⼗分耗时的。现代的 Unix 系统采取了更多的优化，例如 Linux，采⽤了写时复制的⽅法，⽽不是对⽗进程空间进程整体复制。

 * ⽗进程产⽣⼦进程使⽤ fork 拷⻉出来⼀个⽗进程的副本，此时只拷⻉了⽗进程的⻚表，两个进程都读同⼀块内存。
 * 当有进程写的时候使⽤写实拷⻉机制分配内存，exec 函数可以加载⼀个 elf ⽂件去替换⽗进程，从此⽗进程和⼦进程就可以运⾏不同的程序了。
 * fork 从⽗进程返回⼦进程的 pid，从⼦进程返回 0，调⽤了 wait 的⽗进程将会发⽣阻塞，直到有⼦进程状态改变，执⾏成功返回 0，错误返回 -1。
 * exec 执⾏成功则⼦进程从新的程序开始运⾏，⽆返回值，执⾏失败返回 -1。

```c++
int main(int argc, char *argv[])
{
    printf("hello world (pid:%d)\n", (int) getpid());
    
    // fork以后子进程pid=0，父进程pid=子进程
    int rc = fork();
    if (rc < 0) {
        // fork failed; exit
        fprintf(stderr, "fork failed\n");
        exit(1);
    } else if (rc == 0) {
        // child (new process)
        printf("hello, I am child (pid:%d)\n", (int) getpid());

        // 子进程程序执行为execvp的命令
        char *myargs[3];
        myargs[0] = strdup("wc");   // program: "wc" (word count)
        myargs[1] = strdup("exec.c"); // argument: file to count
        myargs[2] = NULL;           // marks end of array
        execvp(myargs[0], myargs);  // runs word count

        // 子进程已经执行了wc程序，因此不会返回此处执行
        printf("this shouldn't print out");
    } else {
        // 父进程等待子进程结束，如果为多个的话等待其中一个结束
        // parent goes down this path (original process)
        int wc = wait(NULL);
        printf("hello, I am parent of %d (wc:%d) (pid:%d)\n",
	       rc, wc, (int) getpid());
    }
    return 0;
}
```


## 回调函数

当发⽣某种事件时，系统或其他函数将会⾃动调⽤你定义的⼀段函数，相当于**中断处理函数**，当系统在符合条件的时候自动调用，其通过函数指针调用的函数，如果将某函数的指针作为参数给另外一个函数，当另外的这个函数在满足一定条件后通过函数指针完成函数的调用，那么称这个被调用的函数为回调函数

步骤：声明，定义，设置触发条件



## lambda 表达式

提供了一种**匿名函数**的特性，可以编写内嵌的匿名函数，用于替换独立函数，而且更可读
本质上来讲， lambda 表达式只是**一种语法糖**，因为所有其能完成的⼯作都可以⽤其它稍微复杂的代码来实现。



从[]开始，结束于{}，{}内定义的是lambda表达式体

```c++
auto basicLambda = [] { cout << "Hello, world!" << endl; };
basicLambda(); 
```

带返回值类型的
```c++
auto add[](int a, int b) -> int{return a+b;};

auto multiply = [](int a, int b)-> {return a*b;};

int sum = add(2,3);
int product = multiply(2, 5);
```
[]闭包：
实现原理是每次定义lambda表达式后，都会自动生成匿名类，称为**闭包类型**。运行时候，lambda表达式会返回一个匿名闭包实例，实际是右值。其可以通过**传值或引用**的方式捕捉封装作用域的变量


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
[]：默认不捕获任何变量
[=]：默认以值捕获所有变量；
[&]：默认以引⽤捕获所有变量；
[x]：仅以值捕获x，其它变量不捕获；
[&x]：仅以引⽤捕获x，其它变量不捕获；
[=, &x]：默认以值捕获所有变量，但是x是例外，通过引⽤捕获；
[&, x]：默认以引⽤捕获所有变量，但是x是例外，通过值捕获；
[this]：通过引⽤捕获当前对象（其实是复制指针）；
[*this]：通过传值⽅式捕获当前对象

应用于函数的参数，实现回调

```c++
int val = 3;
vector<int> v{1,8,3,4,7,3};
int count = std::count_if(v.begin(), v.end(), [val](int x) {return x >3;});
```


## 右值引用

```c++
Person get(){
  Person p;
  return p;
}
Person p = get();
```
上述获得并初始化过程涉及3次构造、2次析构，因此为了方便传递，引入右值引用，类似move语义，从右值直接拿数据初始化并修改左值，不需要重新构造再析构


```c++
class Person{
public:
 Person(Person&& rhs){...}
 ...
};
```

## 泛化常数 constexpr

```c++
// 告诉编译器这是编译期常量
constexpr int N = 5;
int arr[N];

// 也可以为常量表达式
constexpr int getFive(){ return 5; }
int arr[getFive() + 1];
```


## 初始化列表 std::initializer_list

```c++
class A{
public:
  A(std::initializer_list<int> list);
};
A a = {1,2,3};
// 只是初始化的时候长度可以变化，只能静态构造
A b = {1,2};
```




## 范围for循环
 

## 构造函数委托

可以在构造函数中调用同一个类的其他构造函数

## final 和override

final用来禁止虚函数被重写/禁止类被继承
override用来显示地重写虚函数，这样可以提供更多有用错误和警告

## default 和delete

用于显式指定和禁止某些行为
```c++
struct classA {
 classA() = defauult; // 声明⼀个⾃动⽣成的函数
 classA(T value);
 void *operator new(size_t) = delete; // 禁⽌⽣成new运算符
};
```

## assert断言

一般用于debug程序的逻辑，不用于release版本
 * assert宏
   * `assert(x >0)`
 * #error方法

```c++
#if defined(DEBUG)
    // 在调试模式下执行某些操作
#else
    #error "DEBUG macro is not defined. Please define DEBUG for release builds."
#endif
```


 * 模板的assert

```c++
template< class T >
struct Check {
 static_assert( sizeof(int) <= sizeof(T), "T is not big enough!" ) ;
} ;

```


## 正则表达式

```c++
const char *reg_esp = "[ ,.\\t\\n;:]";
std::regex rgx(reg_esp) ;
std::cmatch match ; 
const char *target = "Polytechnic University of Turin " ;
if( regex_search( target, match, rgx ) ) {
 const size_t n = match.size();
 for( size_t a = 0 ; a < n ; a++ ) {
 string str( match[a].first, match[a].second ) ;
 cout << str << "\n" ;
 }
}
```

## 元组

```c++
typedef std::tuple< int , double, string > tuple_1 t1;
typedef std::tuple< char, short , const char * > tuple_2 t2 ('X', 2,
"Hola!");
t1 = t2 ; // 隐式类型转换

```

## 哈希表

 * map , multimap , set , multiset 使⽤红⿊树实现， 插⼊和查询都是 O(lgn) 的复杂度
 * C++11 为这四种模板类提供了（底层哈希实现）以达到 O(1) 的复杂度
 * unordered_map：无序，搜索效率高，额外空间大
 * unordered_multimap：速度快，无序，空间大，kv可以重复
 * unordered_set：值单个，无序，速度快，空间大
 * unordered_multiset：值可以有多个，无序，速度快，空间大

## 

