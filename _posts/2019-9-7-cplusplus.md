---
layout: article
title: C++运行时特征
key: 100002
tags: C++ 多态
category: blog
date: 2019-09-07 00:00:00 +08:00
mermaid: true
---

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


## 元组

```c++
typedef std::tuple< int , double, string > tuple_1 t1;
typedef std::tuple< char, short , const char * > tuple_2 t2 ('X', 2,
"Hola!");
t1 = t2 ; // 隐式类型转换

```
