---
layout: article
title: 进程
key: 100013
tags: 操作系统 进程
category: blog
date: 2020-11-03 00:00:00 +08:00
mermaid: true
---

***fork复制内部，为什么fork返回0？***


## 大端模式和小端模式
 * ⼤端模式：是指数据的⾼字节保存在内存的低地址中，⽽数据的低字节保存在内存的⾼地址端。
 * ⼩端模式，是指数据的⾼字节保存在内存的⾼地址中，低位字节保存在在内存的低地址端。
 * 判断方法，直接读取十六进制的值 or 使用共同体来判断


```c++

#include <stdio.h>
int main() {
 
 union {
  int a; //4 bytes
  char b; //1 byte
 } data;


 data.a = 1; //占4 bytes，⼗六进制可表示为 0x 00 00 00 01
 
 //b因为是char型只占1Byte，a因为是int型占4Byte
 //所以，在联合体data所占内存中，b所占内存等于a所占内存的低地址部分
 if(1 == data.b) {
 //⾛到这⾥意味着说明a的低字节，被取给到了b
 //即a的低字节存在了联合体所占内存的(起始)低地址，符合⼩端模式特征
 printf("Little_Endian\n");
 } else {
 printf("Big_Endian\n");
 }
 return 0;

}```

## 手写实现智能指针

```c++
template<typename T>
class SharedPtr {
private:
 size_t* m_count_;
 T* m_ptr_;
public:
 // 无参构造
 SharedPtr(): m_ptr_(nullptr),m_count_(new size_t) {}
 // 原生指针
 SharedPtr(T* ptr): m_ptr_(ptr),m_count_(new size_t) { m_count_ = 1;}
 //析构函数
 ~SharedPtr() {
 -- (*m_count_);
 if (*m_count_ == 0) {
 delete m_ptr_;
 delete m_count_;
 m_ptr_ = nullptr;
 m_count_ = nullptr;
 }
 }
 //拷⻉构造函数
 SharedPtr(const SharedPtr& ptr) {
 m_count_ = ptr.m_count_;
 m_ptr_ = ptr.m_ptr_;
 ++(*m_count_);
 }
 //拷⻉赋值运算
 void operator=(const SharedPtr& ptr) { SharedPtr(std::move(ptr)); }
 //移动构造函数
 SharedPtr(SharedPtr&& ptr) : m_ptr_(ptr.m_ptr_),
m_count_(ptr.m_count_) { ++(*m_count_); }
//移动赋值运算
 void operator=(SharedPtr&& ptr) { SharedPtr(std::move(ptr)); }
 //解引⽤
 T& operator*() { return *m_ptr_; }
 //箭头运算
 T* operator->() { return m_ptr_; }
 //᯿载bool操作符
 operator bool() {return m_ptr_ == nullptr;}
 T* get() { return m_ptr_;}
 size_t use_count() { return *m_count_;}
 bool unique() { return *m_count_ == 1; }
 void swap(SharedPtr& ptr) { std::swap(*this, ptr); }
};
```



# 进程
 * 每一个进程都会有属于自己独立的内存空间，磁盘空间，I\O设备等等。
   * 每个进程都有自己独立的堆栈，即局部变量对于线程来说是私有的。创建多进程代价大

## 操作系统的进程管理
 * 进程间通信
 * 进程同步
 * 死锁
 * 进程调度

## 进程间通信

 * 进程之间要交换数据必须通过内核，内核是可以共享的。在内核中开辟一块缓冲区，进程1把数据从用户空间拷到内核缓冲区，进程2再从内核缓冲区把数据读走，内核提供的这种机制称为进程间通信

### 管道

  进程以先进先出的方式传送数据，是半双工的，意味着数据只能往一个方向流动。因此当双方通信时，必须建立两个管道。


  **实质**
   * 内核中创建一个缓冲区，管道一端的进程进入管道写数据，另一端的进程进入管道读取数据
     * PIPE：相关联的进程
     * FIFO：任何进程都可以根据管道文件名打开
       * 周期随着进程的创建而创建，销毁而销毁


  **缺点**
   * 管道通信效率低，不适合频繁数据交换
     * 有内核态到用户态之间的数据传递

<!--more-->

### 消息队列

  保存在内核中的链表，由一个个独立的数据块组成，消息的接收方和发送方要约定具体的消息类型。与收发邮件类似。两个进程你来我往的进行沟通


  **实质**
   * 当进程从消息队列中读取了相关数据块，则内核会将该数据块删除
   * 不一定按照先进先出的方式读取，按照消息类型进行兑取
   * 生命周期与内核相关，如果不显示的删除消息队列，则消息队列会一直存在


  **缺点**
   * 不能实现实时通信
   * 数据块是有大小限制
   * 消息队列通信过程中，存在用户态与内核态之间的数据拷贝开销
     * 进程写入数据到内核中的消息队列时，会发生从用户态拷贝数据到内核态的过程，同理另一进程读取内核中的消息数据时，会发生从内核态拷贝数据到用户态的过程

### 共享内存

  解决用户态和内核态之间频繁发生拷贝过程，数据不需要在不同进程之间进行复制


  **实质**
   * 拿出一块虚拟地址空间，映射到相同的物理内存中
     * 一个进程写入数据后另一个进程可以立刻看到，不用进行拷贝。效率很高。

### 信号

  异常状态下用信号通知进程
  * 可以在任何时刻给进程发送信号，进程间通信或操作是一种异步传输机制
  
  **信号处理的方式**

   * 系统定义的信号函数
     * **SIGINT：**程序终止信号。程序运行过程中，按`Ctrl+C`键将产生该信号
     * **SIGQUIT：**程序退出信号。程序运行过程中，按`Ctrl+\\`键将产生该信号
     * **SIGALRM：**定时器信号
     * **SIGTERM：**结束进程信号。shell下执行`kill 进程pid`发送该信号
   * 捕捉信号
     * 用户可以给信号定义信号处理函数，表示收到信号后该进程该怎么做
   * 忽略信号
     * 不希望处理某些信号的时候，就可以忽略该信号，不做任何处理
     * `SIGKILL` 和 `SEGSTOP`是应用进程无法捕捉和忽略的。它们用于在任何时候中断或结束某一进程。

### unix域间套接字

  * 一开始发展的unix域套接字是为了网络通信设计，后来发展用于进程通信的机制。
    * 不需要网络协议栈，不需要打包拆包、计算校验和、维护序号和应答等，只是将应用层数据从一个进程拷贝到另一个进程。
  * UNIX域套接字与TCP套接字相比较，在同一台主机的传输速度前者是后者的两倍
    * PIC机制是可靠通信
    * UNIX Domain Socket提供面向流和面向数据包两种API接口，类似于TCP和UDP
      * 但是面向消息的UNIX Domain Socket也是可靠的，消息既不会丢失也不会顺序错乱。

### 信号量

  * 与共享内存混合使用
    * 如果多个进程同时对一个共享内存进行操作，会产生冲突造成不可预计的后果
    * 为了不冲突，共享内存在一个时间段只能有一个进程访问，就出现了信号量


  **本质**
   * 信号量是一个计数器，用于实现进程间的互斥和同步

  **原理**

  **P操作**
  * 将信号量-1，之后如果信号量 <0，表明资源被占用，进程需要阻塞等待
    * 如果之后>=0，表明进程可以正常执行
  
  **V操作**
  * 与P操作正好相反


  **过程**
  * 进程 A 在访问共享内存前，先执行 P 操作，由于信号量的初始值为 1，故在进程 A 执行 P 操作后信号量变为 0，表示共享资源可用，于是进程 A 就可以访问共享内存。
  * 若此时，进程 B 也想访问共享内存，执行了 P 操作，结果信号量变为了 -1，这就意味着临界资源已被占用，因此进程 B 被阻塞。
  * 直到进程 A 访问完共享内存，才会执行 V 操作，使得信号量恢复为 0，接着就会唤醒阻塞中的线程 B，使得进程 B 可以访问共享内存，最后完成共享内存的访问后，执行 V 操作，使信号量恢复到初始值 1。


  **信号量与互斥量**
   * 互斥量：线程的互斥
     * 某一资源同时只允许一个访问者对其进行访问，具有唯一性和排它性。但互斥无法限制访问者对资源的访问顺序，即访问是无序的。
   * 信号量：线程的同步
     * 在互斥的基础上（大多数情况），通过其它机制实现访问者对资源的有序访问。
     * 互斥量的加锁和解锁必须由同一线程分别对应使用，信号量可以由一个线程释放，另一个线程得到。

## 进程同步
  * 进程同步主要考虑多个进程对于临界区资源的抢夺和多个线程之间的关系
    * 资源共享情况
    * 进程间协作的缘故
      * 当A向B提供数据，当A缓存没数据的时候B就阻塞，A缓存满时A就阻塞。

### 临界区

  **定义**

  是指一个访问公共资源（共用设备或是共用存储器）的程序片段

  **临界区访问特点**
    * 进程之间采取互斥方式，实现对这种资源的共享

### 进程同步的原则
  * 空闲让进
    * 当无进程处于临界区时，表明临界资源处于空闲状态，应允许一个请求进入临界区的进程立即进入自己的临界区，以有效的利用临界资源。
  * 忙则等待
    * 当已有进程进入临界区时，表明临界资源正在被访问，因而其他视图进入临界区的进程必须等待，以保证对临界资源的互斥访问。
  * 有限等待
    * 对要求访问临界资源的进程，应保证在有限时限内能进入自己的临界区，以免陷入死等状态。
  * 让权等待
    * 当进程不能进入自己的临界区时，应立即释放处理机，以免进程陷入忙等状态。

### 实现方法
  * 提高临界区代码执行中断的优先级
    * 提高临界区中断优先级方法就可以屏蔽了其它中断，保证了临界段的执行不被打断，从而实现了互斥。
  * 自旋锁
    * 最多只有一个执行者，当又有资源调用者，该调用会始终检查是否释放了锁（处于自旋状态）直到释放锁
    * 实现使用汇编实现
  * 信号量机制
    * 与进程通信信号量机制相同

### 进程同步场景

#### 生产者——消费者场景


  **目标**
  * 两个进程（生产者和消费者）**共享一个**公共的固定大小的**缓冲区**。
  * 生产者将数据放入缓冲区，消费者从缓冲区中取数据。也可以扩展成m个生产者和n个消费者。
  * 当**缓冲区空**的时候，消费者因为取不到数据就会睡眠，直到缓冲区有数据才会被唤醒。
  * 当**缓冲区满**的时候，生产者无法继续往缓冲区中添加数据，就会睡眠，当缓冲区不满的时候再唤醒。


  **问题**
  * 为了时刻监视缓冲区大小，需要有一个变量count来映射。但是这个变量就是映射的共享内存，生产者消费者都可以修改这个变量。由于这里面对count没有加以限制会出现竞争
    * 当缓冲区为空时，count=0，消费者读取到count为0
    * 这个时候如果CPU执行权限突然转移给了生产者。生产者发现count=0，就会马上生产一个数据放到缓冲区中，此时count=1，接着会发送一个wakeup信号给消费者，因为由于之前count=0，生产者以为消费者进入了阻塞状态。
    * 事实上消费者还没有进入阻塞状态，生产者的这个wakeup信号会丢失。接着CPU执行权限有转移到消费者这里，消费者查看自己的进程表项中存储的信息发现count=0然后进入阻塞，永远不去取数据。
    * 生产者迟早会把缓冲区填满，然后生产者也会进入阻塞，然后两个进程都在阻塞下去，出现了问题。


  **涉及两种关系**
    * 生产者与消费者之间的同步关系（缓冲区满，先等消费者线行动，缓冲区空时，等生产者先行动）
    * 生产者和消费者之间的互斥关系（对缓冲区的操作必须与其他进程互斥才行。不然很容易死锁。）

  **解决方案**


  ```c
    pthread_mutex_t mutex;  
    pthread_cond_t producter;  
    pthread_cond_t consumer;
    count = pool.size()

    producter(){
      while(1){
        pthread_mutex_lock(&mutex);
        pool++;
        while(pool == count){
          pthread_cond_wait(producter, &mutex);
        }
        pthread_mutex_signal(consumer, &mutex);
        pthread_mutex_unlock(&mutex);
      }
    }
    consumer(){
      while(1){
        pthread_mutex_lock(&mutex);
        pool++;
        while(pool == 0){
          pthread_cond_wait(consumer, &mutex);
        }
        pthread_mutex_signal(producter, &mutex);
        pthread_mutex_unlock(&mutex);
      }
    }
  ```

#### 哲学家进餐场景

  **目标**

  * 整个模型共5个进程，关系为互斥
  * 临界区资源与进程的关系如下图所示

![](https://img-blog.csdnimg.cn/20200315203417226.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0JsYWNrb3V0ZHJhZ29u,size_16,color_FFFFFF,t_70)
   
  **进程死锁问题**

  * 五位哲学家同时拿起左侧筷子，没有人拿右侧筷子
  * 拿起左侧叉子然后拿右侧，发现没有则放下，然后拿右侧，后来又发现没有左侧筷子
  * 拿起左侧筷子，等待一定的时间，有右侧则拿，否则放下左侧，但对安全系统来说不可靠


  **一种解决方法**

  * 同时拿到两个筷子
  * 对每个哲学家的动作制定规则，避免饥饿或者死锁。


  互斥信号量组用于进程间互斥访问`chopstick=[1,1,1,1,1]`

    ```c++
    semephore chopstick[5] = {1,1,1,1,1};
    Pi(){
    do{
      P(chopstick[i]);
      P(chopstick[i+1] % 5);
      eat;
      V(chopstick[i]);
      V(chopstick[i+1] % 5);
      think;
    }while(1);
    }
    ```

  * 问题：同时就餐拿起左边筷子产生死锁

  **第二种改进方法**

  * 对进程数量限制
    * 至多允许四个哲学家同时进餐
    * 仅当一个哲学家左右两边的筷子都可用时才允许他抓起筷子

  ```c++
  semephore chopstick[5] = {1,1,1,1,1};
  semephore mutex = l;
  Pi(){
  do{
    P(mutex);
    P(chopstick[i]);
    P(chopstick[i+1] % 5);
    V(mutex);
    eat;
    V(chopstick[i]);
    V(chopstick[i+1] % 5);
    think;
  }while(1);
  }
  ```


#### 读者——写者场景

  **目的**
  用于处理文件等类型的多读写操作


  **特点**
  * 允许多个读者可以同时对文件执行读操作
  * 只允许一个写者往文件中写信息
  * 任一写者在完成写操作之前不允许其他读者或写者工作
  * 写者执行写操作前，应让已有的读者和写者全部退出

  **三个关系**

    * 读者和写者是互斥关系
    * 写者和写者是互斥关系
    * 读者和读者不存在互斥关系

  **分析**
    * 写者 -> 和任何进程互斥，使用互斥信号量
    * 读者 -> 实现和写着的互斥，实现和其他读者的同步
    * 一个计数器，判断当前有多少读者在读文件
      * 当有读者的时候不能写入文件，当没有读者的时候写者才会写入文件。
      * 同时计数器也是公共内存，对计数器的访问也应该是互斥的

  **实现**

    ```c
    int count = 0;
    semaphore mutex = 1;
    semaphore rw = 1;

    write(){
      while(1){
        P(rw);
        Writing;
        V(rw);
      }
    }
    reader(){
      while(1){
        P(mutex);
        if(count == 0){
          P(rw);
        }
        count++;
        V(mutex);
        reading;
        P(mutex);
        count--;
        if(count ==0){
          V(rw);
        }
        V(mutex);
      }
    }
    ```


### 进程同步中虚拟内存

**目的**
 * 解决如何不把程序内存全部装入的问题


**思想**
 * 每个程序都拥有自己的**地址空间**，这些空间被分割为**页面**，每个页面都是连续地址范围，页面会**映射到物理内存**
 * 只有当程序引用到物理内存的地址空间时，才执行映射，将不包含在物理内存的地址空间装入物理内存
 * 实际上，它通常是被分隔成多个物理内存碎片，还有部分暂时存储在外部磁盘存储器上，在需要时进行数据交换。
   * 与没有使用虚拟内存技术的系统相比，使用这种技术的系统使得大型程序的编写变得更容易，对真正的物理内存（例如RAM）的使用也更有效率


#### 过程

  虚拟地址被分成虚拟页号+偏移量
  虚拟页号是一个页表的索引，有页面号可以找到对应的页框号，然后将该页框号放到16位地址的前4位，替换掉虚拟页号，就形成了送往内存的地址。
  
 **分页**
  * 虚拟内存首先送往内存管理单元MMU（集成在CPU芯片），然后MMU将虚拟内存映射为物理内存地址
    * 当程序访问地址，其实是访问虚拟地址（页），然后将虚拟地址送到MMU，然后MMU得到物理内存地址（页框），将内存地址送到总线
    * 当访问未映射页面（虚拟地址没有映射到页框）产生缺页中断，随后找到一个很少使用的页框，将页框写入磁盘，然后把需要访问的页面读到页框中，修改映射关系，重新启动引起中断的指令
    * 硬件实现
      * 将地址二进制16位地址分解成4位的页号+12位的偏移量
        * 4位页号表示16个页面，是页面的索引
        * 12位的位偏移可以为一页内的全部4096个字节编址


![](https://s2.loli.net/2022/01/09/1NDpgGZf6xSyVjA.png)
![](https://s2.loli.net/2022/01/09/Tiycz1fOpubvhZk.png)

 **分页中的MMU实现**
  * 16位地址分解成4位的页号+12位的偏移量
  * 4位页号表示16个页面，是页面的索引
  * 12位的位偏移可以为一页内的全部4096个字节编址

![](https://s2.loli.net/2022/01/09/kZnBWuoVwvmclsF.png)


 **分表**

![](https://s2.loli.net/2022/01/09/JTyb39umxQ6Zhtg.png)

![](https://s2.loli.net/2022/08/14/bxrp8vDNqh4TfyM.png)

**页表结构**
  * 页框号
    * 页到框映射
  * 在/不在位
    * 此位上的值是 1，那么页表项是有效的并且能够被使用
    * 此值是 0 的话，则表示该页表项对应的虚拟页面不在内存中，访问该页面会引起一个缺页异常(page fault)
  * 保护位
    * 最简单的表示形式是这个域只有一位，0 表示可读可写，1 表示的是只读
  * 修改位
    * 当一个页面被写入时，硬件会自动的设置修改位
    * 如果一个页面已经被修改过（即它是 脏 的），则必须把它写回磁盘。
    * 如果一个页面没有被修改过（即它是 干净的），那么重新分配时这个页框会被直接丢弃，因为磁盘上的副本仍然是有效的。
  * 访问位
    * 帮助操作系统在发生缺页中断时选择要淘汰的页
  * 高速缓存禁止位
    * 通过这一位可以禁用高速缓存
    * 具有独立的 I/O 空间而不是用内存映射 I/O 的机器来说，并不需要这一位


**采用虚拟内存带来的问题**
 * 映射必须非常快
   * 每次访问内存都需要进行虚拟地址到物理地址的映射
   * 所有的指令最终都必须来自内存，并且很多指令也会访问内存中的操作数
     * 因此每条指令进行多次页表访问是必要的
   * 避免映射成为主要的瓶颈
 * 虚拟地址空间很大，页表也会很大
   * 存储开销就很大


**解决虚拟内存速度问题**

 为解决多次内存访问，在CPU与CPU缓存之间建立TLB**实质是MMU硬件上的内存缓存**（保存虚拟页号、修改位、保护码、对应的物理页框号）

  * 过程
    * **硬件**首先检查虚拟页号与 TLB 中所有表项进行并行匹配，**判断虚拟页是否在 TLB 中**
    * 如果找到了**有效匹配**项，并且要进行的访问操作**没有违反保护位**的话，则将页框号直接**从 TLB 中取出**而不用再直接访问页表
    * 如果虚拟页在 TLB 中但是**违反了保护位的权限**的话（比如只允许读但是是一个写指令），则会**生成一个保护错误**(protection fault) 返回
    * 如果 MMU 检测到**没有有效的匹配项**，就会进行正常的**页表查找**，然后从TLB中逐出一个表项(修改位复制到内存中页表项)然后把从页表中找到的项**放在 TLB** 中(所有的值都来自于内存)

  * TLB中的各种情况
    * 处理与TLB错误通过硬件完成，**页面不在内存**中时，会发生操作系统的陷入，有的也通过软件处理
      * TLB 条目由操作系统显示**加载**。当发生 **TLB 访问丢失**时，生成一个 TLB 失效并将问题交给操作系统解决，操作系统必须找到该页，把它从 **TLB 中移除**（移除页表中的一项），然后把**新找到的页放在 TLB** 中，最后再**执行先前出错的指令**，不在 TLB 中，这将在处理过程中导致其他 TLB 错误。
        * **改善**方法是可以在内存中的固定位置维护一个大的 TLB 表项的**高速缓存**来减少 TLB 失效，首先检查软件的高速缓存，能有效的减少 TLB 失效问题。
    * TLB除了软失效（在内存不在TLB，会在页表中查找映射进行页表遍历）还有**硬失效**（需要产生磁盘I/O）
    * 实际情况更加复杂，可能出现**都不是这两种情况**
      * 如果**页表遍历没有找到**所需要的页
        * 页面在内存中，却没有记录在进程的页表中
          * 这种情况可能是由其他进程从磁盘掉入内存，这种情况只需要把页正确映射就可以了，而**不需要在从硬盘调入**，这是一种软失效，称为 次要缺页错误
          * 如果需要**从硬盘直接调入页面**：严重缺页错误
      * 程序可能访问了一个**非法地址**，根本无需向 TLB 中增加映射。
      * 操作系统**报告 段错误** 来终止程序。


**解决虚拟内存内存占用太大问题**
  * 多级页表
    * 目的
      * 避免把全部页表一直保存在内存中。 
    * 页表存储位置


    <img src="https://s2.loli.net/2022/08/14/eBafURrZNoKQGWT.jpg" alt="img" style="float: left;" />

    
      * 每个进程都有对应的页表
      * Linux中为每一个进程维护了一个tast_struct结构体（进程描述符PCB）
      * 其中tast_struct->mm_struct结构体成员用来保存该进程页表
      * 当切换的时候，会将页表地址写入控制寄存器（保存页目录表的物理内存地址）
     * 原理
       * 绝大部分程序仅仅使用了几个页，只需要几个页的映射就可以了
       * 为了避免内存浪费，计算机系统开发人员想出了一个方案，多级页表
     * 
   * 倒排页表
     * 建立实际内存到虚拟页面的表
     * 问题
       * 从虚拟地址到物理地址的转换会变得很困难
       * 必须搜索整个倒排表来查找某个表项
       * 搜索必须对每一个内存访问操作都执行一次，而不是在发生缺页中断时执行
       * 解决
         * 使用 TLB，
         * 当发生 TLB 失效时，需要用软件搜索整个倒排页表。
         * 一个可行的方式是建立一个散列表，用虚拟地址来散列。当前所有内存中的具有相同散列值的虚拟页面被链接在一起。
         * 如果散列表中的槽数与机器中物理页面数一样多，那么散列表的冲突链的长度将会是 1 个表项的长度，这将会大大提高映射速度。
         * 一旦页框被找到，新的（虚拟页号，物理页框号）就会被装在到 TLB 中。
         <img src="https://s2.loli.net/2022/08/14/TujXUp9YxwOdDGe.png" alt="img" style="zoom: 47%;float:left" />


### 进程中的共享内存

 **优点**
  * 最有用的进程间通信方式，也是最快的IPC形式
   * 同一块物理内存被映射到进程A、B各自的进程地址空间。进程A可以即时看到进程B对共享内存中数据的更新，反之亦然。
   * 由于多个进程共享同一块内存区域，必然需要某种同步机制，互斥锁和信号量都可以。
  * 效率高
   * 管道和消息队列等通信方式，则需要在内核和用户空间进行四次的数据拷贝(用户空间buf到内核，内核把数据拷贝到内存，内存拷贝到内核，内核到用户空间)
   * 共享内存则只拷贝两次数据(一次从输入文件到共享内存区，另一次从共享内存区到输出文件)
   * 并不总是读写少量数据后就解除映射，共享内存中的内容往往是在解除映射时才写回文件的。因此，采用共享内存的通信方式效率是非常高的。


 **最大限制**
  * 单个：32M
  * 最大个数：4096
  * 系统中共享内存页总数默认值：2097152*4096=8GB
 共享内存不保证同步，可以使用信号量来保证共享内存同步


 **与内存映射的区别**
  * 共享内存可以直接创建，内存映射需要磁盘文件
  * 共享内存效率更高
  * 内存
    * 共享内存：所有的进程操作的是同一块共享内存
    * 内存映射：每个进程在自己的虚拟地址空间中有一个独立的内存
  * 数据安全
    * 共享内存还存在，内存映射区消失
  * 运行进程的电脑死机
    * 在共享内存中的数据会消失
    * 内存映射区的数据，由于磁盘文件中的数据还在，所以内存映射区的数据还存在
  * 生命周期
    * 内存映射区：进程退出，内存映射区销毁
    * 共享内存：进程退出，共享内存还在
      * 标记删除（所有的关联的进程数为0），关机，或进程退出，会自动和共享内存进行取消关联。






### 进程死锁

**原因**
 * 资源分配不当
 * 进程运行的顺序不合理


**必要条件**
 * 互斥
   * 某个资源只允许一个进程访问，如果已经有进程访问该资源，则其他进程就不能访问，直到该进程访问结束
 * 占有的同时等待
   * 个进程占有其他资源的同时，还有资源未得到，需要其他进程释放该资源。
 * 不可抢占
   * 别的进程已经占有某资源，自己不能去抢。
 * 循环等待
   * 存在一个循环，每个进程都需要下一个进程的资源。


**破坏方法**
 * 破坏“占有且等待条件”
   * 所有进程在开始运行之前，一次性申请到所有所需要的资源
   * 进程用完的资源释放掉，然后再去请求新的资源，提高利用率

 * 破坏“不可抢占”条件
   * 当进程提出在得到一些资源时候不被满足的情况下，必须释放自己已经保存的资源

 * 破坏“循环等待”
   * 实现资源有序分配策略，所有进程申请资源必须按照顺序执行

 * 银行家算法
   * 进程提出资源请求且系统的资源能够满足该请求时，系统将判断满足此次资源请求后系统状态是否安全
     * 安全状态是非死锁状态，而不安全状态并不一定是死锁状态。
     * 即系统处于安全状态一定可以避免死锁，而系统处于不安全状态则仅仅可能进入死锁状态。
   * 前提条件
     * 要求进程预先提出自己的最大资源请求，并假设系统拥有固定的资源总量。


### 进程阻塞

**原因**
 * 正在运行的进程由于提出系统服务请求（如I/O操作），但因为某种原因
 * 未得到操作系统的立即响应
   * CPU 资源有限
 * 需要从其他合作进程获得的数据尚未到达等原因
   * 进程时常需要等待外部事件的发生，例如 I/O 事件、定时器事件等


**操作系统处理**
 * 把进程分为“运行”和“等待”等几种状态
 * 如果进程在时间片结束前阻塞或结束，则CPU当即进行切换
 * 进程有一个对象执行某个方法将当前进程阻塞了，内核会将进程从工作队列中移除，同时创建等待队列，并新建一个引用指向进程。
 * 进程被排在了工作队列之外，不受系统调度了，挂起。
   * 这也提现了阻塞和挂起的关系。
   * 阻塞是人为安排的，让你程序走到这里阻塞。
   * 而阻塞的实现方式是系统将进程挂起。
 * 当这个对象受到某事件触发后，操作系统将该对象等待队列上的进程重新放回到工作队列上就绪，等待时间片轮转到该进程。


**资源消耗**

 * 不消耗CPU资源
   * 得不到CPU，不能消耗资源
 * 消耗系统资源（内存、磁盘I/O）




### CPU调度进程算法

**类型**
 * 抢占式调度
   * 程序正在运行时可以被打断，把CPU让给其他进程
 * 非抢占式调度
   * 一个进程正在运行，当进程完成或者阻塞的时候把CPU让出来


**调度算法**
 * 先来先服务调度算法
   * 从就绪队列选择最先进入队列的进程，然后一直运行，直到进程退出或被阻塞，才会继续从队列中选择第一个进程接着运行。
   
   缺点
    * 不利于短作业。


   适用对象
    * 适用于 CPU 繁忙型作业的系统，而不适用于 I/O 繁忙型作业的系统
 * 最短作业优先调度算法
   * 优先选择运行时间最短的进程来运行，这有助于提高系统的吞吐量。

  缺点
   * 对长作业不利
 * 高响应比优先调度算法
   * 每次进行进程调度时，先计算「响应比优先级」，然后把「响应比优先级」最高的进程投入运行
   * 优先权=（等待时间+服务时间）/ 服务时间

 * 时间片轮转调度算法
   * 每个进程被分配一个时间段，允许该进程在该时间段中运行
   * 如果时间片用完，进程还在运行，那么将会把此进程从 CPU 释放出来，并把 CPU 分配另外一个进程；如果该进程在时间片结束前阻塞或结束，则 CPU 立即进行切换；通常时间片设为 `20ms~50ms` 通常是一个比较合理的折中值。
 * 最高优先级调度算法
   * 对多用户计算机系统，调度程序能从就绪队列中选择最高优先级的进程进行运行，这称为最高优先级调度算法。
     * 非抢占式：当就绪队列中出现优先级高的进程，运行完当前进程，再选择优先级高的进程。
     * 抢占式：当就绪队列中出现优先级高的进程，当前进程挂起，调度优先级高的进程运行。
 * 多级反馈队列调度算法
   * 是「时间片轮转算法」和「最高优先级算法」的综合和发展
   * 有多个队列，每个队列优先级从高到低，同时优先级越高时间片越短。
   * 如果有新的进程加入优先级高的队列时，立刻停止当前正在运行的进程，转而去运行优先级高的队列
   工作流程
    * 设置多个队列，每个队列不同优先级，越高，时间片越短
    * 新的进程进入会进入第一级队列，按先来先服务等待调度，如果第一级队列未完成，转为第二级队列，以此类推，直到完成
    * 直到高优先级的队列全部完成后，才调度低优先级队列，如果调度低的时候，有高的加入，则中断进入高优先级

