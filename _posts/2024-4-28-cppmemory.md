---
layout: article
title: C++ 运行时处理
key: 100007
tags: C++ 内存布局
category: blog
date: 2019-09-07 00:00:00 +08:00
mermaid: true
---


## 运行时的堆与栈上的对象

### 堆对象与栈对象的区别

  * 对象生命周期区别
    * 栈对象生命周期只能在作用域内，堆生命周期可以在动态地分配和释放内存的过程中随时变化。
  * 对象的性能
    * 栈性能更快，栈有专门的寄存器，压栈出栈指令效率更高，堆是由OS动态调度，堆内存可能被OS调度在非物理内存中，或是申请内存不连续，造成碎片过多等问题；
    * 堆都是动态分配的，栈是编译器完成的。栈的分配和堆是不同的，他的动态分配是由编译器进行释放，无需我们手工实现
  * 某些情况如果是在栈上创建，但数据仍然在堆上，如`std::vector v`，对象v创建在栈，但其数据在堆上，只是指针指向堆，堆上的数据由std负责维护

### 只在堆上生成对象的类

```c++
class A {
    // A a; 创建对象是利用编译器确定，需要public的构造和析构，因此使用private或protected构造和析构可以取消静态创建，但针对需要继承的类型有进一步限制为protected
protected:
    A() {} 
    ~A() {}
public:
    static A* create() {
        return new A();
    }
    // 在析构中因为无法调用，使用单独的delete()函数
    void destroy() {
        delete this;
    }

};

int main() {
    A* object = A::create();

    object->destroy();
    object = nullptr;

    return 0;
}
```

### 只在栈上生成对象的类


```c++
class A {
private:
    void operator delete(void* ptr) {}
    void* operator new (size_t t) {}
public:
    A() {}
    ~A() {}
};

int main() {
    A a;
    return 0;
}

```


## 栈上分配内存

 **alloca**

  * 不需要手动释放，超出作用域自动释放


 **问题**

  * 会爆栈

## 野指针和悬空指针

 * 野指针：没有初始化的指针
 * 悬空指针：指向内存已经被释放了

## 内存泄漏
 
 申请了一块内存，使用完毕后没有释放。程序运⾏时间越⻓，占⽤内存越多，最终⽤尽全部内存，整个系统崩溃。
 
 由程序申请的⼀块内存，且没有任何⼀个指针指向它，那么这块内存就泄漏了。
 
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

## 智能指针种类
  
 * unique_ptr（独占式）、shared_ptr（共享式、强引用）、weak_ptr（弱引用，只提供访问，但不管理）


 ```c++
 T* get();  // 获得原生指针
 T& operator*();  // 重写*
 T* operator->(); // 重写->
 T& operator=(const T& val);  // 重写=
 T* release();  // 释放智能指针，返回原生指针
 void reset (T* ptr = nullptr); // 释放原先的对象，将智能指针管理新指针的对象
 ```

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


### 内存泄漏问题

 **原因**
  * 在局部作用域消失时，data区仍然保存其内存空间
  * 执行路径不明
    * 对于局部静态变量，构造和析构都取决于程序的执行顺序。程序的实际执行路径不可预知的
  * 关系不明
    * 局部静态变量分布在程序代码各处，彼此直接没有明显的关联，很容易让开发者忽略它们之间的这种关系

 **建议**
  * 减少使用局部静态变量
