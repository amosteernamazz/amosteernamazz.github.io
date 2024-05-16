---
layout: article
title: 设计模式
key: 100006
tags: 设计模式
category: blog
date: 2020-01-10 00:00:00 +08:00
mermaid: true
---


# 工厂模式

 **目的**

  * 程序更规范有条理
    * 当我们创建实例对象时,如果不仅仅做赋值这样简单的事情,而是有一大段逻辑,那这个时候我们要将所有的初始化逻辑都写在构造函数中吗? 显然，这样使得代码很难看
  * 降低耦合度，提高可阅读性
    * 将长的代码进行"切割"，再将每一个小逻辑都"封装"起来，这样可以降低耦合度,修改的时候也可以直奔错误段落

## 实现方式


 **区别**

  * 懒汉式指系统运行中，实例并不存在，只有当需要使用该实例时，才会去创建并使用实例
  * 饿汉式指系统一运行，就初始化创建实例，当需要时，直接调用即可


<!--more-->

### 简单工厂方式
 简单工厂模式是由一个工厂对象决定创建出来哪一种产品类的实例

 **特点**

  * 由一个工厂类根据传入的参数，动态决定应该创建哪一类产品类

 **缺点**

  * 每次新增一个类时，都需要改变工厂函数，破坏了封装性
    * （如果生产厂家需要新增一个产品,那么工厂函数Factory就需要跟着改变,所以上面的工厂模式违背了**开放封闭原则**）
    * **开放封闭原则**：软件实体（类、模块、函数）可以扩展，但是不可修改

  ```c++
  #include <iostream>
  #include "stdafx.h"

  using namespace std;
  class Product{

  public:
    virtual void show = 0;
  }

  class Product_A : public Product{
    public:
    void show{
      cout << "Product_A" << endl;
    }
  }
  class Product_B : public Product{
    public:
    void show{
      cout << "Product_B" << endl;
    }
  }

  class Factory{
  public:
    Product* Create(int i){
      switch(i){
        case 1:
          return new Product_A;
          break;
        case 2:
          return new Product_A;
          break;
        default:
          break;
      }
    }
  }
  ```
### 工厂方法模式

 多个工厂，多个产品，每个产品对应一个工厂
 工厂和产品都是通过虚基类的方式构建

 **特点**

  * 定义用于创建对象的接口，让子类决定实例化哪个类
  * 当增加一个新产品时，同时增加一个新工厂。
    * 增加新工厂属于扩展，不会修改以前工厂类和产品类的任何代码


 **缺点**

  * 每增加一个新的产品,就需要增加一个新的工厂


  ```c++
  #include <iostream>
  #include "stdafx.h"
  using namespace std;
  class Product{
    public:
      virtual void show() =0;
  };
  class Product_A : public Product{
    public:
      void show(){
        cout << "Product_A" << endl;
      }
  };
  class Product_B : public Product{
    public:
      void show(){
        cout << "Product_B" << endl;
      }
  };
  class Factory{
    public:
      virtual Product* create() = 0;
  };
  class Factory_A{
    public:
      Product* create(){
        return new Product_A;
      }
  };
  class Factory_B{
    public:
      Product* create(){
        return new Product_B;
      }
  };
  ```
### 抽象工厂模式

 多个工厂，多个产品，并且每个产品可以包含多个型号
 工厂和产品通过虚基类构建，每个工厂类可以生产同一个产品的多个型号

 **特点**

  * 提供一个创建一系列相关或相互依赖对象的接口，而无需指定它们具体的类

 **优点**

  * 当一个产品族中的多个对象被设计成一起工作时，它能保证客户端始终只使用同一个产品族中的对象
  * 易于交换产品系列，由于具体工厂类在一个应用中只需要在初始化的时候出现一次，这样就使得改变一个应用的具体工厂变得非常容易，只需要改变具体工厂即可使用不同的产品配置。
  * 让具体的创建实例过程与客户端分离，客户端是通过它们的抽象接口操纵实例，产品的具体类名也被具体工厂实现分离，不会出现在客户代码中。

 **缺点**

  * 产品族扩展非常困难，要增加一个系列的某一产品，既要在抽象的 Creator 里加代码，又要在具体的里面加代码。

  ```c++
  #include <iostream>
  using namespace std;
  class product1{
    public:
      virtual void show() = 0;
  };
  class product1_a : public product1{
    public:
      void show(){
        cout << "product1 a" <<endl;
      }
  };
  class product1_b : public product1{
    public:
      void show(){
        cout << "product1 b" <<endl;
      }
  };
  class product2{
    public:
      virtual void show() = 0;
  };
  class product2_a : public product1{
    public:
      void show(){
        cout << "product2 a" <<endl;
      }
  };
  class product2_b : public product1{
    public:
      void show(){
        cout << "product2 b" <<endl;
      }
  };
  class Factory{
    public:
      virtual product1* create1() = 0;
      virtual product2* create2() = 0;
  };
  class FactoryA{
    public:
      product1* create1(){
        return new product1_a;
      }
      product2* create2(){
        return new product1_b;
      }
  };
  class FactoryB{
    public:
      product1* create1(){
        return new product2_a;
      }
      product2* create2(){
        return new product2_b;
      }
  };
  ```

# 单例模式

 **目的**
  * 全局只有一个实例
    * 不能通过new来构建对象
      * 构造函数必须私有
      * 类内部必须new出对象 -> static方法

 **要点**
  * 全局只有一个实例，使用static实现，构造设为私有
  * 通过公共接口获得实例
  * 线程安全
  * 禁止拷贝和赋值


## 实现方式

 **区别**
  * 懒汉式指系统运行中，实例并不存在，只有当需要使用该实例时，才会去创建并使用实例
  * 饿汉式指系统一运行，就初始化创建实例，当需要时，直接调用即可


<!--more-->

### 懒汉式线程不安全

  ```c++
  #include <iostream>
  #include <mutex>
  #incldue <pthread.h>

  class SingleInstance{

  public:

    // 获取单例对象
    static SingleInstance* GetInstance();

    // 释放单例
    static void deleteInstance();

    // 打印
    void Print(); 

  private:

    // 禁止构造和析构
    SingleInstance();
    ~SingleInstance();

    // 禁止拷贝和赋值
    SingleInstance(const SingleInstance& single);
    const SingleInstance& operator=(const SingleInstance& single);

    // 对象
    static SingleInstance* m_SingleInstance; 
  };

  SingleInstance* SingleInstance::m_SingleInstance = NULL;

  SingleInstance* SingleInstance::GetInstance(){
    if(m_SingleInstance == NULL){
      // 有线程竞争的问题，可能会创建多个实例
      // 使用new构建，并没有用shared_ptr构建
      m_SingleInstance = new (std::nothrow) SingleInstance;
    }
    return m_SingleInstance;
  }
  void SingleInstance::deleteInstance(){
    if(m_SingleInstance){
      delete m_SingleInstance;
      m_SingleInstance = NULL;
    }
  }
  void SingleInstance::Print(){
    std::cout << this << std::endl;
  }
  SingleInstance::SingleInstance()
  {
    std::cout << "构造函数" << std::endl;
  }
    
  SingleInstance::~SingleInstance()
  {
    std::cout << "析构函数" << std::endl;
  }

  ```
### 线程安全的懒汉式使用静态声明的方式
  ```c++
  class Singleton
  {
  public:
      ~Singleton(){
          std::cout<<"destructor called!"<<std::endl;
      }
      //或者放到private中
      Singleton(const Singleton&)=delete;
      Singleton& operator=(const Singleton&)=delete;
      static Singleton& get_instance(){
          //关键点！
          static Singleton instance;
          return instance;
      }
      //不推荐，返回指针的方式
      /*static Singleton* get_instance(){
          static Singleton instance;
          return &instance;
  	}*/
  private:
      Singleton(){
          std::cout<<"constructor called!"<<std::endl;
      }
  };
  ```
### 线程安全懒汉式使用锁的方式

  ```c++
  #include <iostream>
  #include <memory>
  #include <mutex>

  class Singleton{

  public:
    typedef std::shared_ptr<Singleton> Ptr;
    ~Singleton(){
      std::cout<< "destructor called" <<std::endl;
    }
    Singleton(Singleton&) = delete; 
    Singleton& operator=(const Singleton&) = delete;
    static Ptr Singleton(){
      // 先判断是否符合加锁条件
      if(m_singleton == nullptr){
        std::lock_guard<std::mutex> lk(m_mutex);
        // 保证单例
        if(m_singleton == nullptr){
          m_singleton = std::shared_ptr<Singleton> (new Singleton);
        }
      }
      return m_singleton;
    }

  private:
    static Singleton m_singleton;
    static std::mutex m_mutex;
  };

  Singleton::Ptr Singleton::m_singleton = nullptr;
  std::mutex Singleton::m_mutex;

  ```

### 饿汉式
```c++

class Singleton{
public:
  static Singleton* getSingleton();
  static void deletesingle();

private:
  Singleton();
  ~Singleton();
  Singleton(const Singleton& single);
  const Singleton& operator=(const Singleton& single);
  static Singleton* single;
};
Singleton* Singleton::single = new(std::nothrow) Singleton;

Singleton* Singleton::getSingleton(){
  return single;
}
void Singleton::deletesingle(){
  if(single){
    delete single;
    single = NULL;
  }
}

Singleton::Singleton(){

}

Singleton::~Singleton(){

}
```


## 面试题

 * 懒汉模式和恶汉模式的实现（判空！！！加锁！！！），并且要能说明原因（为什么判空两次？）
 * 构造函数的设计（为什么私有？除了私有还可以怎么实现（进阶）？）
 * 对外接口的设计（为什么这么设计？）
 * 单例对象的设计（为什么是static？如何初始化？如何销毁？（进阶））
 * 对于C++编码者，需尤其注意C++11以后的单例模式的实现（为什么这么简化？怎么保证的（进阶））


# 观察者模式
![](https://images2015.cnblogs.com/blog/765168/201608/765168-20160814145059437-1060662569.png)

 **定义**
  * 被观察者叫做subject，观察者叫做observer
  * 定义对象之间的一对多的依赖关系，当每个对象改变，所有依赖它的对象都会得到通知并更新

 **观察者中的角色**
  * subject
    * 目标知道它的观察者。可以有任意多个观察者观察同一个目标
    * 提供注册和删除观察者对象的接口
  * Observer
    * 为那些在目标发生改变时需获得通知的对象定义一个更新接口
  * ConcreteSubject
    * 将有关状态存入各ConcreteObserver对象
    * 当它的状态发生改变时，向它的各个观察者发出通知
  * ConcreteObserver
    * 维护一个指向ConcreteSubject对象的引用
    * 存储有关状态，这些状态应与目标的状态保持一致
    * 实现Observer的更新接口以使自身状态与目标的状态保持一致
 
 **优点**
  * 实现表示层和数据逻辑层的分离
  * 观察者支持广播通信，观察目标会向所有的注册的
 
 **目的**
  * 程序更规范有条理
    * 当我们创建实例对象时,如果不仅仅做赋值这样简单的事情,而是有一大段逻辑,那这个时候我们要将所有的初始化逻辑都写在构造函数中吗? 显然，这样使得代码很难看
  * 降低耦合度，提高可阅读性
    * 将长的代码进行"切割"，再将每一个小逻辑都"封装"起来，这样可以降低耦合度,修改的时候也可以直奔错误段落

 **场景**
  * 当一个抽象模型有两个方面，其中一个方面依赖于另一方面。将这二者封装在独立的对象中以使它们可以各自独立的改变和复用
  * 当对一个对象的改变需要同时改变其它对象，而不知道具体有多少对象有待改变
  * 当一个对象必须通知其它对象，而它又不能假定其它对象是谁；也就是说，你不希望这些对象是紧密耦合的。

## 实现方式

  ```c++
  #include <iostream>
  #include <list>
  using namespace std;

  class Observer{
    public:
      virtual void Update(int) =0;
  };



  class Subject{
    public:
      virtual void attach(c* ) =0;
      virtual void detach(Observer* ) =0;
      virtual void notify() =0;
  };

  class concreateobserver : public Observer{
    public:
      concreateobserver(Subject* subject){
        m_subject = subject;
      }

      void Update(int value){
        cout << "concreate observer get the update: " << value << endl;
      }

    private:
      Subject* m_subject;
  };


  class concreatesubject : public Subject{
    public:
      void attach(Observer* pobserver);
      void detach(Observer* pobserver);
      void notify();

      void setstate(int state){
        m_state = state;
      }

    private:
      std::list<Observer*> m_observerList;
      int m_state;
  };

  void ConcreteSubject::Attach(Observer *pObserver)
  {
    m_ObserverList.push_back(pObserver);
  }
  
  void ConcreteSubject::Detach(Observer *pObserver)
  {
    m_ObserverList.remove(pObserver);
  }

  void ConcreteSubject::Notify()
  {
      std::list<Observer *>::iterator it = m_ObserverList.begin();
      while (it != m_ObserverList.end())
      {
          (*it)->Update(m_iState);
          ++it;
      }
  }

  ```
