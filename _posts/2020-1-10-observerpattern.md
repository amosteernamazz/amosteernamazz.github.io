---
layout: article
title: 观察者模式
key: 100007
tags: 设计模式 观察者模式 
category: blog
date: 2020-01-10 00:00:00 +08:00
mermaid: true
---


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
