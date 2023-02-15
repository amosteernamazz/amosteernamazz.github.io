---
layout: article
title: 单例模式
key: 100005
tags: 设计模式 单例模式
category: blog
date: 2020-01-03 00:00:00 +08:00
mermaid: true
---


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
