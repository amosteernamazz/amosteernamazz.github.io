---
layout: article
title: 工厂模式
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
