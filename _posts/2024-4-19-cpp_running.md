---
layout: article
title: C++ 运行时关键字
key: 100004
tags: C++ 运行时 关键字
category: blog
date: 2024-04-19 15:20:13 +08:00
mermaid: true
---



## 正则表达式

 **常用的C++正则表达式特殊字符与含义**

  * `.`：匹配任意单个字符（除了换行符）。`hello\nworld\nhello again`中的`.e`
  * `^`：匹配输入字符串的开始位置。如果用在多行模式里，它还可以匹配每一行的开头。`hello\nworld\nhello again`中的`^hello`
  * `$`：匹配输入字符串的结束位置。如果用在多行模式里，它还可以匹配每一行的结尾。`This is line 1.\nThis is line 2.\nThis is line 3.`中的`\\n$`
  * `*`：匹配前面的子表达式零次或多次。例如，`zo*` 能匹配 `z` 以及 `zoo`
  * `+`：匹配前面的子表达式一次或多次。例如，`zo+` 能匹配 `zo` 以及 `zoo`，但不能匹配 `z`
  * `?`：匹配前面的子表达式零次或一次。例如，`do(es)?` 可以匹配 `do` 或 `does`
  * `{n}`：n 是一个非负整数。匹配确定的 n 次。例如，`o{2}` 不能匹配 `Bob` 中的 `o`，但是能匹配 `food` 中的两个 `o`
  * `{n,}`：n 是一个非负整数。匹配至少 n 次。例如，`o{2,}` 不能匹配 `Bob` 中的 `o`，但能匹配 `foooood` 中的所有 `o`
  * `{n,m}`：m 和 n 均为非负整数，其中 n <= m。匹配至少 n 次，但不超过 m 次。例如，`o{1,3}` 将匹配 `fooooood` 中的前三个 `o`
  * `[...]`：字符集。匹配方括号内的任意一个字符。例如，`[abc]` 将匹配 `plain` 中的 `a`
  * `[^...]`：否定字符集。匹配任何不在方括号内的字符。例如，`[^abc]` 将匹配 `plain` 中的 `p`，`l`，`i` 和 `n`
  * `\`：转义字符。用于匹配具有特殊含义的字符，或者将特殊字符转义为普通字符。例如，`\.` 匹配句点字符
  * `|`：或者。匹配 | 符号前后的任意一个表达式。例如，`zoo|dog` 能匹配 `zoo` 或 `dog`
  * `(...)`：捕获括号。用于将匹配到的子串分组，并可以在后续的正则表达式或替换操作中引用。
  * `\d`：匹配任意数字字符，等价于 `[0-9]`。
  * `\D`：匹配任意非数字字符，等价于 `[^0-9]`。
  * `\s`：匹配任意空白字符，包括空格、制表符、换页符等。
  * `\S`：匹配任意非空白字符。
  * `\w`：匹配任意字母、数字或下划线字符，等价于 `[a-zA-Z0-9_]`
  * `\W`：匹配任意非字母、非数字、非下划线字符。

 **应用**

  * 一种文本处理工具，通过定义一种模式，在一个字符串中搜索匹配该模式的子串。
    * 验证数据格式：验证用户输入是否符合预期格式，如电子邮件地址、电话号码、社会安全号码等；验证文件路径或URL的格式是否正确。
    * 文本搜索和替换：在大量文本中查找特定的模式或单词，并进行替换；在源代码或日志文件中搜索特定的错误或警告信息。
    * 词法分析：在编译器或解释器的词法分析阶段，使用正则表达式来识别编程语言中的关键字、标识符、数字、运算符等。
    * 分割字符串：使用正则表达式作为分隔符来分割字符串，这比传统的基于固定字符的分割方法更灵活。
    * 数据清洗：清理HTML或XML标签，从文本中移除不必要的字符或格式；去除文本中的特殊字符或空格。
    * 模式匹配：在生物学中，用于匹配DNA或RNA序列；在网络安全领域，用于识别恶意代码或网络流量中的异常模式。
    * 提取信息：从日志文件中提取日期、时间、事件类型等信息；从网页或文本文件中提取特定的数据字段。

```c++
// 验证电子邮箱格式
#include <iostream>  
#include <regex>  
#include <string>  
  
bool isValidEmail(const std::string& email) {  
    std::regex e ("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$");  
    return std::regex_match(email, e);  
}  
  
int main() {  
    std::string email = "example@example.com";  
    if (isValidEmail(email)) {  
        std::cout << email << " is a valid email address." << std::endl;  
    } else {  
        std::cout << email << " is not a valid email address." << std::endl;  
    }  
    return 0;  
}

// 文本替换
#include <iostream>  
#include <regex>  
#include <string>  
  
std::string replaceNumbersWithStars(const std::string& text) {  
    std::regex e ("\\d+"); // 匹配一个或多个数字  
    return std::regex_replace(text, e, "*");  
}  
  
int main() {  
    std::string text = "There are 123 apples and 456 oranges.";  
    std::string result = replaceNumbersWithStars(text);  
    std::cout << result << std::endl; // 输出: There are *** apples and *** oranges.  
    return 0;  
}

// 词法分析，识别文本种的数字
#include <iostream>  
#include <regex>  
#include <string>  
#include <vector>  
  
std::vector<std::string> tokenizeNumbers(const std::string& text) {  
    std::regex e ("\\d+"); // 匹配一个或多个数字  
    std::sregex_token_iterator iter(text.begin(), text.end(), e);  
    std::sregex_token_iterator end;  
    std::vector<std::string> tokens(iter, end);  
    return tokens;  
}  
  
int main() {  
    std::string text = "The price is 123 dollars.";  
    auto tokens = tokenizeNumbers(text);  
    for (const auto& token : tokens) {  
        std::cout << token << std::endl; // 输出: 123  
    }  
    return 0;  
}

// 分割字符串
#include <iostream>  
#include <regex>  
#include <string>  
#include <vector>  
  
std::vector<std::string> splitString(const std::string& text, const std::string& delimiter) {  
    std::regex e (delimiter);  
    std::sregex_token_iterator iter(text.begin(), text.end(), e, -1);  
    std::sregex_token_iterator end;  
    std::vector<std::string> tokens(iter, end);  
    return tokens;  
}  
  
int main() {  
    std::string text = "apple,banana,cherry";  
    auto tokens = splitString(text, ",");  
    for (const auto& token : tokens) {  
        std::cout << token << std::endl; // 输出: apple, banana, cherry  
    }  
    return 0;  
}

// 数据清洗，清除HTML标签
#include <iostream>  
#include <regex>  
#include <string>  
  
std::string removeHtmlTags(const std::string& html) {  
    std::regex e ("<[^>]*>"); // 匹配所有HTML标签  
    return std::regex_replace(html, e, "");  
}  
  
int main() {  
    std::string html = "<p>Hello, <b>world</b>!</p>";  
    std::string text = removeHtmlTags(html);  
    std::cout << text << std::endl; // 输出: Hello, world!  
    return 0;  
}

// 模式匹配
#include <iostream>  
#include <regex>  
#include <string>  
  
int main() {  
    // 假设我们有一些可能包含恶意内容的文本  
    std::string text = "This is a normal text. However, it contains some BAD_STUFF that we need to detect.";  
  
    // 定义一个正则表达式来匹配恶意内容  
    // 这个例子中，我们简单地查找"BAD_STUFF"这个字符串  
    std::regex pattern("BAD_STUFF");  
  
    // 使用std::sregex_search来搜索匹配项  
    if (std::sregex_search(text, pattern)) {  
        std::cout << "Malicious content detected!" << std::endl;  
    } else {  
        std::cout << "No malicious content found." << std::endl;  
    }  
  
    return 0;  
}

// 提取信息
#include <iostream>  
#include <regex>  
#include <string>  
#include <sstream>  
  
int main() {  
    // 假设我们有一个包含姓名和年龄的字符串  
    std::string input = "John Doe, 30 years old";  
  
    // 定义一个正则表达式来匹配姓名和年龄  
    std::regex pattern(R"((\w+)\s+(\w+),\s+(\d+)\s+years\s+old)");  
  
    // 创建一个smatch对象来保存匹配结果  
    std::smatch match;  
  
    // 尝试匹配字符串  
    if (std::regex_search(input, match, pattern)) {  
        // 提取姓名和年龄  
        std::string name = match[1].str() + " " + match[2].str();  
        int age = std::stoi(match[3].str());  
  
        // 输出提取的信息  
        std::cout << "Name: " << name << std::endl;  
        std::cout << "Age: " << age << std::endl;  
    } else {  
        std::cout << "No match found." << std::endl;  
    }  
  
    return 0;  
}
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




## this指针使用

 **作用**
  * 指向非静态成员函数所作用的对象

 **什么时候创建**
  * 调用非静态函数时才会使用的

 **delete this**
  * 为将被释放的内存调用一个或多个析构函数（因此不能在析构中调用delete this），类对象的内存空间被释放，之后不能涉及this指针，如操作数据成员，调用虚函数等






## lambda 表达式
 lambda表达式是一种匿名函数对象，可以捕获所在作用域中的变量，并在需要时执行特定的操作。
提供了一种**匿名函数**的特性，可以编写内嵌的匿名函数，用于替换独立函数，而且更可读
本质上来讲， lambda 表达式只是**一种语法糖**，因为所有其能完成的⼯作都可以⽤其它稍微复杂的代码来实现。

 ****

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


## 类型转换
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




## 类型检查

用于检查变量或表达式的类型。例如：

```c++
int n = 10;
if (typeid(n) == typeid(int)) {
  // n 是 int 类型
}
```

## 动态分配和释放内存

### new/delete与malloc/free
 **相同**
  * 申请动态内存和释放动态内存


 **不同**
 new/delete带构造析构部分
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

## 非静态this指针的处理

见编译中的静态thie指针的处理


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
