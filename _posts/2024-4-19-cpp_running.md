---
layout: article
title: C++ 运行时关键字
key: 100004
tags: C++ 运行时 关键字
category: blog
date: 2024-04-19 15:20:13 +08:00
mermaid: true
---


  | 类别 | 问题 | 
  |---|---|
  | ++i与i++ | ++i |
  | ++i与i++ | i++ |
  | C++关键字 | include<> 和 include ""的不同 |
  | C++关键字 | inline的优缺点 |
  | C++关键字 | 虚函数是否可以是inline |
  | C++关键字 | class与struct的默认继承方式 |
  | C++关键字 | 不能继承的类或函数 |
  | C++关键字 | 类与类之间关系 |
  | C++关键字 | 继承控制方式对属性的影响 |
  | C++关键字 | 组合 |
  | C++关键字 | 多态原因 |
  | C++关键字 | 多态类型 |
  | C++关键字 | 多态实现 |



### ++i与i++

 **++i**

  ++i 不会产生临时对象

  ```c++
  int& int::operator++ (){
  *this +=1;
  return *this;
  }
  ```

 **i++**
  
  i++ 产生临时对象，会导致效率降低

  ```c++
  const int int::operator(int){
  int oldValue = *this;
  ++(*this);
  return oldValue;
  }

  ```



### 正则表达式

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



### assert断言

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




### this指针使用

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

