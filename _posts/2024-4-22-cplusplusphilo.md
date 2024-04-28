---
layout: article
title: C++ philosophy
key: 100001
tags: C++ philosophy
category: blog
date: 2024-04-22 14:43:03 +08:00
mermaid: true
---

# C++ philosophy

## 1. express ideas directly in code

- 如果能使用const，则使用const保证代码健壮性。【use **const** consistently, check if member functions modify their object; check if functions modify arguments passed by pointer or reference】

```cpp
#include <iostream>

class MyClass {
private:
    int value;

public:
    // Constructor
    MyClass(int val) : value(val) {}

    // Getter function, does not modify the object, should be marked as const
    int getValue() const {
        return value;
    }

    // Setter function, modifies the object
    void setValue(int newVal) {
        value = newVal;
    }

    // Function that does not modify the object, should be marked as const
    void printValue() const {
        std::cout << "Value: " << value << std::endl;
    }

    // Function that modifies the object
    void incrementValue() {
        value++;
    }
};

// Function that takes a pointer to MyClass and modifies it
void modifyObject(MyClass* obj) {
    obj->setValue(42);
}

int main() {
    const MyClass obj1(10); // obj1 is const, can only call const member functions
    obj1.printValue(); // OK

    // obj1.setValue(20); // Error: cannot modify const object

    MyClass obj2(30);
    obj2.printValue(); // OK

    obj2.incrementValue();
    obj2.printValue(); // Value: 31

    modifyObject(&obj2);
    obj2.printValue(); // Value: 42

    return 0;
}
```

- 返回类型应保证可懂。【**flag uses of casts**, casts neuter the type system】

```cpp
class Date {
    public:
        Month month() const;  // do
        int month();          // don't
        // ...
    };
```

- 建议使用标准lib库。【detect code that mimics the **standard lib】**

```cpp
// bad code example  
  void f(vector<string>& v)
    {
        string val;
        cin >> val;
        // ...
        int index = -1;                    // bad, plus should use gsl::index
        for (int i = 0; i < v.size(); ++i) {
            if (v[i] == val) {
                index = i;
                break;
            }
        }
        // ...
    }

// good code example

    void f(vector<string>& v)
    {
        string val;
        cin >> val;
        // ...
        auto p = find(begin(v), end(v), val);  // better
        // ...
    }
```

- 有量纲的使用变化量。【for number change, use **delta type** rather then double or int】

```cpp
// bad code example
    change_speed(double s);// bad: what does s signify?
// ...
    change_speed(2.3);

// good code example
    change_speed(Speed s);// better: the meaning of s is specified
// ...
    change_speed(2.3);// error: no unit
    change_speed(23_m/ 10s);// meters per second
```

## 2. Say what should be done, rather than just how it should be done.

- 建议使用范围for循环。【simple for loops vs. **range-for loops】**

```cpp
// bad code example
    gsl::index i= 0;
while (i< v.size()) {
// ... do something with v[i] ...
    }

// good code example
for (constauto& x : v) {/* do something with the value of x */ }

// good code example
for (auto& x : v) {/* modify x */ }

// good code example
    for_each(v, [](int x) {/* do something with the value of x */ });
    for_each(par, v, [](int x) {/* do something with the value of x */ });
```

- 建议封装指针。【f(T*, int) interfaces vs. **f(span<T>)** interfaces】

```cpp
// bad code example
template<typename T>void f(T* ptr,int size) {
for (int i= 0; i< size;++i) {
        std::cout<< ptr[i]<< " ";
    }
    std::cout<< std::endl;
}

// good code example
template<typename T>void f(std::span<T> arr) {
for (constauto& element : arr) {
        std::cout<< element<< " ";
    }
    std::cout<< std::endl;
}
```

- 变量有效空间问题。【**loop variables in too large a scope(avoid)】**

```cpp
#include <iostream>

int main() {
    // Example 1: Loop variable i has too large a scope
    int sum1 = 0;
    for (int i = 1; i <= 5; ++i) {
        sum1 += i;
    }
    std::cout << "Sum1: " << sum1 << std::endl;

    // Later in the code, you might accidentally reuse the same loop variable name
    // This could lead to confusion or unintended behavior
    for (int i = 6; i <= 10; ++i) {
        sum1 += i;
    }
    std::cout << "Sum1 after reusing i: " << sum1 << std::endl;

    // Example 2: Properly scoping the loop variable
    int sum2 = 0;
    for (int j = 1; j <= 5; ++j) {
        sum2 += j;
    }
    std::cout << "Sum2: " << sum2 << std::endl;

    // This would result in a compilation error since j is not visible here
    // j += 1; // Error: 'j' was not declared in this scope

    return 0;
}
```

- 建议不使用原生内存管理方法。【n**aked new and delete(avoid)】**

```cpp
#include <iostream>

class MyClass {
public:
    MyClass() {
        std::cout << "MyClass Constructor" << std::endl;
    }

    ~MyClass() {
        std::cout << "MyClass Destructor" << std::endl;
    }

    void doSomething() {
        std::cout << "Doing something..." << std::endl;
    }
};

int main() {
    // Example with naked new and delete (avoid)
    MyClass* obj1 = new MyClass();
    obj1->doSomething();

    // Missing delete can lead to memory leak
    // delete obj1;

    // Improved version using std::unique_ptr
    std::unique_ptr<MyClass> obj2 = std::make_unique<MyClass>();
    obj2->doSomething(); // No need for explicit delete

    // obj2 will be automatically deleted when it goes out of scope

    return 0;
}
```

- 建议不使用太多参数构造，使用多个函数。【**functions with many parameters of built-in types(avoid)】**

```cpp
#include <iostream>

// Function with many parameters of built-in types (avoid)
void processUserData(std::string name, int age, double height, bool isStudent) {
    // Function body - processing user data
    std::cout << "Name: " << name << ", Age: " << age
              << ", Height: " << height << ", Is Student: " << isStudent << std::endl;
    // Additional processing...
}

// Improved version using a structure to encapsulate parameters
struct UserData {
    std::string name;
    int age;
    double height;
    bool isStudent;
};

void processUserDataImproved(const UserData& userData) {
    // Function body - processing user data
    std::cout << "Name: " << userData.name << ", Age: " << userData.age
              << ", Height: " << userData.height << ", Is Student: " << userData.isStudent << std::endl;
    // Additional processing...
}

int main() {
    // Example of using the original function
    processUserData("John Doe", 25, 1.75, true);

    // Example of using the improved function with a structure
    UserData user1 = {"Jane Smith", 30, 1.68, false};
    processUserDataImproved(user1);

    return 0;
}
```

## 3. a program would be completely compile-time type safe

- unions vs. **variant**

```cpp
union MyUnion {
    int intValue;
    float floatValue;
};

int main() {
    union MyUnion myUnion;
    myUnion.intValue = 42;

    // 以下代码尝试以浮点数的方式读取整数值，这可能导致不确定的行为
    float result = myUnion.floatValue;

    printf("%f\n", result);

    return 0;
}

using MyVariant = std::variant<int, double, std::string>;

int main() {
    // 创建一个包含 int 类型的 variant
    MyVariant myVar = 42;

    // 使用 std::get 来获取 variant 中的值
    if (std::holds_alternative<int>(myVar)) {
        std::cout << "The variant contains an int: " << std::get<int>(myVar) << std::endl;
    } else if (std::holds_alternative<double>(myVar)) {
        std::cout << "The variant contains a double: " << std::get<double>(myVar) << std::endl;
    } else if (std::holds_alternative<std::string>(myVar)) {
        std::cout << "The variant contains a string: " << std::get<std::string>(myVar) << std::endl;
    }

    // 修改 variant 中的值
    myVar = 3.14;

    // 再次使用 std::get 获取新的值
    if (std::holds_alternative<double>(myVar)) {
        std::cout << "The variant now contains a double: " << std::get<double>(myVar) << std::endl;
    }

    return 0;
}
```

- 建议使用template替代casts。【casts(avoid or minimize using) vs. **template】**

```cpp
#include <iostream>

void processInteger(int value) {
    // Attempting to cast to double
    double doubleValue = static_cast<double>(value);
    std::cout << "Processed Integer as Double: " << doubleValue << std::endl;
}

int main() {
    int integerValue = 42;
    processInteger(integerValue);

    return 0;
}

template <typename T>
void processValue(const T& value) {
    // Processing the value without explicit cast
    std::cout << "Processed Value: " << value << std::endl;
}

int main() {
    int integerValue = 42;
    double doubleValue = 3.14;

    processValue(integerValue);
    processValue(doubleValue);

    return 0;
}
```

- 建议使用span用以解决数组遍历等问题。【array decay vs. **span】**

```cpp
int main() {
    int arr[5] = {1, 2, 3, 4, 5};

    // 在函数调用中，数组名 arr 发生衰减，被解释为指向数组第一个元素的指针
    printArray(arr, 5);

    return 0;
}

void printArray(int *ptr, int size) {
    for (int i = 0; i < size; ++i) {
        printf("%d ", ptr[i]);
    }
    printf("\n");
}

void printSpan(std::span<int> arr) {
    for (int value : arr) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

int main() {
    int arr[5] = {1, 2, 3, 4, 5};

    // 使用 std::span 避免数组衰减
    printSpan(arr);

    return 0;
}
```

- 建议使用span用以解决数组范围问题【range errors vs. **span】**

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};

    // Attempting to access an element out of range
    int value = numbers[10];  // This can lead to undefined behavior

    std::cout << "Value: " << value << std::endl;  // Undefined behavior, may crash

    return 0;
}

#include <iostream>
#include <span>

int main() {
    int numbers[] = {1, 2, 3, 4, 5};

    // Using std::span to represent a view of the array
    std::span<int> numbersSpan(numbers);

    // Accessing elements without risk of range error
    for (int value : numbersSpan) {
        std::cout << "Value: " << value << std::endl;
    }

    return 0;
}
```

- 在要求精度较高的场景，使用narrow_cast在精度降低时，可以抛出异常【narrowing conversions(avoid or minimize using) vs. **narrow or narrow_cast(GSL)】**

```cpp
double largeValue = 123456789.123;
int intValue = largeValue;  // 缩窄转换，可能导致精度损失或溢出

int main() {
    int intValue = 42;
    short shortValue = gsl::narrow_cast<short>(intValue);

    std::cout << "Original value: " << intValue << std::endl;
    std::cout << "Narrowed value: " << shortValue << std::endl;

    return 0;
}
```

## 4. Don't postpone to run time what can be done well at compile time

- 使用span解决遍历索引等问题【look for pointer arguments **span】**

```cpp
#include <iostream>
#include <cstddef>  // For std::ptrdiff_t

// Function using pointer and size for array processing
void processArray(const int* ptr, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        std::cout << "Value: " << ptr[i] << std::endl;
    }
}

int main() {
    int numbers[] = {1, 2, 3, 4, 5};

    // Using pointer and size to represent a view of the array
    const int* ptr = numbers;
    std::size_t size = sizeof(numbers) / sizeof(numbers[0]);

    // Calling the function with pointer and size
    processArray(ptr, size);

    return 0;
}

#include <iostream>
#include <span>

// Function using std::span for array processing
void processArray(std::span<const int> numbers) {
    for (int value : numbers) {
        std::cout << "Value: " << value << std::endl;
    }
}

int main() {
    int numbers[] = {1, 2, 3, 4, 5};

    // Using std::span to represent a view of the array
    std::span<const int> numbersSpan(numbers);

    // Calling the function with std::span
    processArray(numbersSpan);

    return 0;
}
```

- look for run-time checks for **range violations**
    - 检查程序中是否有对数组范围进行检查的代码
    - 静态分析工具检查程序是否有潜在的范围越界
    - 软件测试中，尝试故意超过范围的值，看程序是否会正确处理

## 5. 一些无法在编译器确定的错误需要在运行时能够检测错误

should endeavor to write programs that in principle can be checked, given sufficient resources(analysis programs, run-time checks, machine resources, time)

- **compile & run-time checkable** with ownership and /or imformation

存在无法通过静态与动态检测其中的内存指针分配

```cpp
// separately compiled, possibly dynamically loaded
extern void f(int* p);

void g(int n)
{
    // bad: the number of elements is not passed to f()
    f(new int[n]);
}

```

大小传递不符合参数规范

```cpp
// separately compiled, possibly dynamically loaded
extern void f2(int* p, int n);

void g2(int n)
{
    f2(new int[n], m);  // bad: a wrong number of elements can be passed to f()
}
```

C++ lib的资源管理无法传递空间大小

```cpp
// separately compiled, possibly dynamically loaded
// NB: this assumes the calling code is ABI-compatible, using a
// compatible C++ compiler and the same stdlib implementation
extern void f3(unique_ptr<int[]>, int n);

void g3(int n)
{
    f3(make_unique<int[]>(n), m);    // bad: pass ownership and size separately
}
```

传递指针和大小作为整体进行传递

```cpp
extern void f4(vector<int>&);   // separately compiled, possibly dynamically loaded
extern void f4(span<int>);      // separately compiled, possibly dynamically loaded
                                // NB: this assumes the calling code is ABI-compatible, using a
                                // compatible C++ compiler and the same stdlib implementation

void g3(int n)
{
    vector<int> v(n);
    f4(v);                     // pass a reference, retain ownership
    f4(span<int>{v});          // pass a view, retain ownership
}
```

将所有权和所有信息进行传递

```cpp
vector<int> f5(int n)    // OK: move
{
    vector<int> v(n);
    // ... initialize v ...
    return v;
}

unique_ptr<int[]> f6(int n)    // bad: loses n
{
    auto p = make_unique<int[]>(n);
    // ... initialize *p ...
    return p;
}

owner<int*> f7(int n)    // bad: loses n and we might forget to delete
{
    owner<int*> p = new int[n];
    // ... initialize *p ...
    return p;
}
```

- 当接口使用字符串参数表示各种选项时，它可能直接在内部知道这些选项的有效性。这种情况下，可以避免不必要的检查，因为实现已经了解可能的选项，并且不需要显式验证每个字符串
- Flag(pointer, count) style interfaces 与 **安全的容器类或标准容器库**

## 6. catch run-time errors early

- Look at pointers and arrays: **Do range-checking early and not repeatedly**

```cpp
void increment1(int* p, int n)    // bad: error-prone
{
    for (int i = 0; i < n; ++i) ++p[i];
}

// 应该更早地去检测代码，而不是在runtime去检测
void use1(int m)
{
    const int n = 10;
    int a[n] = {};
    // ...
    increment1(a, m);   // maybe typo, maybe m <= n is supposed
                        // but assume that m == 20
    // ...
}
```

```cpp
void increment2(span<int> p)
{
    for (int& x : p) ++x;
}

// 根据m与n的关系对函数结果进行判断并输出
void use2(int m)
{
    const int n = 10;
    int a[n] = {};
    // ...
    increment2({a, m});    // maybe typo, maybe m <= n is supposed
    // ...
}
```

```cpp
void use3(int m)
{
    const int n = 10;
    int a[n] = {};
    // ...
    increment2(a);   // the number of elements of a need not be repeated
    // ...
}
```

- Look at conversions: **Eliminate or mark narrowing conversions**

动态检查中dynamic_cast和typeid一般只用于处理多态类型，一般不用于检查转换的精度和范围，对于精度检查，一般通过条件语句来检查

```cpp
int main() {
    try {
        int largeValue = 1000;
        short smallValue = largeValue; // Narrowing conversion
        
        if (smallValue != largeValue) {
            throw std::runtime_error("Narrowing conversion occurred! Data loss or overflow might have happened.");
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    return 0;
}
```

- Look for unchecked values coming from **input**

在runtime检查unchecked values coming from input是在程序runtime时，检查程序中的unchecked输入。这种检查通常是为了确保程序在处理输入时不会出现意外情况或漏洞。如对用户登录系统，检查用户名是否符合输入要求。

```cpp
int main() {
    std::string username;
    std::string password;

    // Simulating user input
    std::cout << "Enter your username: ";
    std::getline(std::cin, username); // Assuming the user enters nothing and just presses enter

    // Runtime check for unchecked values coming from input
    if (username.empty()) {
        std::cerr << "Error: Username cannot be empty." << std::endl;
        // Take appropriate action, such as terminating the program or prompting for input again
        return 1; // Exiting the program with an error code
    }

    // Further checks such as password strength, etc.
    
    return 0;
}
```

- Look for **structured data** (objects of classes with invariants) **being converted into strings(avoid)**

```cpp
Date read_date(istream& is);    // read date from istream

Date extract_date(const string& s);    // extract date from string

void user1(const string& date)    // manipulate date
{
    auto d = extract_date(date);
    // ...
}

void user2()
{
    Date d = read_date(cin);
    // ...
    user1(d.to_string());
    // ...
}
```

1. don‘t add validity checks that **change the asymptotic hebavior** of the interface. O(n) check added into average complexity of O(1)

```cpp
class Jet {    // Physics says: e * e < x * x + y * y + z * z
    float x;
    float y;
    float z;
    float e;
public:
    Jet(float x, float y, float z, float e)
        :x(x), y(y), z(z), e(e)
    {
        // 不需要检查该变量的范围，因为计算复杂度
    }

    float m() const
    {
        // Should I handle the degenerate case here?
        return sqrt(x * x + y * y + z * z - e * e);
    }

    ???
};
```

## Don't leak any resources

leak与cleanable

```cpp
void createVector() {
    std::vector<int>* ptr = new std::vector<int>(); // 分配一个vector<int>的动态数组
    // 这个动态数组被创建，但在函数结束后，指针ptr将会丢失，没有办法释放对应的内存

    // 假设这里发生了一些其他的逻辑操作，但最终函数结束时，ptr仍然没有被释放
}

int main() {
    createVector(); // 调用createVector函数

    // 假设这里是程序的其他部分，这部分不会访问createVector函数中分配的内存

    // 在程序结束时，操作系统会回收createVector函数中分配的内存，
    // 所以这个例子中的内存泄漏是可以被清理的，但它是一种泄漏，因为程序员忘记了释放资源。
    
    return 0;
}
```

```cpp
void f(char* name)
{
    FILE* input = fopen(name, "r");
    // ...
    if (something) return;   // bad: if something == true, a file handle is leaked
    // ...
    fclose(input);
}

void f(char* name)
{
    ifstream input {name};
    // ...
    if (something) return;   // OK: no leak
    // ...
}
```

上述被称为RAII技术，可以保证无泄漏。同时如果保证 type and bounds profiles 那么会保证type和resources safety

- 需要明确pointer的owner【pointers: **non-owners and owners**, for non-owners pointers, mark and owner as such using owner from the GSL】

```cpp
void processArray(int* arr, int size) {
    // 处理数组
}

int main() {
    int* ptr = new int(10); // 使用new在堆上分配一个整数，并将指针赋给ptr，ptr是所有者指针
    std::cout << *ptr << std::endl;

    delete ptr; // 手动释放资源，避免内存泄漏
    return 0;
}

int main() {
    std::unique_ptr<int> ptr(new int(10)); // 使用std::unique_ptr替代所有者指针
    std::cout << *ptr << std::endl;

    // 不需要手动释放资源，当ptr超出作用域时，资源会自动释放

    return 0;
}

void processArray(gsl::owner<int*> arr, int size) {
    // 处理数组
    delete[] arr; // 手动释放资源
}

```

- look for **naked new** and delete

- look for known resource allocating functions return raw pointers(such as fopen()【file open】, malloc()【空间分配】 and strdup()【空间复制】)

```cpp
// 函数声明：返回原始指针的资源分配函数
char* allocateMemory() {
    char* ptr = strdup("Hello, world!"); // strdup 函数用于复制字符串，返回动态分配的内存的指针
    return ptr;
}

int main() {
    char* ptr = allocateMemory(); // 调用返回原始指针的资源分配函数

    // 使用返回的指针
    if (ptr != nullptr) {
        std::cout << ptr << std::endl;
        free(ptr); // 手动释放内存，因为 strdup 返回的是动态分配的内存
    }

    return 0;
}
```

## Don’t waste time or space

```cpp
// bad
struct X {
    char ch;
    int i;
    string s;
    char ch2;

    X& operator=(const X& a);
    X(const X&);
};

X waste(const char* p)
{
    if (!p) throw Nullptr_error{};
    int n = strlen(p);
    auto buf = new char[n];
    if (!buf) throw Allocation_error{};
    for (int i = 0; i < n; ++i) buf[i] = p[i];
    // ... manipulate buffer ...
    X x;
    x.ch = 'a';
    x.s = string(n);    // give x.s space for *p
    for (gsl::index i = 0; i < x.s.size(); ++i) x.s[i] = buf[i];  // copy buf into x.s
    delete[] buf;
    return x;
}

void driver()
{
    X x = waste("Typical argument");
    // ...
}
```

问题：

- X struct空间过大，数据有冗余
- copy构造(X&&)无法使用
- new delete 冗余
- 函数复杂

```cpp
// good
#include <string>
#include <stdexcept> // For exception handling

struct X {
    char ch;
    int i;
    std::string s;
    char ch2;

    X& operator=(const X& a) {
        // Implementation of copy assignment operator
        if (this != &a) {
            ch = a.ch;
            i = a.i;
            s = a.s;
            ch2 = a.ch2;
        }
        return *this;
    }

    X(const X& other) {
        // Implementation of copy constructor
        ch = other.ch;
        i = other.i;
        s = other.s;
        ch2 = other.ch2;
    }

    // Move constructor
    X(X&& other) noexcept : ch(other.ch), i(other.i), s(std::move(other.s)), ch2(other.ch2) {}

    // Move assignment operator
    X& operator=(X&& other) noexcept {
        if (this != &other) {
            ch = other.ch;
            i = other.i;
            s = std::move(other.s);
            ch2 = other.ch2;
        }
        return *this;
    }
};

X waste(const char* p) {
    if (!p) throw std::invalid_argument("Null pointer exception");
    int n = strlen(p);
    // Directly use std::string instead of manual memory allocation
    std::string buf(p);

    X x;
    x.ch = 'a';
    // Use std::string constructor to allocate space for p
    x.s = std::move(buf);
    return x;
}

void driver() {
    X x = waste("Typical argument");
    // ...
}
```

```cpp
void lower(zstring s)
{
    for (int i = 0; i < strlen(s); ++i) s[i] = tolower(s[i]);
}

// good
void lower(zstring s)
{
    int len = strlen(s);
		for(int i = 0; i < len; i++)
				s[i] = tolower(s[i]);
}
```

对非user-defined postfix++和**prefix++**建议使用prefix

## prefer immutable data to mutable data

**immutable** vs. mutable 

It is easier to reason about constants than about variables. 

Something immutable cannot change unexpectedly. 

Sometimes immutability enables better optimization.

```cpp
#include <iostream>
#include <vector>

// 使用常量定义数组大小
const int ARRAY_SIZE = 5;

// 使用常量引用传递参数
void printVector(const std::vector<int>& vec) {
    for (int num : vec) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

int main() {
    // 定义一个常量
    const int MAX_VALUE = 100;

    // 常量数组
    int array[ARRAY_SIZE] = {1, 2, 3, 4, 5};

    // 常量向量
    std::vector<int> vec = {10, 20, 30, 40, 50};

    // 使用常量
    std::cout << "Max value: " << MAX_VALUE << std::endl;

    // 修改常量的值，编译器将报错
    // MAX_VALUE = 200; // Error: assignment of read-only variable 'MAX_VALUE'

    // 使用常量数组
    std::cout << "Array elements: ";
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;

    // 使用常量引用传递参数
    std::cout << "Vector elements: ";
    printVector(vec);

    // 修改常量向量，编译器将报错
    // vec.push_back(60); // Error: passing 'const std::vector<int>' as 'this' argument discards qualifiers [-fpermissive]

    return 0;
}
```

## Encapsulate messy constructs, rather than spreading through the code

```cpp
int sz = 100;
int* p = (int*) malloc(sizeof(int) * sz);
int count = 0;
// ...
for (;;) {
    // ... read an int into x, exit loop if end of file is reached ...
    // ... check that x is valid ...
    if (count == sz)
        p = (int*) realloc(p, sizeof(int) * sz * 2);
    p[count++] = x;
    // ...
}

// 建议使用STL或者GSL实现的方法
vector<int> v;
v.reserve(100);
// ...
for (int x; cin >> x; ) {
    // ... check that x is valid ...
    v.push_back(x);
}
```

## Use supporting tools as appropriate

静态分析工具

动态分析工具

测试工具

## Use support libraries as appropriate

ISO c++ standard lib（SL the standard library）

guideline suuport lib（GSL）