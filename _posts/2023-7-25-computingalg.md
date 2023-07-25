---
layout: article
title: computional alg
key: 100001
tags: 
category: blog
date: 2023-07-25 00:00:00 +08:00
mermaid: true
---

# 算法

## 常用算法方法

* 凸包算法：凸包是指一组点在平面上的最小凸多边形，也就是说，它包含了所有的点，并且没有任何一个点在它的内部或边界上。计算凸包的算法有很多，比如Graham扫描法，Jarvis步进法，分治法等
* 线段相交算法：线段相交是指给定平面上的一些线段，求出它们之间的所有交点。线段相交的算法有很多，比如扫描线法，平面扫描法，弯曲扫描法等
* 最近点对算法：最近点对是指给定平面上的一些点，求出距离最近的两个点。最近点对的算法有很多，比如分治法，旋转卡壳法，随机化增量法等。
* 最大空圆算法：最大空圆是指给定平面上的一些点，求出一个圆，使得它不包含任何一个点，并且它的半径最大。最大空圆的算法有很多，比如随机化增量法，Voronoi图法，Delaunay三角剖分法等。

### 知道两个点，求直线方程

```c++
#include <iostream>
#include <tuple>
#include <optional>
#include <format>
using namespace std;

const double epsilon = 1e-6;

struct Point{
    double x;
    double y;
};

std::optional<std::tuple<double,double,double>> line_equation(const Point& p1, const Point& p2){
  bool x_equal = abs(p1.x - p2.x) < epsilon;
  bool y_equal = abs(p1.y - p2.y) < epsilon;
  if(x_equal && y_equal){
    return std::nullopt;
  }
  double a = p2.y - p1.y;
  double b = p1.x - p2.x;
  double c = p2.x * p1.y - p1.x * p2.y;
  return std::make_tuple(a,b,c);
}

```


### 点到直线的距离

```c++
#include <cmath>

//定义一个函数，接收点的坐标和直线的系数作为参数，返回点到直线的距离
constexpr double distance(double x0, double y0, double A, double B, double C) noexcept
{
    //判断A和B是否同时为0
    if (A == 0 && B == 0)
    {
        //抛出异常
        throw std::invalid_argument("A and B cannot be both zero");
    }
    //计算分子
    double numerator = std::abs(A * x0 + B * y0 + C);
    //计算分母
    double denominator = std::hypot(A, B);
    //计算并返回距离
    return numerator / denominator;
}
```

### 点到直线的投影

```c++
#include <cmath>

struct Point
{
    double x;
    double y;
};

//定义一个函数，接收点的坐标和直线的系数作为参数，返回点到直线的投影点的坐标
constexpr Point projection(double x0, double y0, double A, double B, double C) noexcept
{
    //判断A和B是否同时为0
    if (A == 0 && B == 0)
    {
        //抛出异常
        throw std::invalid_argument("A and B cannot be both zero");
    }
    //取直线上任意一点Q，可以令x1=0, y1=-C/B
    double x1 = 0;
    double y1 = -C / B;
    //计算向量PQ和n的内积
    double dot = (x0 - x1) * A + (y0 - y1) * B;
    //计算向量n的模长
    double norm = std::hypot(A, B);
    //计算投影点R的坐标
    double x2 = x0 - dot / norm * A;
    double y2 = y0 - dot / norm * B;
    //返回投影点R
    return Point{x2, y2};
}

```

### 点关于直线的对称点


```c++
#include <cmath>

//定义一个结构体，表示一个点的坐标
struct Point
{
    double x;
    double y;
};

//定义一个函数，接收点的坐标和直线的系数作为参数，返回点关于直线的对称点的坐标
constexpr Point symmetric_point(double x0, double y0, double A, double B, double C) noexcept
{
    //判断A和B是否同时为0
    if (A == 0 && B == 0)
    {
        //抛出异常
        throw std::invalid_argument("A and B cannot be both zero");
    }
    //计算k和b
    double k = -A / B;
    double b = -C / B;
    //计算d
    double d = std::abs(k * x0 - y0 + b) / std::sqrt(k * k + 1);
    //计算d'
    double d_prime = d * std::abs(k) / std::sqrt(k * k + 1);
    //计算n的模长
    double norm = std::hypot(A, B);
    //计算R的坐标
    double x2 = x0 - 2 * d_prime * (-B) / norm;
    double y2 = y0 - 2 * d_prime * A / norm;
    //返回R
    return Point{x2, y2};
}
```

### 两条直线的位置关系

```c++

#include <cmath>

//定义一个结构体，表示一个二维平面上的点
struct Point
{
    double x; // x坐标
    double y; // y坐标
};

//定义一个结构体，表示一个二维平面上的直线
struct Line
{
    double A; // 直线方程的系数A
    double B; // 直线方程的系数B
    double C; // 直线方程的系数C

    // 构造函数，通过两个点来确定一条直线
    Line (const Point& p1, const Point& p2)
    {
        A = p2.y - p1.y;
        B = p1.x - p2.x;
        C = p2.x * p1.y - p1.x * p2.y;
    }

    // 构造函数，通过直线方程的系数来确定一条直线
    Line (double A, double B, double C)
    {
        this->A = A;
        this->B = B;
        this->C = C;
    }

    // 判断两条直线是否相交，如果相交，返回true，并计算出交点
    constexpr bool intersect (const Line& l, Point& p) const noexcept
    {
        // 计算两条直线方程的行列式
        double D = std::hypot(A * l.B, B * l.A);
        // 如果行列式为零，说明两条直线平行或重合
        if (D == 0)
        {
            return false;
        }
        // 否则，计算出交点的坐标
        p.x = (B * l.C - C * l.B) / D;
        p.y = (C * l.A - A * l.C) / D;
        return true;
    }

    // 判断两条直线是否平行，如果平行，返回true，并计算出平行距离
    constexpr bool parallel (const Line& l, double& d) const noexcept
    {
        // 计算两条直线方程的行列式
        double D = std::hypot(A * l.B, B * l.A);
        // 如果行列式不为零，说明两条直线不平行
        if (D != 0)
        {
            return false;
        }
        // 否则，计算出平行距离
        d = std::abs (C - l.C) / std::hypot (A, B);
        return true;
    }

    // 判断两条直线是否重合，如果重合，返回true
    constexpr bool coincide (const Line& l) const noexcept
    {
        // 计算两条直线方程的行列式
        double D = std::hypot(A * l.B, B * l.A);
        // 如果行列式不为零，说明两条直线不重合
        if (D != 0)
        {
            return false;
        }
        // 否则，判断常数项是否相等
        return C == l.C;
    }
};
```


### 判断点是否在多边形内部


```c++
#include <cmath>
#include <vector>

//定义一个结构体，表示一个二维平面上的点
struct Point
{
    double x; // x坐标
    double y; // y坐标
};

//定义一个函数，接收一个点和一个多边形（用std::vector表示）作为参数，返回这个点是否在多边形内部
constexpr bool point_in_polygon(const Point& p, const std::vector<Point>& polygon) noexcept
{
    bool inside = false; // 初始化结果为false
    for (std::size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) // 遍历多边形的每条边
    {
        // 判断p的y坐标是否在边的y坐标范围内
        if ((polygon[i].y > p.y) != (polygon[j].y > p.y))
        {
            // 计算p在边上的投影点的x坐标
            auto x = (polygon[j].x - polygon[i].x) * (p.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x;
            // 判断p是否在投影点的左侧
            if (p.x < x)
            {
                // 如果是，说明射线与边相交，结果取反
                inside = !inside;
            }
        }
    }
    return inside; // 返回结果
}
```


### 给定平面内n个点，构造一个多边形


```c++
#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
using namespace std;

// 定义一个二维点的结构体
struct Point {
    double x, y; // 点的横纵坐标
    Point(double x = 0, double y = 0) : x(x), y(y) {} // 构造函数
};

// 计算两个向量的叉积
double cross(const Point& a, const Point& b) {
    return a.x * b.y - a.y * b.x;
}

// 计算两个向量的夹角
double angle(const Point& a, const Point& b) {
    return atan2(cross(a, b), a.x * b.x + a.y * b.y);
}

// 比较两个点的极角大小，如果相等则比较距离大小
bool cmp(const Point& p1, const Point& p2) {
    double ang = angle(p1, p2);
    if (ang == 0) { // 极角相等，比较距离
        return p1.x * p1.x + p1.y * p1.y < p2.x * p2.x + p2.y * p2.y;
    }
    return ang > 0; // 极角大于0，说明p1在p2的逆时针方向
}

// 求给定平面内n个点的凸包
vector<Point> convexHull(vector<Point>& points) {
    int n = points.size(); // 点的个数
    if (n <= 3) return points; // 如果点数小于等于3，直接返回

    // 找到最左下角的点，并将其交换到第一个位置
    int minIndex = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].y < points[minIndex].y || (points[i].y == points[minIndex].y && points[i].x < points[minIndex].x)) {
            minIndex = i;
        }
    }
    swap(points[0], points[minIndex]);

    // 将其他点按照极角排序
    sort(points.begin() + 1, points.end(), cmp);

    // 用一个栈来存储凸包上的点
    stack<Point> s;
    s.push(points[0]); // 先将极点入栈
    s.push(points[1]); // 再将第二个点入栈

    // 从第三个点开始遍历
    for (int i = 2; i < n; i++) {
        // 如果栈顶的两个点和当前点不构成逆时针旋转，则出栈
        while (s.size() > 1) {
            Point top = s.top(); // 栈顶元素
            s.pop(); // 出栈
            Point nextToTop = s.top(); // 出栈后的栈顶元素
            if (cross(top - nextToTop, points[i] - top) > 0) { // 构成逆时针旋转
                s.push(top); // 将原来的栈顶元素再入栈
                break; // 跳出循环
            }
        }
        s.push(points[i]); // 将当前点入栈
    }

    // 将栈中的点放入结果向量中
    vector<Point> res;
    while (!s.empty()) {
        res.push_back(s.top());
        s.pop();
    }
    return res;
}
```

### 给定n个点，找出距离最小的点

```c++
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
using namespace std;

// 定义一个点结构体
struct Point {
    double x, y; // 点的坐标
    Point(double x = 0, double y = 0): x(x), y(y) {} // 构造函数
};

// 定义一个比较函数，按照x坐标升序排序
bool cmp_x(const Point& a, const Point& b) {
    return a.x < b.x;
}

// 定义一个比较函数，按照y坐标升序排序
bool cmp_y(const Point& a, const Point& b) {
    return a.y < b.y;
}

// 定义一个计算两点之间距离的函数
double dist(const Point& a, const Point& b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// 定义一个求解给定点集中最小距离的函数
double solve(vector<Point>& points, int left, int right) {
    // 如果只有一个点，返回无穷大
    if (left == right) return 1e9;
    // 如果只有两个点，返回它们之间的距离
    if (left + 1 == right) return dist(points[left], points[right]);
    // 如果有三个点，返回任意两点之间最小的距离
    if (left + 2 == right) return min(dist(points[left], points[left + 1]), min(dist(points[left], points[right]), dist(points[left + 1], points[right])));
    // 否则，将点集分成两部分
    int mid = (left + right) / 2;
    // 分别求解左右两部分的最小距离
    double d1 = solve(points, left, mid);
    double d2 = solve(points, mid + 1, right);
    // 取两者中较小的值作为当前最小距离
    double d = min(d1, d2);
    // 找出所有距离中线不超过最小距离的点，并按照y坐标排序
    vector<Point> temp;
    for (int i = left; i <= right; i++) {
        if (fabs(points[i].x - points[mid].x) <= d) {
            temp.push_back(points[i]);
        }
    }
    sort(temp.begin(), temp.end(), cmp_y);
    // 遍历这些点，检查是否有更小的距离
    for (int i = 0; i < temp.size(); i++) {
        for (int j = i + 1; j < temp.size() && j <= i + 7; j++) {
            d = min(d, dist(temp[i], temp[j]));
        }
    }
    // 返回最小距离
    return d;
}

```

### 对于平面内n条水平线，m条竖直线，求交点集合

```c++
// 定义点的数据类型
struct Point {
    int x; // 横坐标
    int y; // 纵坐标
    Point(int x = 0, int y = 0) : x(x), y(y) {} // 构造函数
    // 重载比较运算符
    bool operator < (const Point& p) const {
        return x < p.x || (x == p.x && y < p.y);
    }
    bool operator == (const Point& p) const {
        return x == p.x && y == p.y;
    }
    bool operator != (const Point& p) const {
        return !(*this == p);
    }
    // 重载加法运算符
    Point operator + (const Point& p) const {
        return Point(x + p.x, y + p.y);
    }
};

// 定义线段的数据类型
struct Segment {
    Point a; // 端点a
    Point b; // 端点b
    Segment(Point a = Point(), Point b = Point()) : a(a), b(b) {} // 构造函数
};

// 定义端点类型的枚举类型
enum EndpointType {
    LEFT, // 左端点
    RIGHT, // 右端点
    UP, // 上端点
    DOWN // 下端点
};

// 定义横纵坐标范围的常量
const int MIN_X = -10000; // 最小横坐标
const int MAX_X = 10000; // 最大横坐标
const int MIN_Y = -10000; // 最小纵坐标
const int MAX_Y = 10000; // 最大纵坐标


// 定义一个比较函数，用于优先队列的排序规则
struct CompareEndpoint {
    bool operator () (const Endpoint& e1, const Endpoint& e2) const {
        return e1.first > e2.first || (e1.first == e2.first && e1.second > e2.second);
    }
};

// 定义一个优先队列，存储所有的端点，并按照横坐标从小到大排序
priority_queue<Endpoint, vector<Endpoint>, CompareEndpoint> pq;

// 定义一个比较函数，用于set的排序规则
struct CompareHorizontal {
    bool operator () (const int& y1, const int& y2) const {
        return y1 < y2;
    }
};

// 定义一个set，存储当前活跃的水平线，并按照纵坐标从小到大排序
set<int, CompareHorizontal> horizontal;

// 定义一个map，存储当前活跃的竖直线，并记录它们的横坐标
map<int, bool> vertical;


// 定义一个函数，实现扫描线算法的主要逻辑
vector<Point> scanLine(vector<Segment>& horizontalLines, vector<Segment>& verticalLines) {
    // 定义一个向量，存储所有的交点集合
    vector<Point> intersections;

    // 将所有的水平线和竖直线的端点放入优先队列中，并按照横坐标从小到大排序
    for (auto& l : horizontalLines) {
        pq.push({l.a, LEFT});
        pq.push({l.b, RIGHT});
    }
    for (auto& l : verticalLines) {
        pq.push({l.a, UP});
        pq.push({l.b, DOWN});
    }

    // 从左到右扫描平面上的所有线段的端点
    while (!pq.empty()) {
        Endpoint p = pq.top();
        pq.pop();

        // 根据端点的类型和所属的线段，更新活跃集合，并检查是否有新的交点产生
        switch (p.second) {
            case LEFT: // 如果p是水平线l的左端点
                horizontal.insert(p.first.y); // 将l插入到平衡二叉树中
                for (auto& v : vertical) { // 遍历哈希表中所有竖直线
                    if (v.second && v.first >= p.first.x && v.first <= p.first.x + 1) { // 如果v存在且与l有交点（假设水平线和竖直线都是单位长度）
                        intersections.push_back(Point(v.first, p.first.y)); // 将交点加入到向量中
                    }
                }
                break;
            case RIGHT: // 如果p是水平线l的右端点
                horizontal.erase(p.first.y); // 将l从平衡二叉树中删除
                break;
            case UP: // 如果p是竖直线v的上端点
                vertical[p.first.x] = true; // 将v插入到哈希表中，并标记为存在
                auto it = horizontal.lower_bound(p.first.y); // 找到平衡二叉树中第一个大于等于p纵坐标的水平线y1
                if (it != horizontal.end() && *it <= p.first.y + 1) { // 如果y1存在且与v有交点（假设水平线和竖直线都是单位长度）
                    intersections.push_back(Point(p.first.x, *it)); // 将交点加入到向量中
                }
                if (it != horizontal.begin()) { // 如果y1不是第一个水平线
                    it--; // 找到前一个水平线y2
                    if (*it >= p.first.y - 1) { // 如果y2与v有交点（假设水平线和竖直线都是单位长度）
                        intersections.push_back(Point(p.first.x, *it)); // 将交点加入到向量中
                    }
                }
                break;
            case DOWN: // 如果p是竖直线v的下端点
                vertical[p.first.x] = false; // 将v从哈希表中删除，并标记为不存在
                break;
        }
    }

    // 返回交点集合
    return intersections;
}


// 定义一个函数，生成n条随机的水平线
vector<Segment> generateHorizontalLines(int n) {
    vector<Segment> horizontalLines;
    srand(time(NULL)); // 设置随机数种子
    for (int i = 0; i < n; i++) {
        int x1 = rand() % (MAX_X - MIN_X + 1) + MIN_X; // 随机生成左端点的横坐标
        int x2 = x1 + 1; // 右端点的横坐标为左端点加1（假设水平线都是单位长度）
        int y = rand() % (MAX_Y - MIN_Y + 1) + MIN_Y; // 随机生成纵坐标
        horizontalLines.push_back(Segment(Point(x1, y), Point(x2, y))); // 将水平线加入到向量中
    }
    return horizontalLines;
}

// 定义一个函数，生成m条随机的竖直线
vector<Segment> generateVerticalLines(int m) {
    vector<Segment> verticalLines;
    srand(time(NULL)); // 设置随机数种子
    for (int i = 0; i < m; i++) {
        int x = rand() % (MAX_X - MIN_X + 1) + MIN_X; // 随机生成横坐标
        int y1 = rand() % (MAX_Y - MIN_Y + 1) + MIN_Y; // 随机生成下端点的纵坐标
        int y2 = y1 + 1; // 上端点的纵坐标为下端点加1（假设竖直线都是单位长度）
        verticalLines.push_back(Segment(Point(x, y1), Point(x, y2))); // 将竖直线加入到向量中
    }
    return verticalLines;
}

```

### 求凸包的直径算法

```c++
旋转卡壳方法是一种求解凸包的直径的高效算法，它的基本思想是：从凸包上任意选取两个点作为初始对踵点，然后沿着凸包的边界逆时针旋转这两个点，直到找到最远的一对点，这就是凸包的直径。旋转卡壳方法的时间复杂度是O(n)，其中n是凸包上的点的个数。

用C++实现旋转卡壳方法的代码如下：

#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// 定义一个二维点的结构体
struct Point {
    double x, y; // 点的坐标
    Point(double x = 0, double y = 0) : x(x), y(y) {} // 构造函数
};

//定义一个向量结构体
struct Vector {
    double x, y; //向量的坐标
    Vector(double x = 0, double y = 0): x(x), y(y) {} //构造函数
    Vector(Point a, Point b): x(b.x - a.x), y(b.y - a.y) {} //由两点构造向量
};

// 计算两个点之间的距离
double dist(const Point& a, const Point& b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

//计算向量的叉积
double cross(Vector a, Vector b) {
    return a.x * b.y - a.y * b.x;
}

//计算向量的模长
double length(Vector v) {
    return sqrt(v.x * v.x + v.y * v.y);
}

//旋转卡壳方法求凸包的直径
double rotating_calipers(vector<Point>& points) {
    int n = points.size(); //凸包上的点数
    if (n == 2) return dist(points[0], points[1]); //如果只有两个点，直接返回距离
    double ans = 0; //记录最大距离
    int j = 2; //第二个指针，初始指向第三个点
    for (int i = 0; i < n; i++) { //第一个指针，从第一个点开始逆时针旋转
        Point a = points[i]; //第一个顶点
        Point b = points[(i + 1) % n]; //第二个顶点，如果超过n，取余数
        while (cross(Vector(a, b), Vector(a, points[j])) < cross(Vector(a, b), Vector(a, points[(j + 1) % n]))) { //如果叉积变小，说明第二个指针需要继续旋转
            j = (j + 1) % n; //第二个指针逆时针旋转一步
        }
        ans = max(ans, max(dist(a, points[j]), dist(b, points[j]))); //更新最大距离，比较当前两对顶点的距离
    }
    return ans; //返回最大距离
}


```


