---
layout: article
title: 数据结构
key: 100008
tags: 算法 排序 CPU算法
category: blog
date: 2020-04-12 00:00:00 +08:00
mermaid: true
---

string转化为int string2int = atoi(string1.c_str())

# STL算法总结

迭代器
begin() end()
rbegin() rend()

sort()方法：vector

## vector

vector<int> vec;

容量
**vec.size()**
**vec.empty()**
**vec.resize()**
vec.reverse()

元素访问
vec[]
vec.front()：返回第一个元素
vec.back()：返回最后一个元素
vec.data()：返回第一个元素的指针

修改

**push_back()**
**pop_back()**
insert()
**erase()**
**clear()**
**swap()**
assign()
**emplace()**
**emplace_back()**

## unordered_map 

unordered_map<int,int> res;

容量
**res.empty()**
**res.size()**

元素访问
**res.find()** 如果找到返回迭代器，否则返回end()
**res.count()**
[]
at()
->second

修改
**res.insert()**
**res.erase()**
**res.clear()**

hash相关

## string

string s1;

容量
s1.length()
**s1.size()**
**s1.resize()**
s1.reserve()

元素访问
s1.front()
s1.back()
**s1.find()**
**s1.substr()**
**s1.empty()**

修改
s1.assign(str1);
**+=**
s1.swap()
s1.push_back()
**s1.pop_back()**
s1.clear()
s1.insert()
s1.append()
s1.erase()
s1.replace()
s1.copy()
s1.tranfrom()

## stack

stack<int> st;

st.push()
st.top()
st.pop()
st.empty()
st.size()

## priority_queue

priority_queue<int, vector<int>, less<int>> queue1;     大堆根
priority_queue<int, vector<int>, greater<int>> queue2;  小堆根

queue1.push()
queue1.top()
queue1.empty()
queue1.pop()
queue1.emplace()

自定义比较函数

## unorder_set

unorder_set<string> sets;

sets.empty()
sets.size()
sets.find()
sets.count()
sets.insert()
sets.erase()
sets.clear()


## ⼿写字符串函数 strcat，strcpy，strncpy，memset，memcpy实现

strcpy

```c++
//把 src 所指向的字符串复制到 dest，注意：dest定义的空间应该⽐src⼤。
char* strcpy(char *dest,const char *src) {
 char *ret = dest;
 assert(dest!=NULL);//优化点1：检查输⼊参数
 assert(src!=NULL);
 while(*src!='\0')
 *(dest++)=*(src++);
 *dest='\0';//优化点2：⼿动地将最后的'\0'补上
 return ret;

//考虑内存᯿叠的字符串拷⻉函数 优化的点
char* strcpy(char *dest,char *src) {
 char *ret = dest;
 assert(dest!=NULL);
 assert(src!=NULL);
 memmove(dest,src,strlen(src)+1);
 return ret;
}

}
```

strcat

```c++
char* strcat(char *dest,const char *src) {
 //1. 将⽬的字符串的起始位置先保存，最后要返回它的头指针
 //2. 先找到dest的结束位置,再把src拷⻉到dest中，记得在最后要加上'\0'
 char *ret = dest;
 assert(dest!=NULL);
 assert(src!=NULL);
 while(*dest!='\0')
 dest++;
 while(*src!='\0')
 *(dest++)=*(src++);
 *dest='\0';
 return ret;
}
```


strcmp

```c++
int strcmp(const char *s1,const char *s2) {
 assert(s1!=NULL);
 assert(s2!=NULL);
 while(*s1!='\0' && *s2!='\0') {
 if(*s1>*s2)
 return 1;
 else if(*s1<*s2)
 return -1;
 else {
 s1++,s2++;
 }
 }
 //当有⼀个字符串已经⾛到结尾
 if(*s1>*s2)
 return 1;
 else if(*s1<*s2)
 return -1;
 else
 return 0;
}
```

strstr

```c++
char* strstr(char *str1,char *str2) {
 char* s = str1;
 assert(str1!='\0');
 assert(str2!='\0');
 if(*str2=='\0')
 return NULL;//若str2为空，则直接返回空
 while(*s!='\0') {//若不为空，则进⾏查询
 char* s1 = s;
 char* s2 = str2;
 while(*s1!='\0'&&*s2!='\0' && *s1==*s2)
 s1++,s2++;
 if(*s2=='\0')
 return s;//若s2先结束
 if(*s2!='\0' && *s1=='\0')
 return NULL;//若s1先结束⽽s2还没结束，则返回空
 s++;
 }
 return NULL;
}
```


memcpy 

```c++
void* memcpy(void* dest, void* src, size_t num) {
 void* ret = dest ;
 size_t i = 0 ;
 assert(dest != NULL ) ;
 assert(src != NULL) ;
 for(i = 0; i<num; i++) {
 //因为void* 不能直接解引⽤，所以需要强转成char*再解引⽤
 //此处的void*实现了泛型编程
 *(char*) dest = *(char*) src ;
 dest = (char*)dest + 1 ;
 src = (char*) src + 1 ;
 }
 return ret ;
}

```

memmove

```c++
//考虑内存᯿叠的memcpy函数 优化的点
void* memmove(void* dest, void* src, size_t num) {
 char* p1 = (char*)dest;
 char* p2 = (char*)src;
 if(p1<p2) {//p1低地址p2⾼地址
 for(size_t i=0; i!=num; ++i)
 *(p1++) = *(p2++);
 }
 else {
 //从后往前赋值
 p1+=num-1;
 p2+=num-1;
 for(size_t i=0; i!=num; ++i)
 *(p1--) = *(p2--);
 }
 return dest;
}
```


# STL
 容器、算法、迭代器、仿函数、配接器、配置器
 容器通过配置器获得空间、算法通过迭代器获得容器内容、仿函数完成策略变化、配接器用于算法、仿函数和迭代器

## 容器
 各种数据结构，vector、list、deque、set、map，用来存放数据

## 算法
 各种常用算法，sort（插入、快排、堆排序）、search（二分查找）

## 迭代器
 将operator* operator-> operator++ operator--等指针操作重载


## 仿函数
 重载operator()的类或类模板

## 配置器
 实现动态空间配置、空间管理和释放的类模板

## 内存管理allocator

 **双层配置器**
  * 第⼀级配置器直接使⽤ malloc()和 free()完成内存的分配和回收
  * 第⼆级配置器则根据需求的⼤⼩选择不同的策略执⾏
    * 如果需求块⼤⼩⼤于 128bytes，则直接转⽽调⽤第⼀级配置器，使⽤malloc()分配内存。
    * 如果需求块⼤⼩⼩于 128bytes，第⼆级配置器中维护了 16 个⾃由**链表**，负责 16 种⼩型区块的次配置能⼒。
    * ⾸先查看所需需求块⼤⼩所对应的链表中是否有空闲空间，如果有则直接返回，如果没有，则向**内存池**中申请所需需求块⼤⼩的内存空间，如果申请成功，则将其加⼊到⾃由链表中。如果内存池中没有空间，则使⽤ malloc() 从**堆**中进⾏申请，且申请到的⼤⼩是需求的⼆倍（或⼆倍＋n 附加），⼀倍放在⾃由空间中，⼀倍（或⼀倍＋n）放⼊内存池中。
    * 如果 malloc()也失败，则会遍历⾃由空间**链表**，四处寻找“尚有未⽤区块，且区块够⼤”的freelist，找到⼀块就挖出⼀块交出。如果还是没有，仍交由 malloc()处理，因为 malloc() 有out-of-memory 处理机制或许**有机会释放其他的内存**拿来⽤，如果可以就成功，如果不⾏就报bad_alloc **异常**。




### vector

 特点：连续空间、三个迭代器、扩充空间

  是动态空间，随着元素的加⼊，它的内部机制会**自行扩充空间**以容纳新元素。vector 维护的是⼀个**连续的线性空间**，⽽且普通指针就可以满⾜要求作为 vector 的迭代器
  
  有**三个迭代器**
   * ⼀个指向⽬前使⽤空间头的 iterator
   * ⼀个指向⽬前使⽤空间尾的 iterator
   * ⼀个指向⽬前可⽤空间尾的 iterator
  
  当有**新的元素插⼊**时，如果⽬前够⽤则直接插⼊，如果不够，则扩充⾄两倍，如果两倍不⾜，就扩张⾄⾜够⼤的容量。是申请⼀块连续空间，将原有的数据拷⻉到新空间中，再释放原有空间，完成⼀次扩充。需要注意的是，每次扩充是重新开辟的空间，所以扩充后，**原有的迭代器**将会失效

### list

 特点：非连续空间、双向
  每次**插⼊**或删除⼀个元素，就配置或释放⼀个空间，⽽且**原有的迭代器**也不会失效。
  双向链表。普通指针已经不能满⾜ list 迭代器的需求，因为 list 的**存储空间**是不连续的。
  list 的迭代器必需具备**前移和后退功能**，所以 list 提供的是 BidirectionalIterator。list 的数据结构中只要⼀个指向 node 节点的指针就可以了。

### deque

  deque 双向开⼝的连续线性空间。⽀持从头尾两端进⾏元素的**插⼊和删除**操作。
  deque 更贴切实现了动态空间的概念。deque 没有容量的概念，因为它是动态地以**分段连续空间**组合⽽成，随时可以增加⼀段新的空间并连接起来。
  要维护这种整体连续的假象，并提供随机存取的接⼝（即也提供RandomAccessIterator），避开了“重新配置，复制，释放”的轮回，代价是**复杂的迭代器结构**。也就是说除⾮必要，我们应该**尽可能使⽤ vector，⽽不是 deque**。
  **迭代器**：缓冲区现行元素、缓冲区头、缓冲区尾、指向map
  deque 采⽤⼀块所谓的**map作为主控**，这⾥的 map 实际上就是⼀块⼤⼩连续的空间，其中每⼀个元素，我们称之为节点 node，都指向了另⼀段连续线性空间称为缓冲区，缓冲区才是 deque 的真正存储空间主体。默认大小512bytes，当满载时候，会申请一块更大的空间


### stack

  是⼀种**先进后出的数据结构**，只有⼀个出⼝，stack 允许从最顶端新增元素，移除最顶端元素，取得最顶端元素。deque 是双向开⼝的数据结构，所以**使⽤ deque 作为底部结构**并封闭其头端开⼝，就形成了⼀个 stack。

### queue

  是⼀种**先进先出的数据结构**，有两个出⼝，允许从最底端加⼊元素，取得最顶端元素，从最底端新增元素，从最顶端移除元素。deque 是双向开⼝的数据结构，若以**deque 为底部结构**并封闭其底端的出⼝，和头端的⼊⼝，就形成了⼀个 queue。（其实 list 也可以实现 deque

### heap

  堆并**不属于 STL 容器**组件，它是个幕后英雄，扮演 priority_queue 的助⼿，priority_queue 允许⽤户以任何次序将任何元素推⼊容器内，但取出时⼀定是从优先权最⾼（数值最⾼）的元素开始取。**⼤根堆**作为 priority_queue 的底层机制。

  ⼤根堆，是⼀个**满足**每个节点的键值都⼤于或等于其⼦节点键值的⼆叉树（**具体实现是⼀个vector，⼀块连续空间**，通过维护某种顺序来实现这个⼆叉树），**新加⼊元素时**，新加⼊的元素要放在最下⼀层为叶节点，即具体实现是填补在由左⾄右的第⼀个空格（即把新元素插⼊在底层 vector 的 end()），然后执⾏⼀个所谓上溯的程序：将新节点拿来与 ⽗节点（i>>1）⽐较，如果其键值⽐⽗节点⼤，就⽗⼦对换位置，如此⼀直上溯，直到不需要对换或直到根节点为⽌。当**取出⼀个元素时**，最⼤值在根节点，取⾛根节点，要割舍最下层最右边的右节点，并将其值重新安插⾄最⼤堆，最末节点放⼊根节点后(nums[1] = nums[len -1];)，进⾏⼀个下溯程序：将空间节点和其较⼤的节点对调，并持续下⽅，直到叶节点为⽌。

### priority_queue

  底层时⼀个**vector**，使⽤heap形成的**算法**，插⼊，获取 heap 中元素的算法，维护这个vector，以达到允许⽤户以任何次序将任何元素插⼊容器内，但取出时⼀定是从优先权最⾼（数值最⾼）的元素开始取的⽬的。

### slist
  slist 是⼀个单向链表。

### vector注意事项
 * 注意插⼊和删除元素后迭代器失效的问题
 * 清空 vector 数据时，如果保存的数据项是指针类型，需要逐项 delete，否则会造成内存泄

 **频繁调⽤ push_back()影响**
  * 向 vector 的尾部添加元素，很有可能引起整个对象 存储空间的᯿新分配，᯿新分配更⼤的内存，再将原数据拷⻉到新空间中，再释 放原有内存，这个过程是耗时耗⼒的，频繁调**push_back()会导致性能的下降**
  *  C++11 之后， vector 容器中添加了新的⽅法： emplace_back() ，和 push_back()⼀样的是都是在容器末尾添加⼀个新的元素进去，不同的是 **emplace_back()** 在效率上相较于 push_back() 有了⼀定的提升。
  * 内存优化主要体现在使⽤了**就地构造**（直接在容器内构造对象，不⽤拷⻉⼀个复制品再使⽤）+**强制类型转换**的⽅法来实现，在运⾏效率⽅⾯，由于省去了拷⻉构造过程，因此也有⼀定的提升。

### map 和 set

 * map 和 set 都是 C++ 的关联容器，其底层实现都是**红⿊树**（RB-Tree）⼏乎所有的 map 和 set的操作⾏为，都只是转调 RB-tree 的操作⾏为。
 * map 中的**元素**是 key-value（关键字—值）对：关键字起到索引的作⽤，值则表示与索引相关联的数据；Set与之相对就是关键字的简单集合，set 中每个元素只包含⼀个关键字。
 * set 的**迭代器**是 const 的，不允许修改元素的值；map允许修改value，但不允许修改key。其原因是因为map和set是根据关键字排序来保证其有序性的，如果允许修改key的话，那么⾸先需要删除该键，然后调节平衡，再插⼊修改后的键值，调节平衡，如此⼀来，破坏了map和set的结构，导致iterator失效，不知道应该指向改变前的位置，还是指向改变后的位置。所以STL中将set的迭代器设置成const，不允许修改迭代器的值；⽽map的迭代器则不允许修改key值，允许修改value值。
 * map⽀持**下标操作**，set不⽀持下标操作。map可以⽤key做下标，map的下标运算符[ ]将关键码作为下标去执⾏查找，如果关键码不存在，则插⼊⼀个具有该关键码和mapped_type类型默认值的元素⾄map中，因此下标运算符[ ]在map应⽤中需要慎⽤，const_map不能⽤，只希望确定某⼀个关键值是否存在⽽不希望插⼊元素时也不应该使⽤，mapped_type类型没有默认值也不应该使⽤。如果find能解决需要，**尽可能⽤find**。



### stl迭代器删除元素

 * 序列容器 vector，deque来说，使⽤ erase(itertor) 后，后边的每个元素的迭代器都会失效，但是后边每个元素都会往前移动⼀个位置，但是 **erase 会返回下⼀个有效的迭代器**；

错误做法：
```c++
for(vector<int>::iterator iter = vecInt.begin(); iter != vecInt.end(); iter++){
		if(*iter == 444){
			vecInt.erase(iter); // iter 移除后，成为野指针，无法在得到下一个地址
		}
	}
```

正确做法:
```c++
for(vector<int>::iterator iter = vecInt.begin(); iter != vecInt.end(); iter++){
		if(*iter == 444){
			iter = vecInt.erase(iter);  // 移除后，将指针返回（返回指针为下一个iterator的值）
			iter--;
		}
	}
```
 * 对于关联容器 map set 来说，使⽤了 erase(iterator) 后，当前元素的迭代器失效，但是其结构是红⿊树，删除当前元素的，不会影响到下⼀个元素的迭代器，所以在调⽤ erase 之前，**记录下⼀个元素的迭代器**即可。

```c++
if (it != myMap.end()) {
        auto next = std::next(it);  // 记录下一个元素的迭代器
        // 操作next iterator

        myMap.erase(it);
}
```

 * 对于 list 来说，它使⽤了不连续分配的内存，并且它的 erase ⽅法也会返回下⼀个有效的iterator，因此**上⾯两种正确的⽅法**都可以使⽤。

### 迭代器和指针

 * 迭代器
   * 提供⼀种**方法**顺序访问⼀个聚合对象中各个元素, ⽽⼜不需暴露该对象的内部表示。过运⽤该模式，使得我们可以在不知道对象内部表示的情况下，按照⼀定顺序（由iterator提供的⽅法）访问聚合对象中的各个元素。
   * 是**封装**了原⽣指针，根据不同类型的数据结构来实现不同的++，--等操作

### STL ⾥ resize 和 reserve 的区别
 * resize()：改变当前容器内含有元素的数目(size())，值为默认为0.当v.push_back(3);之后，则是3是放在了v的末尾，即下标为len，此时容器是size为len+1；
 * reserve()：改变当前容器的最⼤容量（capacity）,它不会⽣成元素，只是确定这个容器允许放⼊多少对象，如果reserve(len)的值⼤于当前的capacity()，那么重新新分配⼀块能存len个的空间，然后把之前v.size()个对象通过 copy construtor 复制过来，销毁之前的内存；

## 二叉树

  ⼀个⼆叉树如果不为空，便是由⼀个根节点和左右两个⼦树构成，左右⼦树都可能为空。

## 二叉搜索树

  ⼆叉搜索树：⼆叉搜索树可以提供对数时间的元素插⼊和访问。
  节点的放置规则是：任何节点的键值⼀定⼤于其左⼦树的每⼀个节点的键值，并⼩于其右⼦树中的每⼀个节点的键值。
  **插⼊**：从根节点开始，遇键值较⼤则向左，遇键值较⼩则向右，直到尾端，即插⼊点。
  **删除**：如果删除点只有⼀个⼦节点，则直接将其⼦节点连⾄⽗节点。如果删除点有两个⼦节点，以右⼦树中的最⼩值代替要删除的位置。

## 平衡二叉树

  “平衡”的⼤致意思是：没有任何⼀个节点过深，不同的平衡条件会造就出不同的效率表现。以及不同的实现复杂度。有数种特殊结构例如 AVL-tree, RB-tree, AA-tree，均可以实现平衡⼆叉树。

## 严格的平衡二叉树（AVL）

  AVL-tree 是要求任何节点的左右⼦树⾼度相差最多为 1 的平衡⼆叉树。
  当插⼊新的节点破坏平衡性的时候，从下往上找到第⼀个不平衡点，需要进⾏单旋转，或者双旋转进⾏调整。


## 红黑树（RB-Tree）

  要求：
  性质1：每个节点要么是⿊⾊，要么是红⾊。
  性质2：根节点是⿊⾊。
  性质3：每个叶⼦节点（NIL）是⿊⾊。
  性质4：每个红⾊结点的两个⼦结点⼀定都是⿊⾊。
  性质5：任意⼀结点到每个叶⼦结点的路径都包含数目相同的⿊结点。



