---
layout: article
title: 数据结构与算法
key: 100019
tags: 算法
category: blog
date: 2023-02-4 00:00:00 +08:00
mermaid: true
---

# memcpy
 * 完成内存的拷贝
   * 如果内存有重叠（对小端），则从高位到低位复制
   * 如果内存无重叠，则从低位到高位传递


  ```c++
  void* my_memcpy(void *dst, void *src, size_t count){
    if(dst == nullptr || src == nullptr){
      return nullptr;
    }
    char* temp_dst = (char*) dst;
    char* temp_src = (char*) src;
    if(temp_dst > temp_src && temp_dst < temp_src+ count){
      // 内存重叠
      temp_dst = temp_dst + count -1;
      temp_src = temp_src + count -1;
      while(count--){
        *temp_dst-- = *temp_src--;
      }
    }
    else {
      while(count--){
        *temp_dst++ = *temp_src++;
      }
    }
    return (void * )dst;
  }
  ```

<!--more-->

# 排序算法
 排序算法比较
 ![](https://cdn.jsdelivr.net/gh/luogou/cloudimg/data/20210828133536.png)

## 插入排序
### 直接插入排序：复杂度O(n^2)


![](https://cdn.jsdelivr.net/gh/luogou/cloudimg/data/20210828142217.gif)
 
 原理

 * 将无序元素一个一个插入到有序部分中
   * 对于待排序元素`temp = index.1:length-1`，寻找其左侧有序序列`index.0:temp-1`的插入位置
   * 对index从 temp-1 到 0 过程中
     * temp > index值
       * a[index+1] = temp
     * temp < index值（如果没有到头部） 
       * 在下一次处理(a[index+1] = a[index])
     * temp < index值（在头部）
       * 本次处理 a[index] = temp;

 伪代码

  ```c
  // 待排序数组
  for(index_1 := 1:length-1):
   // 将待排序元素取出
    temp = a[index_1]
    // 排序位置index
    for(index_2 := index_1:0):
    // for循环执行之后的下一次处理
      a[index_2+1] = a[index_2];
      // 如果可以放入
      if(temp > a[index_2]){
        a[index_2 + 1] = temp;
      }
      // 对于头部的处理
      if(temp < a[index_2] && index_2 ==0){
        a[index_2] = temp;
      }

      
  ```

 代码
  
  ```c++
  void straight_inserting_sort(int a[], int length){
    // 对所有待排序元素来说
    for(int i = 1; i <length; i++){
      if(a[i] < a[i-1]){
        int temp = a[i];
        // 确定排序位置ptr
        for(int j = i - 1; j >=0 ; j--){
          a[j + 1] = a[j];
          if(a[j] < temp){
            a[j+1] = temp;
            break;
          }
          if(a[j] > temp && j == 0 ){
            a[j] = temp;
          }
        }
      }
    }
  }
  ```

### 折半插入排序

![](https://cdn.jsdelivr.net/gh/luogou/cloudimg/data/20210828142548.png)


 * 将右侧数组的元素逐个插入到左侧排好序的数组
 * 确定排好序数组的ordered_index := 0:index-1
 * 利用二分搜索查找插入点index+1 >temp
 * 将index右侧所有值均右移一位
 * 将值赋值到index+1位置


 伪码
  ```c
  for(index := 1:length-1){
    temp = a[index];
    left = 0, right = index -1
    while(left <= right){
      mid = left + (right-left) /2;
      if(a[mid] > temp){
        right = mid-1
      }
      else{
        left = mid+1;
      }
      
    }
    for(right右侧){
      全部右移一位
    }
    值赋值
  }
  ```


  ```c++
  void binary_insert_sout(int a[], int length){
    int low, high, mid;
    for(int i = 1 ; i <length ; i++){
      low = 0;
      high = i - 1;
      temp = a[i];
      while(high >=low){
        mid = low+ (high - low) /2;
        if(temp < a[mid]){
          high = mid -1;
        }
        else {
          low = mid +1;
        }
      }
      for(int j =i - 1 ; j > high ; j--){
        a[j+1] = a[j];
      }
      a[j+1] = temp;
    }
  }
  ```

### 冒泡排序：O(n^2)

![](https://cdn.jsdelivr.net/gh/luogou/cloudimg/data/20210829093855.gif)

 * 冒泡排序每次都会将最大值放在数组末尾


  ```c++
  void bubble_sort(int a[], int length){
    for(int i = 0; i  <length -1; i++){
      for(int j = 0; j < length - i-1; j++){
        if(a[j] >a[j+1]){
          int temp = a[j];
          a[j] = a[j+1];
          a[j]+1 = temp;
        }

      }
    }
  }
  ```

## 选择排序：O(n^2)

![](https://cdn.jsdelivr.net/gh/luogou/cloudimg/data/20210829105353.png)

  ```c++
  void select_sort(int a[], int length){
    for(int i = 0; i <length-1; i++){
      int min_index = i;
      for(int j = i+1; j <length; j++){
        if(a[min_index] >a[j]){
          min_index = j;
        }
      }
      if(i != min_index){
        int temp = a[i];
        a[i] = a[min_index];
        a[min_index] = temp;
      }
    }
  }
  ```

## 希尔排序：O(nlogn)


![](https://cdn.jsdelivr.net/gh/luogou/cloudimg/data/20210829090937.png)

  ```c++
  void shell_sort(int a[], int length){
    int i, j, gap;
    for(gap = length/2; gap >0; gap /=2){
      for(i = 0; i < gap; i++){
        for(j = i + gap; j < length; j += gap){
          int temp = a[j];
          k = j - gap;
          while(k >=0 && a[k] > temp){
            a[k + gap] = a[k];
            k -= gap; 
          }
          a[k + gap] = temp;
        }
      }
    }
  }
  ```

## 快速排序：O(nlogn)


![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%92%E5%BA%8F/6-8.png)

  ```c++
  void quick_sort(int[] a, int low, int high){
    if(low < high){
      int key = quick_sort_index(a, low, high);
      quick_sort(a, low, key-1);
      quick_sort(a, key+1, high);
    }
  }

  int quick_sort_index(int[] a, int low, int high){
    int temp = a[low];
    int rightvalue;
    int leftvalue;
    while(low < high){
      while(low < high && a[high] >= temp){
        high--;
      }
      a[low] = a[high];
      while(low < high && a[low] <=temp){
        low++;
      }
      a[high] = a[low];
    }
    a[low] = temp;
    return low;
  

  }
  ```

## 堆排序：O(nlogn)

 步骤
  * 无序序列建立完全二叉树


  ![](https://images2018.cnblogs.com/blog/1307402/201804/1307402-20180407155121364-1143663369.png)
  * 从最后一个叶子节点开始，从左到右，从下到上调整，将完全二叉树调整为大根堆
    * 找到第1个非叶子节点6，由于6的右子节点9比6大，所以交换6和9。交换后，符合大根堆的结构

  ![](https://images2018.cnblogs.com/blog/1307402/201804/1307402-20180407155122347-353144474.png)
    * 找到第2个非叶子节点4，由于的4左子节点9比4大，所以交换4和9。交换后不符合大根堆的结构，继续从右到左，从下到上调整。

  ![](https://images2018.cnblogs.com/blog/1307402/201804/1307402-20180407155123881-1600164453.png)

  ![](https://images2018.cnblogs.com/blog/1307402/201804/1307402-20180407155124922-114571381.png)
  * 交换堆元素（交换堆首和堆尾元素--获得最大元素）


  ![](https://images2018.cnblogs.com/blog/1307402/201804/1307402-20180407155125504-881684214.png)
  * 重建大根堆（前n-1个元素）
  
  ![](https://images2018.cnblogs.com/blog/1307402/201804/1307402-20180407155126551-236420319.png)

  * 重复执行步骤二和步骤三，直到整个序列有序

  ![](https://images2018.cnblogs.com/blog/1307402/201804/1307402-20180407155127513-1825791452.png)

  ```c++

  void heap_sort(vector<int>& arr, int length){
    for(int i = length/2 -1; i>=0 ; i++){
      adjust(arr, length, i);
    }
    for(int i = size -1; i>=0 ; i--){
      swap(arr[0], arr[i]);
      adjust(arr, i, 0);
    }
  }

  void adjust(vector<int>& arr, int length, int index){
    int left = 2*index + 1;
    int right = 2*index +2;
    int maxindex = index;
    if(left < length && arr[left] > arr[maxindex])  maxindex = left;
    if(right< length && arr[right] > arr[maxindex]) maxindex = right;

    if(maxindex != index){
      swap(arr[maxindex], arr[index]);
      adjust(arr, length, maxindex);
    }
  }
  
  void swap (int value1, int value2){
    int temp = value1;
    value1 = value2;
    value2 = temp;
  }
  ```



## 归并排序：O(nlogn)


![](https://cdn.jsdelivr.net/gh/luogou/cloudimg/data/20210829134401.jpeg)

  ```c++
  void merge_sort(int[] a, int low, int high){
    if(low < high){
      int mid = low + (high - low) / 2;
      merge_sort(a, low, mid);
      merge_sort(a, mid+1, high);

      merge(a, low, mid, high);
    }

  void merge(int[] a, int low, int mid, int high){
    int i = low;
    int j = mid +1;
    k = 0;
    int* temp = new[high - low - 1];
    while(i <=mid && j <= high){
      if(a[i] <= a[j]){
        temp[k++] = a[i++];
      }
      else {
        temp[k++] = a[j++];
      }
    }
    while(i<=mid){
      temp[k++] = a[i++];
    }
    while(j <= high){
      temp[k++] = a[j++];
    }
    for(i = low, k = 0; i<= high; i++,k++){
      a[i] = temp[i];
    }
    delete[] temp;
  }
  }
  ```


## 计数排序O(n+k)

![](https://cdn.jsdelivr.net/gh/luogou/cloudimg/data/20210829135544.gif)

  ```c++
  void count_sort(int[] a, int length){
    int max = a[0];
    int i = 0;

    while( i< length -1){
      max = (a[i] > a[i+1]) ? a[i]: a[i+1];
      i++;
    }
    int* countArray = new int[max+1]{0};
    int* temp = new int[length];

    for(int i = 0; i < length; i++){
      countArray[a[i]]++;
    }
    // 特别注意此方法实现，可以减少复杂度
    for(int i = 1; i < length + 1; i++){
      countArray[i] += countArray[i-1];
    }
    // 反向遍历
    for(int i = length-1; i >= 0 ; i--){
      temp[countArray[a[i]]-1] = a[i];
      countArray[a[i]]--;
    }

    for(int i = 0 ; i < length; i++){
      a[i] = temp[i];
    }
    delete[] countArray;
    delete[] temp;
  }
  ```


## 基数排序O(n*k)

![](https://cdn.jsdelivr.net/gh/luogou/cloudimg/data/20210829140105.gif)


  ```c++
  int get_max_digits(int[] a, int length){
    int max = a[0];
    int i = 0;
    while(i < length-1){
      max = (a[i] > a[i+1]) ? a[i]: a[i+1];
    }
    int b =0;
    while(max > 0){
      b++;
      max /= 10;
    }
    return b;
  }

  void sort(int[] a, int length){
    int d = get_max_digits(a, length);
    int* temp = new int[length];
    
    int padding = 1;
    for(int i = 0; i < d; i++){
      int count[10]={0};
      
      for(int j = 0; j <length; j++){
        int tail_number = (a[j]/padding) % 10;
        count[tail_number]++;
      }

      for(j = 1; j <10; j++){
        count[j] += count[j-1];
      }

      for(int j = length-1; j >=0; j--){
        int tail_number = (a[j] / padding) % 10;
        temp[count[tail_number] - 1] = a[j];
        count[tail_number]--;
      }
      for(int j = 0; j < length;j++){
        b[j] = temp[j];
      }
      radix *= 10;
    }
    delete[] temp;
  }
  ```

## 桶排序：O(n+k)

![](https://cdn.jsdelivr.net/gh/luogou/cloudimg/data/20210829144129.png)

  ```c++
  void bucket_sort(int[] a, int length){
    int max = INT_MIN;
    int min = INT_MAX;

    for(int i = 0; i <length; i++){
      if(a[i] > max) max = a[i];
      if(a[i] < min) min = a[i];
    }

    int bucket_len = max - min +1;
    int bucket[bucket_len];
    for(int i = 0; i < bucket_len; i++){
      bucket[i] = 0;
    }

    int index = 0;
    for(int i = 0 ; i < length ; i++){
      index = arr[i] - min;
      bucket[index]++;
    }
    int start = 0;
    for(int i = 0; i< bucket_len; i++){
      for(int j = start; j < start + bucket[i]; j++){
        a[j] = min + i;
      }
      start += bucket[i];
    }
  }
  ```

# 查找算法

## 顺序查找

 * 适合线性表
   * 存储结构为数组或链表
 * 时间复杂度O(n)

  ```c++
  int seq_search(int[] a, int key, int n){
    for(int index = 0; index <n; index++){
      if(a[index] == key){
        return index;
      }
    }
    return -1;
  }
  ```

 带哨兵的优化查找方法

  ```c++
  int seq_search(int[] a, int key, int n){
    int i;
    
    a[0] = key;
    
    for(i = n; a[0] != a[i]; i--){

    }
    return i;
  }
  ```

## 二分查找

 * 要求：元素必须有序
 * 时间复杂度O(logn)

  ```c++
  int binary_search(int[] a, int key, int n){
    int left = 0;
    int right = n - 1;
    int mid;
    while(left <= right){
      mid = left + (right - left) / 2;
      if(a[mid] == key){
        return mid;
      }
      else if (a[mid] > key){
        right = mid - 1;
      }
      else{
        left + mid + 1;
      }
    }
    return -1;
  }
  ```
 递归方法

  ```c++
  int binary_search(int[] a, int key, int left, int right){
    int mid = left + (right - left) / 2;
    if(a[mid] == key){
      return mid;
    }
    else if (a[mid] > key){
      binary_search(a, key, left, mid - 1);
    }
    else{
      binary_search(a, key, mid + 1, right);
    }
  }
  ```


## 插值查找

 * 在二分查找中每次都是从中间走，自主性较差
   * 对于插值查找其中`mid = low + (key - a[low])/(a[high]- a[low]) *(high - low)`来自适应选择


  ```c++
  int insert_search(int[] a, int key, int left, int right){
    int mid = left + (right - left)* (key - a[left])/(a[right] - a[left]); 
    if(a[mid] == key){
      return mid;
    }
    else if (a[mid] > key){
      binary_search(a, key, left, mid - 1);
    }
    else{
      binary_search(a, key, mid + 1, right);
    }
  }
  ```

## 二叉查找树

 * 性质
   * 当某节点左子树不为空，则左子树上所有节点均小于根节点
   * 当某节点右子树不为空，则右子树上所有节点均大于根节点
   * 任意节点的左右子树也分别为二叉搜索树

 * 复杂度
   * 插入和查找时间复杂度均为O(logn)，最坏情况下为O(n)

 二叉树查找

  ```c++
  BSTree* search(BSTree pTree, int key){
    if(pTree == nullptr || pTree->data == key){
      return pTree;
    }
    if(pTree->data > key){
      return BSTree(pTree->left, key);
    }
    if(pTree->data < key){
      return BSTree(pTree->right, key);
    }
  } 
  ```

## 红黑树

## B Tree 和 B+ Tree

## 分块查找

 * 把元素放在不同块，每个块内部元素不必有序，但是块之间有序
 * 先用二分查找查找在哪个块，然后在进行顺序查找

  ```c++
  typedef struct{
    int key;
    int start;
    int stop;
  }Node;

  typedef struct{
    Node idx[10];
    int len;
  }IdxTable;

  IdxTable table;

  int block_search(int[] a, int key){
    int low = 1;
    int high = table.len;
    int mid;
    while(low <= high){
      mid = low + (high - low) / 2;
      if(key <= table.idx[mid].key){
        if(key <= table.idx[mid-1].key){
          high = mid -1;
        }
        else{
          for(int i = table.idx[mid].start; i <=table.idx[mid].end; i++){
            if(key == a[i]){
              return (i+1);
            }
          }
          return -1;
        }
      }
      else{
        low = mid + 1;
      }
    }
    return -1;
  }
  ```

## 哈希查找

 * key-value键值对查找，是一种时间换空间的查找方法


# 非递归遍历二叉树

## 先序遍历
  ```c++
  void pre_traverse(BinaryTree* T){}
    if(T == nullptr){
      return;
    }
    printf(T->data);
    pre_traverse(T->left);
    pre_traverse(T->right);
  ```
## 中序遍历
  ```c++
  void mid_traverse(BinaryTree* T){}
    if(T == nullptr){
      return;
    }
    pre_traverse(T->left);
    printf(T->data);
    pre_traverse(T->right);
  ```


## 后序遍历
  ```c++
  void mid_traverse(BinaryTree* T){}
    if(T == nullptr){
      return;
    }
    pre_traverse(T->left);
    pre_traverse(T->right);
    printf(T->data);
  ```

## 层序遍历

```c++
void layer_traverse(TreeNode* Tree){
  queue<TreeNode*> s;
  if(Tree == nullptr){
    return;
  }
  s.push(Tree);
  while(!s.empty()){
    TreeNode* cur = s.front();
    cout << cur->val;
    if(cur->left != nullptr){
      s.push(cur->left);
    }
    if(cur->right != nullptr){
      s.push(cur->right);
    }
    s.pop();
  }
}
```

# KMP算法

 [KMP算法](https://blog.csdn.net/ooblack/article/details/109329361)
 传统的字符串匹配是通过暴力算法得到，为了减少匹配次数，对待匹配子串建立next数组通过next数组确定跳转位置

 next数组是pattern串的属性，`next[i]`表示字符串前i+1位前缀和后缀相同的位数
  * 如`abcabd`的`next[0] =0`、`next[1]`为前两位ab，前后缀无相同项，`next[3]`表示abca，其中前缀a和后缀a相同，`next[3] = 1`，最终`next = [0,0,0,1,2,0]`

  ```c++
    void get_next(string a, int* next){
      next[0] = 0;
      for(int i = 1; i < a.size(); i++){
        while(j > 0 && a[i] != a[j]){
          j = next[j-1];
        }
        if(a[i] == a[j]){
          j++;
        }
        next[i] = j;

      }
    }
    int kmp(string cur, string pattern){
      int length = pattern.size();
      int* next = int[length];
      get_next(pattern, next);
      for(int i = 0, j = 0; i < cur.size(); i++){
          while(j >0 && a[i] != a[j]){
            j = next[j-1];
          }

        if(s[i] == s[j]){
          j++;
        }
        if(j == pattern.size()) return i - j + 1;
        if(i = cur.size()) return -1;
      }
    }
  ```


# 最小生成树——Prim
 一些宏
  ```c++
  const int MAXN = 1000, INF = 0x3f3f3f3f; // 定义一个INF表示无穷大
  int g[MAXN][MAXN], dist[MAXN],n,m,res;
  // 使用g[][]数组存储图，dist[]存储到集合S的距离，res保存结果
  bool book[MAXN];
  // 存储某个点是否保存到集合S
  ```

 main
  ```c++
  int main()
  {
      cin>>n>>m;  //读入图点数 n 和边数 m

      for(int i = 1; i<= n; i++)
      {
          for(int j = 1; j <= n; j++)
          {
              g[i][j] = INF;  //初始化任意两个点之间的距离为正无穷（表示这两个点之间没有边）
          }
          dist[i] = INF;  //初始化所有点到集合S的距离都是正无穷
      }
      
      for(int i = 1; i <= m; i++)
      {
          int a, b, w;
          cin >> a >> b >> w;  //读入a，b两个点之间的边
          g[a][b] = g[b][a] = w;  //由于是无向边，我们对g[a][b]和g[b][a]都要赋值
      }

      prim();  //调用prim函数
      if(res==INF)  //如果res的值是正无穷，表示不能该图不能转化成一棵树，输出orz
          cout<<"orz";
      else
          cout<<res;//否则就输出结果res
      return 0;
  }
  ```


 prim实现
  ```c++
  void prim(){
    dist[1] = 0;
    book[1] = true;
    for(int i = 2; i <= n; i++){
      dist[i] = min(dist[i], g[1][i]);
    }
    for(int i = 2; i <= n ;i++){
      int temp = INF;
      int index = -1;
      for(int j = 2; j <= n; j++){
        if(!book[j] && dist[j] < temp){
          temp = dist[j];
          index = j;
        }
      }
      if(index == -1){
        res = INF;
        return;
      }
      book[index] = true;
      res += dist[index];
      for(int j = 2; j <= n; j++){
        dist[j] = min(dist[j], g[index][j]);
      }
    }
  }
  ```
# 最小生成树 Kruskal

 类型定义
  ```c++
#define MAXEDGE 100
#define MAXVERTEX 100

  typedef struct Edge{
    int begin;
    int end;
    int weight;
  } Edge;

  typedef struct Graph{
    char vertex[MAXVERTEX];
    Edge edges[MAXEDGE];
    int numvertex, numedges;
  } MGraph;

  ```

 main

  ```c++
  int main(){
    MGraph G;
    CreateGraph(&G);
    Kruskal(&G);
    
    return 0;
  }

  ```
 CreateGraph
  ```c++
  void CreateGraph(MGraph* G){
    scanf("%d%d",&G->numvertex, &G->numedges);
    for(int i = 0; i < G->numvertex; i++){
      scanf("%c",&G->vertex[i]);
    }
    for(int k = 0; k < G->numedges; k++){
      Edge edge;
      scanf("%d%d%d",&edge->begin,&edge->end, &edge->weight);
      G->edges[k] = e;
    }
  }
  ```

 Kruskal算法
  * 重要的是其中的find函数用于判断是否有环，parent实际内容指向下一个节点

  ```c++
  void Kruskal(MGraph* G){
    int parent[MAXVERTEX];
    for(int i = 0; i <G->numvertex; i++){
      parent[i] = 0;
    }
    for(int i = 0 ;i < G->numedges; i++){
      int m = find(parent, G, G->edges[i].begin);
      int n = find(parent, G, G->edges[i].end);
      if(m != n){
        parent[m] = n;
        printf("%d%d%d",G->edges[i].begin,G->edges[i].end,G->edges[i].weight);
      }
    }
  }

  int find(int* parent, int edgepo){
    while(parent[edgepo] >0){
      edgepo = parent[edgepo];
    }
    return edgepo;
  }
  ```


# AVL树

## 树的定义
### 节点定义
  ```c++
  template<class T>
  class AVLTreeNode{
    public:
      T key;
      int height;
      AVLTreeNode *left;
      AVLTreeNode *right;
      AVLTreeNode(T value, AVLTreeNode *l, AVLTreeNode *r): key(value), left(l), right(r){}
  }
  ```

### 树的类
```c++
template<class T>
class AVLTree{
  private:
    AVLTreeNode<T>* root;
  public:

    AVLTree();
    ~AVLTree();

    int height();
    int max(int a, int b);

    void preOrder();
    void inOrder();
    void postOrder();

    AVLTreeNode<T>* search(T key);
    AVLTreeNode<T>* iterativeSearch(T key);
    T minimum();
    T maximum();
    void insert(T key);
    void remove(T key);
    void destroy();
    void print();
  private:
    // 内部接口 
    // 获取树的高度
    int height(AVLTreeNode<T> *tree);
        
    // 前序遍历
    void preOrder(AVLTreeNode<T> *tree) const;
        // 中序遍历
    void inOrder(AVLTreeNode<T> *tree) const;
        // 后序遍历
    void postOrder(AVLTreeNode<T> *tree) const;
        
    // （递归实现）查找AVL树中键值为key的结点
    AVLTreeNode<T>* search(AVLTreeNode<T> *x, T key) const;
    // （非递归实现）查找AVL树中键值为key的结点
    AVLTreeNode<T>* iterativeSearch(AVLTreeNode<T> *x, T key) const;

    // 返回最小结点 
    AVLTreeNode<T>* minimum(AVLTreeNode<T> *tree);
    // 返回最大结点 
    AVLTreeNode<T>* maximum(AVLTreeNode<T> *tree);
        
    // 将结点插入到AVL树中
    AVLTreeNode<T>* insert(AVLTreeNode<T>* &tree, T key);
    // 删除结点，并返回被删除的结点 
    AVLTreeNode<T>* remove(AVLTreeNode<T>* &tree, AVLTreeNode<T> *z);
        
    // 销毁AVL树
    void destroy(AVLTreeNode<T>* &tree);
        
    // 打印AVL树
    void print(AVLTreeNode<T> *tree,T key,int direction);
        
    // LL：左左对应的情况(左单旋转)
    AVLTreeNode<T>* leftLeftRotation(AVLTreeNode<T> *k2);
    // RR：右右对应的情况(右单旋转)
    AVLTreeNode<T>* rightRightRotation(AVLTreeNode<T> *k1);
    // LR：左右对应的情况(左双旋转)
    AVLTreeNode<T>* leftRightRotation(AVLTreeNode<T> *k3);
    // RL：右左对应的情况(右双旋转)
    AVLTreeNode<T>* rightLeftRotation(AVLTreeNode<T> *k1);    
};

```
### 树高度的实现

  ```c++
  template<class T>
  int AVLTree<T>::height(AVLTreeNode<T> *tree){
    // 不为null，返回树最大长度
    if(tree != nullptr){
      return tree->height;
    }
    // 为空树，则返回0
    return 0;
  }

  template<class T>
  int AVLTree<T>::height(){
    return height(root);
  }
  ```

### 旋转的实现

#### LL的单旋转

 当插入或删除节点后，**左子树仍然还有左节点**，并导致AVL不平衡

![](http://images.cnitblog.com/i/497634/201403/281626153129361.jpg)


  ```c++
  template<class T>
  AVLTreeNode<T>* leftLeftRotation(AVLTreeNode<T> *k2){
    AVLTreeNode<T> * k1;
    k1 = k2->left;
    k2->left = k1->right;
    k1->right = k2;
    k2->height = max(height(k2->left), height(k2->right)) + 1;
    k1->height = max(height(k1->left), height(k1->right)) + 1;
  }
  ```

#### RR的单旋转

  当插入或删除节点后，**右子树仍然还有右节点**，并导致AVL不平衡

![](http://images.cnitblog.com/i/497634/201403/281626410316969.jpg)


  ```c++
  template<class T>
  AVLTreeNode<T>* rightRightRotation(AVLTreeNode<T> *k1){
    AVLTreeNode<T>* k2;
    k2 = k1->right;
    k1->right = k2->left;
    k2->left = k1;

    k1->height = max(height(k1->left), height(k1->right)) + 1;
    k2->height = max(height(k2->left), height(k2->right)) + 1;
  }
  ```
#### LR的双旋转

 当插入或删除节点后，**左子树仍然还有右节点**，并导致AVL不平衡
![](http://images.cnitblog.com/i/497634/201403/281627088127150.jpg)

  ```c++
  template<class T>
  AVLTreeNode<T>* leftRightRotation(AVLTreeNode<T> *k3){
    k3->left = rightRightRotation(k3->left);
    return leftLeftRotation(k3);
  }

  ```

####  RL的双旋转
 当插入或删除节点后，**右子树仍然还有左节点**，并导致AVL不平衡
![](http://images.cnitblog.com/i/497634/201403/281628118447060.jpg)


  ```c++
  template<class T>
  AVLTreeNode<T>* rightLeftRotation(AVLTreeNode<T> *k3){
    k3->right = leftLeftRotation(k3->right);
    return rightRightRotation(k3);
  }
  ```


### 插入的实现

  ```c++
  template<class T>
  AVLTreeNode<T>* insert(AVLTreeNode<T>* &tree, T key){
    if(tree == nullptr){
      tree = new AVLTreeNode<T>(key, NULL, NULL); 
    }
    else if(key < tree->key){
      tree->left = insert(tree->left, key);
      // 判断是否失去平衡
      if(key < tree->left->key){
        // 比左节点小，成为左节点的左节点
        leftLeftRotation(tree);
      }else{
        // 比左节点大，成为左节点的右节点，需要LR
        leftRightRotation(tree);
      }

    }
    else{
      tree->right = insert(tree->right, key);
      if(key > tree->right->key){
        // 比右节点大，成为右节点的右节点
        rightRightRotation(tree);
      }else{
        // 比右节点小，成为右节点的左节点，需要RL
        rightLeftRotation(tree);
      }
    }
  }

  ```


### 删除的实现
 删除节点在叶子节点
  ![](https://img-blog.csdnimg.cn/20200418172744475.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxMzg4NTM1,size_16,color_FFFFFF,t_70)
  ![](https://img-blog.csdnimg.cn/20200418173233362.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxMzg4NTM1,size_16,color_FFFFFF,t_70)
  ![](https://img-blog.csdnimg.cn/20200418173550484.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxMzg4NTM1,size_16,color_FFFFFF,t_70)
  ![](https://img-blog.csdnimg.cn/20200418173849705.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxMzg4NTM1,size_16,color_FFFFFF,t_70)

 删除只有一个左节点或右节点
  ![](https://img-blog.csdnimg.cn/20200418174425322.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxMzg4NTM1,size_16,color_FFFFFF,t_70)
  ![](https://img-blog.csdnimg.cn/2020041817570775.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxMzg4NTM1,size_16,color_FFFFFF,t_70)

 删除有左右节点
  ![](https://img-blog.csdnimg.cn/20200418180925175.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxMzg4NTM1,size_16,color_FFFFFF,t_70)
  ![](https://img-blog.csdnimg.cn/20200418181649249.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxMzg4NTM1,size_16,color_FFFFFF,t_70)
  ![](https://img-blog.csdnimg.cn/20200418182253185.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxMzg4NTM1,size_16,color_FFFFFF,t_70)

  ```c++
  AVLTreeNode<T>* remove(AVLTreeNode<T>* &tree, AVLTreeNode<T> *z){
    if(tree == nullptr || z == nullptr){
      return nullptr;
    }
    if(z->key < tree->key){
      AVLTreeNode<T>* r = tree->right;
      tree->left = remove(tree->left, z);
      // 需要进行RL旋转
      if(r->left > r->right){
        tree = rightLeftRotation(tree);
      }else {
        tree = rightRightRotation(tree);
      }
    }
    else if(z->key > tree->key){
      AVLTreeNode<T>* l = tree->left;
      tree->right = remove(tree->right, z);
      // 需要进行RL旋转
      if(r->right > r->left){
        tree = keftRightRotation(tree);
      }else {
        tree = leftLeftRotation(tree);
      }
    }
    else{
      // 需要删除的节点
      if(tree->left !=null) && (tree->right != null){
        if(height(tree->left) > height(tree->right)){
          AVLTreeNode<T>* temp = maximum(tree->left);
          tree->key = temp->key;
          remove(tree->left, temp);
        }
        else{
          AVLTreeNode<T>* temp = minimum(tree->right);
          tree->key = temp->key;
          remove(tree->right, temp);
        }
      }
      else{
        AVLTreeNode<T>* temp = tree;
        tree = (tree->left !=null) ? tree->left:tree->right;
        delete temp;
      }
    }
    return tree;
  }

  template<class T>
  void AVLTree<T>::remove(T key)
  {
      AVLTreeNode<T> *z;
      if((z=search(root,key))!=NULL)
          root = remove(root,z);
  }

  ```

# 字典树



 * 根节点不包含字符，除根节点外每一个节点都只包含一个字符。
 * 从根节点到某一节点，路径上经过的字符连接起来，为该节点对应的字符串。
 * 每个节点的所有子节点包含的字符都不相同。


针对单词为：b，abc，abd，bcd，abcd，efg，hii得到的字典树
![](https://img-blog.csdnimg.cn/20190408163008821.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01PVV9JVA==,size_16,color_FFFFFF,t_70)


## 字典树宏

```c++
const int Num = 26;
```

## 字典树节点定义

```c++
struct TrieNode{
  bool is_word;
  TrieNode* next[Num];
  TrieNode() : is_word(flase){
    memset(next, NULL, sizeof(next));
  }
};
```

## 字典树定义

```c++
class Trie{
public: 
  Trie(){root = new TrieNode();}
  void insert(string word);
  bool search(string word);
  void deleteTrie(TrieNode* root);

private:
  TrieNode* root;
}
```
## 字典树方法

### 字典树插入方法

```c++
void Trie::insert(string word){
  TrieNode* location = root;
  for(int i = 0; i <word.size(); i++){
    if(location->next[word[i]- 'a'] == nullptr){
      TrieNode* temp = new TrieNode();
      location->next[word[i]- 'a'] = temp;
    }
    location = location->next[word[i] - 'a'];
  }
  location->is_word = true;
}
```

### 字典树寻找方法

```c++
bool Trie::search(string word){
  TrieNode* location = root;
  for(int i = 0 ; i < word.size() && location; i++){
    location = location->next[word[i] - 'a'];
  }
  return (location!= NULL && location->is_word);
}
```

### 字典树删除方法

```c++
  void Trie::deleteTrie(TrieNode* root){
    for(int i = 0 ; i < Num; i++){
      if(root->next[i] != NULL){
        deleteTrie(root->next);
      }
    }
    delete root;
  }
```

# 234树


# 红黑树

## 红黑树宏
```c++
enum Colour{
  RED,
  BLACK
} Color;
```

## 节点定义


```c++
template<typename Type>
struct RBTNode{
  Color color;
  Type key;
  RBTNode* left;
  RBTNode* right;
  RBTNode* parent;
};
```
## 红黑树定义

```c++
template<typename Type>
class RBTree{

public:
  RBTree(){
    Nil = BuyNode();
    root = Nil;
    Nil->color = BLACK;
  }

  ~RBTree(){
    destroy(root);
    delete Nil;
    Nil = NULL;
  }
  void InOrder(){InOrder(root);}
  bool Insert(const Type& value);
  void Remove(Type key);
  void InOrderPrint(InOrderPrint(root);)

protected:
  RBTNode<Type>* BuyNode(const Type& x = Type());
  void InOrder(RBTNode<Type>* root);
  void LeftRotate(RBTNode<Type>* z);
  void RightRotate(RBTNode<Type>* z);
  void Insert_fixup(RBTNode<Type>* s);
  RBTNode<Type>* search(RBTNode<Type>* root, Type key) const;

private:
  RBTNode<Type>* root;
  RBTNode<Type>* Nil;
}
```

## 红黑树实现

### 红黑树插入

```c++
bool RBTree::Insert(const Type& value){
  
}
```


# B-Tree

# B+Tree

# LRU

# 
