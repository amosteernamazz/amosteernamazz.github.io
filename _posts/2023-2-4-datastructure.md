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

  ```c++

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
  count_sort(int[] a, int length){
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

## 中序遍历

## 后序遍历


# KMP算法

# 最小生成树——Prim & Kruskal

# AVL树

# 字典树

# 2-3查找树

# 红黑树

# B-Tree

# B+Tree

# LRU

# 