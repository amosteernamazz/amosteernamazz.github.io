---
layout: article
title: 力扣
key: 100027
tags: leedcode
category: blog
date: 2023-02-10 00:00:00 +08:00
mermaid: true
---



***92***

***25***



***35***















#### [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

![](https://assets.leetcode.com/uploads/2021/03/03/pal1linked-list.jpg)

```c++
class Solution {
  public:
    bool isPalindrome(ListNode* head) {
      stack<int> rec;
      ListNode* temp = head;
      while(temp){
        rec.push(temp->val);
        temp = temp -> next;
      }
      while(!rec.empty() || head){
        if(head-> val == rec.top()){
          head = head->next;
          rec.pop(); 
        }
        else {
          return false;
        }
      }
      return true;
    }
}
```

<!--more-->

#### [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)


  ```c++
  ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    if(headA == nullptr || headB == nullptr){
      return nullptr;
    }
    ListNode* temp_a = headA;
    ListNode* temp_b = headB;
    while(temp_a != temp_b){
      if(temp_a == nullptr){
        temp_a = headB;
      }else{
        temp_a = temp_a ->next;
      }
      if(temp_b == nullptr){
        temp_b = headA;
      }else{
        temp_b = temp_b->next;
      }
    }
    return temp_a;
  }
  ```
#### :o:[206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

![](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)
```c++
ListNode* reverseList(ListNode* head) {
  if(head == nullptr || head->next == nullptr){
    return head;
  }
  ListNode* temp = reverseList(head->next);
  head->next->next =head;
  head->next = nullptr;
  return temp;
}

```

#### [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

![](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)

  ```c++
  ListNode* combine(ListNode* l1, ListNode* l2){
    ListNode* head = new ListNode(-1);
    ListNode* pre = head;
    while(l1 && l2){
      ListNode* temp1 = l1;
      ListNode* temp2 = l2;
      if(temp1->val >= temp2->val){
        pre->next = temp2;
        temp2 = temp2->next;
      }else{
        pre-> next = temp1;
        temp1 = temp1->next;
      }
      pre = pre->next;
    }
    if(l1 == nullptr){
      pre->next = l2;
    }
    if(l2 == nullptr){
      pre->next = l1;
    }
    return head->next;

  }
  ```

#### [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

  ```c++
  bool hasCycle(ListNode *head) {
    ListNode* fast = head;
    ListNode* slow = head;
    if(head == nullptr || head->next == nullptr){
      return false;
    }
    while(fast != nullptr && fast->next != nullptr){
      slow = slow->next;
      fast = fast->next->next;
      if(fast == slow){
        return true;
      }
    }
    return false;
  }
  ```


#### [142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)


  ```c++
  ListNode *detectCycle(ListNode *head) {
      ListNode* slow = head;        
      ListNode* fast = head;        
      while(fast && fast->next){           
          slow = slow->next;            
          fast = fast->next->next;            
          if(slow == fast){                
              fast = head;                
              while(fast != slow){                    
                  fast = fast->next;                    
                  slow = slow->next;                
              }                
              return fast;            
          }        
      }        
      return nullptr;    
  }
  ```
#### [19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

![](https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg)


  ```c++
  ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode* p = head;
    ListNode* q = head;
    while(n > 0){
      p = p->next;
      n--;
    }
    if(!p){
      return head->next;
    }
    while(p->next){
      p = p->next;
      q = q->next;
    }
    q->next = q->next->next;
    return q;
  }
  ```

#### [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

![](https://assets.leetcode.com/uploads/2020/10/03/swap_ex1.jpg)



```c++
ListNode* swapPairs(ListNode* head) {
  ListNode* temp = new ListNode(-1);
  temp->next = head;
  while(temp ->next && temp->next->next){
    ListNode* l1 = temp->next;
    ListNode* l2 = temp->next->next;
    temp->next = l2;
    l1 ->next = l2->next;
    l2->next = l1;
    temp = l1;
  }
  return temp-> next;
}
```

#### [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/01/02/addtwonumber1.jpg)


  ```c++
  ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode* node = new ListNode();
    ListNode* cur = node;
    int sum = 0;
    int carry = 0;
    int single = 0;
    while(l1 && l2){
      sum = l1->val + l2->val+ carry;
      carry = sum/10;
      single = sum%10;
      ListNode* current = new ListNode(single);
      node->next = current;
      node = node ->next;
      l1 = l1->next;
      l2 = l2 ->next;
    }
    while(l1){
      sum = l1->val + carry;
      carry = sum/10;
      single = sum%10;
      ListNode* current = new ListNode(single);
      node->next = current;
      node = node ->next;
      l1 = l1->next;
    }
    while(l2){
      sum = l2->val + carry;
      carry = sum/10;
      single = sum%10;
      ListNode* current = new ListNode(single);
      node->next = current;
      node = node ->next;
      l2 = l2->next;
    }
    if(carry){
      ListNode* current = new ListNode(carry);
      node->next = current;
      node = node->next;
    }
    return cur->next;
  }
  ```



#### [445. 两数相加 II](https://leetcode-cn.com/problems/add-two-numbers-ii/)

![](https://pic.leetcode-cn.com/1626420025-fZfzMX-image.png)

  ```c++
  ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    stack<int> s1;
    stack<int> s2;
    int carry = 0;
    int sum = 0;
    int single = 0;
    ListNode* node = new ListNode();
    ListNode* head = node;

    while(l1){
      s1.push(l1->val);
      l1 = l1->next;
    }
    while(l2){
      s2.push(l1->val);
      l2 = l2->next;
    }
    while(!s1.empty() && !s2.empty()){
      sum = s1.top() + s2.top();
      carry = sum /10;
      single = sum %10;
      ListNode* temp = new ListNode(single);
      node->next = temp;
      node = node->next;
      s1.pop();
      s2.pop();
    }
    while(!s1.empty()){
      sum  = s1.top() + carry;
      single = sum % 10;
      carry = sum / 10;
      ListNode * tmp = new ListNode(single);
      node -> next = tmp;
      node = node->next;
      s1.pop();
    }
    while(!s2.empty()){
      sum  = s2.top() + carry;
      single = sum % 10;
      carry = sum / 10;
      ListNode * tmp = new ListNode(single);
      node -> next = tmp;
      node = node->next;
      s2.pop();
    }
    if(carry){
      ListNode * tmp = new ListNode(carry);
      node -> next = tmp;
      node = node->next;
    }
    ListNode* res = reverse(head->next);
    return res;
  }

  ListNode* reverse(ListNode* list){
    if(list == nullptr || list->next == nullptr){
      return list;
    }
    ListNode* temp = reverse(list->next);
    list->next->next = list;
    list->next = nullptr;
    return temp;
  }
  ```

#### [725. 分隔链表](https://leetcode-cn.com/problems/split-linked-list-in-parts/)

![](https://assets.leetcode.com/uploads/2021/06/13/split1-lc.jpg)

  ```c++
  vector<ListNode*> splitListToParts(ListNode* root, int k) {
    int length = 0;
    ListNode* temp = root;
    while(temp){
      length++;
      temp = temp->next;
    }
    int array_length = length / k > 0 ? (length / k) : 1;
    int arr_len[k];
    for(int i = 0 ; i < k ; i++){
      arr_len[i] = array_length;
    }
    int gap = 0;
    if(k * array_length <length){
      gap = length - k * array_length;
      for(int i = 0 ; i <gap ; i++){
        arr_len[i]++;
      }
    }

    vector<ListNode*> ans;
    ListNode* temp;
    for(int i = 0 ; i < k; i++){
      ans.push_back(root);
      for(int j = 0; j <arr_len[i]; j++){
        if(root){
          temp = root;
          root = root->next;
        }
      }
      if(temp){
        temp->next = nullptr;
      }
    }
    return ans;
  }

  ```


#### [328. 奇偶链表] (https://leetcode-cn.com/problems/odd-even-linked-list/)

![](https://assets.leetcode.com/uploads/2021/03/10/oddeven-linked-list.jpg)

  ```c++
  ListNode* oddEvenList(ListNode* head) {
    if(!head || !head->next || !head->next->next){
      return head;
    }
    ListNode* odd = head;
    ListNode* even = head->next;
    ListNode* even_first = even;
    while(odd->next && even ->next){
      odd->next = even->next;
      odd = odd->next;
      even->next = odd->next;
      even = even->next;
    }
    odd->next = even_first;
    return head;

  }

  ```

#### :o:[92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

![](https://assets.leetcode.com/uploads/2021/02/19/rev2ex2.jpg)


```c++
  ListNode* temp = nullptr;
  ListNode* reverseN(int n,ListNode* node){
    if(n == 1){
      temp = node->next;
      return node;
    }
    ListNode* last = reverseN(n-1, node->next);
    node->next->next = node;
    node ->next = temp
  }

```
#### [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

![](https://assets.leetcode.com/uploads/2021/01/04/list1.jpg)

```c++
ListNode* deleteDuplicates(ListNode* head) {
  if(!head || !head->next){
    return head;
  }
  int value = head->val;
  ListNode* temp = head->next;
  ListNode* carry = head;
  while(temp){
    if(temp->val == value){
      ListNode* deleteEle = temp;
      temp = temp->next;
      carry->next = temp;
      delete deleteEle;
      deleteEle = nullptr;
    }else{
      carry = carry->next;
      value = carry->val;
      temp = temp->next;
    }
  }
  return head;
}

```



#### [25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

![](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)





#### [剑指 Offer 06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)




  ```c++
    vector<int> reversePrint(ListNode* head) {
      vector<int> res;
      ListNode* newhead = reverse(head);
      while(newhead){
        res.push_back(newhead->val);
        newhead = newhead->next;
      }
      return res;
    }
    ListNode* reverse(ListNode* head){
      if(!head || !head->next) return head;
      ListNode* temp = reverse(head->next);
      head->next->next = head;
      head->next = nullptr;
      return temp;
    }

  ```


<!-- 
#### [剑指 Offer 35. 复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

```c++
Node* copyRandomList(Node* head) {
  if(!head){
    return head;
  }
  Node* temp = head;
  while(temp){
    Node* tmp = new Node(temp->val);
    tmp ->next = temp ->next;

  }
}

``` -->



#### [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

![](https://assets.leetcode.com/uploads/2021/03/14/invert1-tree.jpg)


  ```c++
  TreeNode* invertTree(TreeNode* root) {
    if(!root){
      return nullptr;
    }
    TreeNode* temp = root->left;
    root->left = roor->right;
    root->right = temp;

    root ->left = invertTree(root->left);
    root ->right = invertTree(root->right);
    return root;
  }
  ```
#### [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

![](https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg)



  ```c++
    bool hasPathSum(TreeNode* root, int sum) {
      if(!root)
      return false;
      sum -= root->val;
      if(!root->left && !root->right){
        return sum == 0;
      }
      return (hasPathSum(root->left, sum) || hasPathSum(root->right, sum));
    }


  ```

#### [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

![](https://assets.leetcode.com/uploads/2021/01/18/pathsumii1.jpg)

  ```c++
    vector<vector<int>> ans;
    vector<int> temp;
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
      recurse(root,targetSum);
      return ans;
    }
    void recurse(TreeNode* root, int targetSum){
      if(!root) return;
      temp.push_back(root->val);
      targetSum -= root->val;
      if(!root->left && !root->right && targetSum ==0){
        ans.push_back(temp);
      }
      recurse(root->left, targetSum);
      recurse(root->right, targetSum);
      temp.pop_back();
    }


  ```

  
#### [116. 填充每个节点的下一个右侧节点指针](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/)

![](https://assets.leetcode.com/uploads/2019/02/14/116_sample.png)


  ```c++
  Node* connect(Node* root) {
          if(!root){
              return NULL;
          }
          if(root->left){
              root->left->next = root->right;
              if(root->next && root->right){
                  root->right->next = root->next->left;
              }
          }
          connect(root->left);
          connect(root->right);
          return root;
      }
  ```


#### [117. 填充每个节点的下一个右侧节点指针 II](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node-ii/)

![](https://assets.leetcode.com/uploads/2019/02/15/117_sample.png)

```c++
Node* connect(Node* root) {

  if(root == nullptr || !(root->left && root->right)){
    return root;
  }

  if(root->left && left->right){
    root->left->next = get_next_node(root);
  }
  if(!root->right){
    root->left->next = get_next_node(root);
  }
  if(!root->left){
    root->right->next = get_next_node(root);
  }


  connect(root->left);
  connect(root->right);
  return root

}

Node* get_next_node(Node* root){
    while(root->next){
      if(root->next->left){
        return root->next->left;
      }
      else if(root->next->right){
        return root->next->right;
      }
    }
    return nullptr;

  }
```


#### [654. 最大二叉树](https://leetcode-cn.com/problems/maximum-binary-tree/)

![](https://assets.leetcode.com/uploads/2020/12/24/tree1.jpg)

  ```c++
  TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
    int left = 0;
    int right= nums.size();
    return build_tree(nums, left, right);
  }

  TreeNode* build_tree(vector<int>& nums, int left, int right){
    int index = -1;
    int max = INT_MIN;
    for(int i = left; i <right; i++){
      if(max > nums[i]){
        max = nums[i];
        index = i;
      }
    }
    TreeNode* root = new TreeNode(max);
    root->left = build_tree(nums, left, index);
    root->right = build_tree(nums, index+1, right);
    
    return root;

  }

  ```


#### :o:[105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)（★面试常考）

![](https://assets.leetcode.com/uploads/2021/02/19/tree.jpg)

```c++
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
  if(preorder.size() == 0 || inorder.size() == 0){
    return nullptr;
  }
  return TreeNode* buildTree(preorder, 0, preorder.size(), inorder, 0, inorder.size());

}

TreeNode* buildTree(vector<int>& preorder, int preorder_left, int preorder_right, vector<int>& inorder, int inorder_left, int inorder_right){
  int index = -1;
  int rootvalue = preorder[preorder_left];
  for(int i = inorder_left; i<inorder_right; i++){
    if(rootvalue == inorder[i]){
      index = i;
      break;
    }
  }
  int left_count = index - preorder_left;
  TreeNode* root = new TreeNode(rootvalue);
  root->left = buildTree(preorder, preorder_left, preorder_left + left_count, inorder, inorder_left, index);
  root->right = buildTree(preorder, preorder_left + left_count, preorder_right, inorder, index+1 ,inorder_right)
  return root;

}
```
#### [652. 寻找重复的子树](https://leetcode-cn.com/problems/find-duplicate-subtrees/)

![](https://assets.leetcode.com/uploads/2020/08/16/e1.jpg)


```c++
  map<string, int> count_tree;
  vector<TreeNode*> map_tree;
  vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
    get_tree(root);
    return map_tree;
  }
  string get_tree(TreeNode* root){
    if(!root){
      return "";
    }
    string left = get_tree(root->left);
    string right = get_tree(root->right);
    string str = left + "," + right + "," + to_string(root->val);
    count_tree[str]++;
    if(count_tree[str] == 2){
      map_tree.push_back(root);
    }
    return str;
  }
```


#### [968. 监控二叉树](https://leetcode-cn.com/problems/binary-tree-cameras/)


![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/29/bst_cameras_02.png)

```c++
public:
  int res;
  int minCameraCover(TreeNode* root) {
    res = 0;
    if(search(root) == 0){
      res++;
    }
    return res;
  }
  int search(TreeNode* root){
    if(root == nullptr){
      return  2;
    }
    int left = search(root->left);
    int right = search(root->right);
    
    if(left ==2 && right ==2){
      return 0;
    }
    if(left ==0 || right == 0){
      res++;
      return 1;
    }
    if(left ==1 || right ==1){
      return 2;
    }
    return -1;
  }


```


#### [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

![](https://assets.leetcode.com/uploads/2020/10/13/exx1.jpg)

```c++
public:
  int ans = INT_MIN;
  int case1;
  int maxPathSum(TreeNode* root) {
    return dfs(root);
  }
  int dfs(TreeNode* root){
    int left = dfs(root->left);
    int right = dfs(root->right);
    case1 = max(case1, left + right + root->val);
    case1 = max(case1, left);
    case1 = max(case1, right);
    int case2 = root->val;
    case2 = case2(case2, root->val+left);
    case2 = case2(case2, root->val+right);
    return case2;
  }
```


#### [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)