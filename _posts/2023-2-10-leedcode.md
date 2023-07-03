---
layout: article
title: 力扣
key: 100027
tags: leedcode
category: blog
date: 2023-02-10 00:00:00 +08:00
mermaid: true
---



#### [1. 两数之和](https://leetcode.cn/problems/two-sum/)

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map <int,int> map;
        for(int i = 0 ; i < nums.size(); i++){
            auto num = map.find(target - nums[i]);
            if(num != map.end()){
                return {i, num->second};
            }
            map[nums[i]] = i;
        }
        return {};
    }
};
```
#### [6. N 字形变换](https://leetcode.cn/problems/zigzag-conversion/)

```c++
class Solution {
public:
	string convert(string s, int numRows) {

		if (numRows == 1) return s;

		vector<string> rows(min(numRows, int(s.size()))); // 防止s的长度小于行数
		int curRow = 0;
		bool goingDown = false;

		for (char c : s) {
			rows[curRow] += c;
			if (curRow == 0 || curRow == numRows - 1) {// 当前行curRow为0或numRows -1时，箭头发生反向转折
				goingDown = !goingDown;
			}
			curRow += goingDown ? 1 : -1;
		}

		string ret;
		for (string row : rows) {// 从上到下遍历行
			ret += row;
		}

		return ret;
	}
};
```


#### [7. 整数反转](https://leetcode.cn/problems/reverse-integer/)
```c++
class Solution {
public:
    int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            if (rev < INT_MIN / 10 || rev > INT_MAX / 10) {
                return 0;
            }
            int digit = x % 10;
            x /= 10;
            rev = rev * 10 + digit;
        }
        return rev;
    }
};
```


***92***

***25***



***35***



### 数组

#### [704. 二分查找](https://leetcode.cn/problems/binary-search/)

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while(left <= right){
            int mid = (right - left) / 2 + left;
            int num = nums[mid];
            if (num == target) {
                return mid;
            } else if (num > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return -1;
    }
};
```

#### [27. 移除元素] (https://leetcode.cn/problems/remove-element/)

```c++
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int n = nums.size();
        int left = 0;
        for (int right = 0; right < n; right++) {
            if (nums[right] != val) {
                nums[left] = nums[right];
                left++;
            }
        }
        return left;
    }
};
```
#### [977. 有序数组的平方]()


```c++
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        vector<int> ans;
        for (int num: nums) {
            ans.push_back(num * num);
        }
        sort(ans.begin(), ans.end());
        return ans;
    }
};

```

#### [209. 长度最小的子数组]()




```c++
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }
        int ans = INT_MAX;
        int start = 0, end = 0;
        int sum = 0;
        while (end < n) {
            sum += nums[end];
            while (sum >= s) {
                ans = min(ans, end - start + 1);
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return ans == INT_MAX ? 0 : ans;
    }
};

```

### 滑动窗口

#### [3. 无重复最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)


```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        // 哈希集合，记录每个字符是否出现过
        unordered_set<char> occ;
        int n = s.size();
        // 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
        int rk = -1, ans = 0;
        // 枚举左指针的位置，初始值隐性地表示为 -1
        for (int i = 0; i < n; ++i) {
            if (i != 0) {
                // 左指针向右移动一格，移除一个字符
                occ.erase(s[i - 1]);
            }
            while (rk + 1 < n && !occ.count(s[rk + 1])) {
                // 不断地移动右指针
                occ.insert(s[rk + 1]);
                ++rk;
            }
            // 第 i 到 rk 个字符是一个极长的无重复字符子串
            ans = max(ans, rk - i + 1);
        }
        return ans;
    }
};

```

#### [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

![](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg)

```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int i = 0, j = height.size() - 1, res = 0;
        while(i < j) {
            res = height[i] < height[j] ? 
                max(res, (j - i) * height[i++]): 
                max(res, (j - i) * height[j--]); 
        }
        return res;
    }
};
```

### 动态规划 

#### [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size();
        if (n < 2) {
            return s;
        }

        int maxLen = 1;
        int begin = 0;
        // dp[i][j] 表示 s[i..j] 是否是回文串
        vector<vector<int>> dp(n, vector<int>(n));
        // 初始化：所有长度为 1 的子串都是回文串
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
        }
        // 递推开始
        // 先枚举子串长度
        for (int L = 2; L <= n; L++) {
            // 枚举左边界，左边界的上限设置可以宽松一些
            for (int i = 0; i < n; i++) {
                // 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
                int j = L + i - 1;
                // 如果右边界越界，就可以退出当前循环
                if (j >= n) {
                    break;
                }

                if (s[i] != s[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }

                // 只要 dp[i][L] == true 成立，就表示子串 s[i..L] 是回文，此时记录回文长度和起始位置
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substr(begin, maxLen);
    }
};
```

### 链表

#### [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/)

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/01/02/addtwonumber1.jpg)

```c++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *head = nullptr, *tail = nullptr;
        int carry = 0;
        while (l1 || l2) {
            int n1 = l1 ? l1->val: 0;
            int n2 = l2 ? l2->val: 0;
            int sum = n1 + n2 + carry;
            if (!head) {
                head = tail = new ListNode(sum % 10);
            } else {
                tail->next = new ListNode(sum % 10);
                tail = tail->next;
            }
            carry = sum / 10;
            if (l1) {
                l1 = l1->next;
            }
            if (l2) {
                l2 = l2->next;
            }
        }
        if (carry > 0) {
            tail->next = new ListNode(carry);
        }
        return head;
    }
};
```

#### [203. 移除链表元素](https://leetcode.cn/problems/remove-linked-list-elements/)

![](https://assets.leetcode.com/uploads/2021/03/06/removelinked-list.jpg)

```c++
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        if (head == nullptr) {
            return head;
        }
        head->next = removeElements(head->next, val);
        return head->val == val ? head->next : head;
    }
};
```



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
