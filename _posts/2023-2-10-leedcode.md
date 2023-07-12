---
layout: article
title: 力扣
key: 100027
tags: leedcode
category: blog
date: 2023-02-10 00:00:00 +08:00
mermaid: true
---

**347**
**209**
***3***
**5**
**968**



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

### 哈希表

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

#### [242. 有效的字母异位词](https://leetcode.cn/problems/valid-anagram/)

```c++
class Solution {
public:
    bool isAnagram(string s, string t) {
        if (s.length() != t.length()) {
            return false;
        }
        vector<int> table(26, 0);
        for (auto& ch: s) {
            table[ch - 'a']++;
        }
        for (auto& ch: t) {
            table[ch - 'a']--;
            if (table[ch - 'a'] < 0) {
                return false;
            }
        }
        return true;
    }
};

```

#### [1002. 查找常用字符](https://leetcode.cn/problems/find-common-characters/)

```c++
class Solution {
public:
    vector<string> commonChars(vector<string>& words) {
        vector<int> minfreq(26, INT_MAX);
        vector<int> freq(26);
        for (const string& word: words) {
            fill(freq.begin(), freq.end(), 0);
            for (char ch: word) {
                ++freq[ch - 'a'];
            }
            for (int i = 0; i < 26; ++i) {
                minfreq[i] = min(minfreq[i], freq[i]);
            }
        }

        vector<string> ans;
        for (int i = 0; i < 26; ++i) {
            for (int j = 0; j < minfreq[i]; ++j) {
                ans.emplace_back(1, i + 'a');
            }
        }
        return ans;
    }
};

```

#### [349. 两个数组的交集](https://leetcode.cn/problems/intersection-of-two-arrays/)

```c++
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        unordered_set<int> set1, set2;
        for (auto& num : nums1) {
            set1.insert(num);
        }
        for (auto& num : nums2) {
            set2.insert(num);
        }
        return getIntersection(set1, set2);
    }

    vector<int> getIntersection(unordered_set<int>& set1, unordered_set<int>& set2) {
        if (set1.size() > set2.size()) {
            return getIntersection(set2, set1);
        }
        vector<int> intersection;
        for (auto& num : set1) {
            if (set2.count(num)) {
                intersection.push_back(num);
            }
        }
        return intersection;
    }
};

```

#### [202. 快乐数](https://leetcode.cn/problems/happy-number/)

```c++
class Solution {
public:
    bool isHappy(int n) {
        unordered_set<int> set;
        int res = n;
        while (true) {
            res = square(res);
            if (res == 1) {
                return true;
            }
            if (set.find(res)!=set.end()) {//找到返回迭代器，失败返回end
                return false;
            }
            set.insert(res);
            
        }    
    }

    int square(int n) {
        int sum = 0, temp = 1;
        while (n != 0) {
            temp = n % 10;
            sum += temp * temp;
            n /= 10;
        }
        return sum;
    }
};


```

#### [454. 四数相加 II](https://leetcode.cn/problems/4sum-ii/)

```c++
class Solution {
public:
    int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
        unordered_map<int, int> countAB;
        for (int u: A) {
            for (int v: B) {
                ++countAB[u + v];
            }
        }
        int ans = 0;
        for (int u: C) {
            for (int v: D) {
                if (countAB.count(-u - v)) {
                    ans += countAB[-u - v];
                }
            }
        }
        return ans;
    }
};


```

#### [383. 赎金信](https://leetcode.cn/problems/ransom-note/submissions/)

```c++
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        if (ransomNote.size() > magazine.size()) {
            return false;
        }
        vector<int> cnt(26);
        for (auto & c : magazine) {
            cnt[c - 'a']++;
        }
        for (auto & c : ransomNote) {
            cnt[c - 'a']--;
            if (cnt[c - 'a'] < 0) {
                return false;
            }
        }
        return true;
    }
};

```


### 字符串

#### [344.反转字符串](https://leetcode.cn/problems/reverse-string/)

```c++
class Solution {
public:
    void reverseString(vector<char>& s) {
        int n = s.size();
        for (int left = 0, right = n - 1; left < right; ++left, --right) {
            swap(s[left], s[right]);
        }
    }
};

```

#### [541.反转字符串II](https://leetcode.cn/problems/reverse-string-ii/)

```c++
class Solution {
public:
    string reverseStr(string s, int k) {
        int n = s.length();
        for (int i = 0; i < n; i += 2 * k) {
            reverse(s.begin() + i, s.begin() + min(i + k, n));
        }
        return s;
    }
};

```


#### [剑指offer 05: 替换空格](https://leetcode.cn/problems/ti-huan-kong-ge-lcof/)


```c++
class Solution {
public:
    string replaceSpace(string s) {
        int count = 0, len = s.size();
        // 统计空格数量
        for (char c : s) {
            if (c == ' ') count++;
        }
        // 修改 s 长度
        s.resize(len + 2 * count);
        // 倒序遍历修改
        for(int i = len - 1, j = s.size() - 1; i < j; i--, j--) {
            if (s[i] != ' ')
                s[j] = s[i];
            else {
                s[j - 2] = '%';
                s[j - 1] = '2';
                s[j] = '0';
                j -= 2;
            }
        }
        return s;
    }
};

```

#### [剑指 Offer 58 - II. 左旋转字符串](https://leetcode.cn/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

```c++
//方法1---借助额外字符串
class Solution {
public:
    string reverseLeftWords(string s, int n) {
        string ans = s;
        int length = s.size();
        for(int i=0;i<length;i++)
        {
            ans[(i+length-n)%length] = s[i];
        }
        return ans;
    }
};

```

### 栈与队列


#### [151. 翻转字符串里的单词](https://github.com/youngyangyang04/leetcode-master)

```c++
class Solution {
public:
    string reverseWords(string s) {
        int left = 0, right = s.size() - 1;
        // 去掉字符串开头的空白字符
        while (left <= right && s[left] == ' ') ++left;

        // 去掉字符串末尾的空白字符
        while (left <= right && s[right] == ' ') --right;

        stack<string> d;
        string word;

        while (left <= right) {
            char c = s[left];
            if (word.size() && c == ' ') {
                // 将单词 push 到队列的头部
                d.push(move(word));
                word = "";
            }
            else if (c != ' ') {
                word += c;
            }
            ++left;
        }
        d.push(move(word));
        
        string ans;
        while (!d.empty()) {
            ans += d.top();
            d.pop();
            if (!d.empty()) ans += ' ';
        }
        return ans;
    }
};

```


#### [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/)


```c++
class Solution {
public:
    bool isValid(string s) {
        // 设置unordered_map m，对应(为1，[为2，{为3，)为4，]为5，}为6
        stack<char> st;
        bool istrue=true;
        for(char c:s){
            int flag=m[c];
            if(flag>=1&&flag<=3) st.push(c);
            else if(!st.empty()&&m[st.top()]==flag-3) st.pop();
            else {istrue=false;break;}
        }
        if(!st.empty()) istrue=false;
        return istrue;
    }
};

```


#### [150. 逆波兰表达式求解](https://leetcode.cn/problems/evaluate-reverse-polish-notation/)

```c++
class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<int> stk;
        int n = tokens.size();
        for (int i = 0; i < n; i++) {
            string& token = tokens[i];
            if (isNumber(token)) {
                stk.push(atoi(token.c_str()));
            } else {
                int num2 = stk.top();
                stk.pop();
                int num1 = stk.top();
                stk.pop();
                switch (token[0]) {
                    case '+':
                        stk.push(num1 + num2);
                        break;
                    case '-':
                        stk.push(num1 - num2);
                        break;
                    case '*':
                        stk.push(num1 * num2);
                        break;
                    case '/':
                        stk.push(num1 / num2);
                        break;
                }
            }
        }
        return stk.top();
    }

    bool isNumber(string& token) {
        return !(token == "+" || token == "-" || token == "*" || token == "/");
    }
};

```

#### [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        priority_queue<pair<int, int>> q;
        for (int i = 0; i < k; ++i) {
            q.emplace(nums[i], i);
        }
        vector<int> ans = {q.top().first};
        for (int i = k; i < n; ++i) {
            q.emplace(nums[i], i);
            while (q.top().second <= i - k) {
                q.pop();
            }
            ans.push_back(q.top().first);
        }
        return ans;
    }
};
```


#### [347. 前K个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)

```c++
class Solution {
public:


    static bool cmp(pair<int, int>&m, pair<int, int> & n){
        return m.second > n.second;
    }

    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int,int> map;
        for(int i = 0 ; i <nums.size(); i++){
            map[nums[i]]++;
        }
        priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(&cmp)> q(cmp);

        for(auto& [num, count] : map){
            if(q.size() == k){
                if(q.top().second < count){
                    q.pop();
                    q.emplace(num, count);
                }
            }else{
                q.emplace(num, count);
            }
        }
        vector<int> res;
        while(!q.empty()){
            res.emplace_back(q.top().first);
            q.pop();
        }
        return res;
    }
};
```

#### [1047. 删除字符串中的所有相邻重复项](https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/)


```c++
class Solution {
public:
    string removeDuplicates(string s) {
        string stk;
        for (char ch : s) {
            if (!stk.empty() && stk.back() == ch) {
                stk.pop_back();
            } else {
                stk.push_back(ch);
            }
        }
        return stk;
    }
};

```



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

#### [27. 移除元素](https://leetcode.cn/problems/remove-element/)

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
#### [977. 有序数组的平方](https://leetcode.cn/problems/squares-of-a-sorted-array/)


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

#### [209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/)


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

#### [59. 螺旋矩阵II](https://leetcode.cn/problems/spiral-matrix-ii/)

![](https://assets.leetcode.com/uploads/2020/11/13/spiraln.jpg)

```c++
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        int maxNum = n * n;
        int curNum = 1;
        vector<vector<int>> matrix(n, vector<int>(n));
        int row = 0, column = 0;
        vector<vector<int>> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};  // 右下左上
        int directionIndex = 0;
        while (curNum <= maxNum) {
            matrix[row][column] = curNum;
            curNum++;
            int nextRow = row + directions[directionIndex][0], nextColumn = column + directions[directionIndex][1];
            if (nextRow < 0 || nextRow >= n || nextColumn < 0 || nextColumn >= n || matrix[nextRow][nextColumn] != 0) {
                directionIndex = (directionIndex + 1) % 4;  // 顺时针旋转至下一个方向
            }
            row = row + directions[directionIndex][0];
            column = column + directions[directionIndex][1];
        }
        return matrix;
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

#### [707. 设计链表](https://leetcode.cn/problems/design-linked-list/)

```c++
class MyLinedList(){
private:
    int size;
    ListNode* head;
public:
    MyLinedList(){
        this->size = 0;
        this->head = new ListNode(0);
    }
    int get(int index){
        if(index<0 || index >= size){
            return 0;
        }
        ListNode* cur = head;
        for(int cur_index = 0; cur_index <= index;cur_index++){
            cur = cur->next;
        }
        return cur->val;
    }
    int add_at_head(int val){
        add_at_index(0,val);
    }
    int add_at_tail(int val){
        add_at_index(size,val);
    }
    void add_at_index(int index, int val){
        if(index < 0 || index > size){
            return;
        }
        size++;
        ListNode* pre = head;
        ListNode* newone = new ListNode(val);

        for(int i = 0 ; i<index; i++){
            pre = pre->next;
        }
        newone->next = pre->next;
        pre->next = newone;
    }
    void delete_at_index(int index){
        if(index < 0 || index >= size){
            return;
        }
        ListNode* pre = head;
        for(int i = 0 ; i <index; i++){
            pre = pre->next;
        }
        ListNode* deleteone = pre->next;
        pre->next = pre->next->next;
        delete deleteone;
}
}
```

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
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (head == nullptr || head->next == nullptr) {
            return head;
        }
        ListNode* newHead = head->next;
        head->next = swapPairs(newHead->next);
        newHead->next = head;
        return newHead;
    }
};

```


#### [445. 两数相加 II](https://leetcode-cn.com/problems/add-two-numbers-ii/)

![](https://pic.leetcode-cn.com/1626420025-fZfzMX-image.png)

  ```c++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        stack<int> s1, s2;
        while (l1) {
            s1.push(l1 -> val);
            l1 = l1 -> next;
        }
        while (l2) {
            s2.push(l2 -> val);
            l2 = l2 -> next;
        }
        int carry = 0;
        ListNode* ans = nullptr;
        while (!s1.empty() || !s2.empty() || carry != 0) {
            int a = s1.empty() ? 0 : s1.top();
            int b = s2.empty() ? 0 : s2.top();
            if (!s1.empty()) s1.pop();
            if (!s2.empty()) s2.pop();
            int cur = a + b + carry;
            carry = cur / 10;
            cur %= 10;
            auto curnode = new ListNode(cur);
            curnode -> next = ans;
            ans = curnode;
        }
        return ans;
    }
};
  ```

#### [725. 分隔链表](https://leetcode-cn.com/problems/split-linked-list-in-parts/)

![](https://assets.leetcode.com/uploads/2021/06/13/split1-lc.jpg)

  ```c++
  class Solution {
public:
    vector<ListNode*> splitListToParts(ListNode* head, int k) {
        int n = 0;
        ListNode *temp = head;
        while (temp != nullptr) {
            n++;
            temp = temp->next;
        }
        int quotient = n / k, remainder = n % k;

        vector<ListNode*> parts(k,nullptr);
        ListNode *curr = head;
        for (int i = 0; i < k && curr != nullptr; i++) {
            parts[i] = curr;
            int partSize = quotient + (i < remainder ? 1 : 0);
            for (int j = 1; j < partSize; j++) {
                curr = curr->next;
            }
            ListNode *next = curr->next;
            curr->next = nullptr;
            curr = next;
        }
        return parts;
    }
};

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
class Solution {
private:
    
public:

    void reverse(ListNode* Node){
        if(Node == nullptr || Node->next == nullptr){
            return;
        }
        ListNode* next = Node->next;
        reverse(Node->next);
        next->next = Node;
    }
    ListNode *reverseBetween(ListNode *head, int left, int right) {
        ListNode* node = new ListNode(-1);
        node->next = head;
        ListNode* pre = node;
        for(int i = 0 ; i < left -1; i++){
            pre = pre->next;
        }
        ListNode* right1 = pre;

        for(int i = 0 ; i < right - left + 1; i++){
            right1 = right1 ->next;
        }

        ListNode* nextpart = right1->next;
        ListNode* changepart = pre->next;
        pre->next = nullptr;
        right1->next = nullptr;
        reverse(changepart);
        pre->next = right1;
        changepart->next = nextpart;
        
        return node->next;
    }
};



```
#### [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

![](https://assets.leetcode.com/uploads/2021/01/04/list1.jpg)

```c++
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if (!head) {
            return head;
        }

        ListNode* cur = head;
        while (cur->next) {
            if (cur->val == cur->next->val) {
                cur->next = cur->next->next;
            }
            else {
                cur = cur->next;
            }
        }

        return head;
    }
};
```







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


### 二叉树

#### [101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/)

![](https://assets.leetcode.com/uploads/2021/02/19/symtree1.jpg)

```c++
class Solution {
public:
    bool check(TreeNode *p, TreeNode *q) {
        if (!p && !q) return true;
        if (!p || !q) return false;
        return p->val == q->val && check(p->left, q->right) && check(p->right, q->left);
    }

    bool isSymmetric(TreeNode* root) {
        return check(root, root);
    }
};

```

#### [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) return 0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
};
```


#### [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

![](https://assets.leetcode.com/uploads/2021/03/14/invert1-tree.jpg)


  ```c++
  TreeNode* invertTree(TreeNode* root) {
    if(!root){
      return nullptr;
    }
    TreeNode* temp = root->left;
    root->left = root->right;
    root->right = temp;

    root ->left = invertTree(root->left);
    root ->right = invertTree(root->right);
    return root;
  }
  ```

#### [111. 二叉树最小深度](https://leetcode.cn/problems/minimum-depth-of-binary-tree/)


```c++
class Solution {
public:
    int minDepth(TreeNode *root) {
        if (root == nullptr) {
            return 0;
        }

        if (root->left == nullptr && root->right == nullptr) {
            return 1;
        }

        int min_depth = INT_MAX;
        if (root->left != nullptr) {
            min_depth = min(minDepth(root->left), min_depth);
        }
        if (root->right != nullptr) {
            min_depth = min(minDepth(root->right), min_depth);
        }

        return min_depth + 1;
    }
};

```

#### [222. 完全二叉树的节点个数](https://leetcode.cn/problems/count-complete-tree-nodes/)

![](https://assets.leetcode.com/uploads/2021/01/14/complete.jpg)

```c++
class Solution {
public:
    int countNodes(TreeNode* root) {
        if(!root) return 0;
        return countNodes(root->left) + countNodes(root->right) + 1;
    }
};
```


#### [257. 二叉树的所有路径](https://leetcode.cn/problems/binary-tree-paths/)

![](https://assets.leetcode.com/uploads/2021/03/12/paths-tree.jpg)

```c++
class Solution {
public:
    void construct_paths(TreeNode* root, string path, vector<string>& paths) {
        if (root != nullptr) {
            path += to_string(root->val);
            if (root->left == nullptr && root->right == nullptr) {  
                
                // 当前节点是叶子节点
                paths.push_back(path);
            } else {
                path += "->";

                // 当前节点不是叶子节点，继续递归遍历
                construct_paths(root->left, path, paths);
                construct_paths(root->right, path, paths);
            }
        }
    }

    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> paths;
        construct_paths(root, "", paths);
        return paths;
    }
};

```

#### [404. 左叶子之和](https://leetcode.cn/problems/sum-of-left-leaves/)

```c++
class Solution {
    
    public int sumOfLeftLeaves(TreeNode root) {
        if(root==null) return 0;
        return sumOfLeftLeaves(root.left) 
            + sumOfLeftLeaves(root.right) 
            + (root.left!=null && root.left.left==null && root.left.right==null ? root.left.val : 0);
    }
};

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
class Solution {
public:
    vector<vector<int>> ret;
    vector<int> path;

    void dfs(TreeNode* root, int targetSum) {
        if (root == nullptr) {
            return;
        }
        path.emplace_back(root->val);
        targetSum -= root->val;
        if (root->left == nullptr && root->right == nullptr && targetSum == 0) {
            ret.emplace_back(path);
        }
        dfs(root->left, targetSum);
        dfs(root->right, targetSum);
        path.pop_back();
    }

    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        dfs(root, targetSum);
        return ret;
    }
};




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
class Solution {
public:
    Node* connect(Node* root) {
        if (root == NULL) return root;
        queue<Node*> q;
        q.push(root);

        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                auto node = q.front();
                q.pop();
                if (i < size - 1) node->next = q.front();
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
        }
        return root;
    }
};

```


#### [654. 最大二叉树](https://leetcode-cn.com/problems/maximum-binary-tree/)

![](https://assets.leetcode.com/uploads/2020/12/24/tree1.jpg)

  ```c++
class Solution {
public:
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        return construct(nums, 0, nums.size() - 1);
    }

    TreeNode* construct(const vector<int>& nums, int left, int right) {
        if (left > right) {
            return nullptr;
        }
        int best = left;
        for (int i = left + 1; i <= right; ++i) {
            if (nums[i] > nums[best]) {
                best = i;
            }
        }
        TreeNode* node = new TreeNode(nums[best]);
        node->left = construct(nums, left, best - 1);
        node->right = construct(nums, best + 1, right);
        return node;
    }
};


  ```


#### [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)（★面试常考）

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
class Solution {
private:
    unordered_map<string,int> ump;
    vector<TreeNode*> ans;
public:
    string dfs(TreeNode* node){
        if(node==nullptr) return "";
        string ss=to_string(node->val)+","+dfs(node->left)+","+dfs(node->right);
        ump[ss]++;
        if(ump[ss]==2) ans.push_back(node);
        return ss;
    }
    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
        dfs(root);
        return ans;
    }
};

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
class Solution {
private:
    int maxSum = INT_MIN;

public:
    int maxGain(TreeNode* node) {
        if (node == nullptr) {
            return 0;
        }
        
        // 递归计算左右子节点的最大贡献值
        // 只有在最大贡献值大于 0 时，才会选取对应子节点
        int leftGain = max(maxGain(node->left), 0);
        int rightGain = max(maxGain(node->right), 0);

        // 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
        int priceNewpath = node->val + leftGain + rightGain;

        // 更新答案
        maxSum = max(maxSum, priceNewpath);

        // 返回节点的最大贡献值
        return node->val + max(leftGain, rightGain);
    }

    int maxPathSum(TreeNode* root) {
        maxGain(root);
        return maxSum;
    }
};

```


#### [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)
