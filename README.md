# My Leetcode 
- Top 100 liked questions ([link](https://leetcode.com/problemset/top-100-liked-questions/))
- Top interview questions ([link](https://leetcode.com/problemset/top-interview-questions/))

## Table of Contents
- Easy
	- [Two Sum](#two-sum)
	- [Valid Parentheses](#valid-parentheses)
	- [Merge Two Sorted Lists](#merge-two-sorted-lists)
	- [Roman to Integer](#roman-to-integer)
	- [Maximum Depth of Binary Tree](#maximum-depth-of-binary-tree)
	- [Single Number](#single-number)
	- [Fizz Buzz](#fizz-buzz)
- Medium
	- [Add Two Numbers](#add-two-numbers)
	- [Longest Substring Without Repeating Characters](#longest-substring-without-repeating-characters)
	- [Longest Palindromic Substring](#longest-palindromic-substring)
	- [Container With Most Water](#container-with-most-water)
	- [3Sum](#3sum)
	- [Letter Combinations of a Phone Number](#letter-combinations-of-a-phone-number)
	- [Remove Nth Node From End of List](#remove-nth-node-from-end-of-list)
	- [Binary Tree Inorder Traversal](#binary-tree-inorder-traversal)
	- [Permutations](#permutations)
	- [Generate Parentheses](#generate-parentheses)

## Two Sum
```
Given nums = [2, 7, 11, 15], target = 9,
Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
```
```
O(n) 
```
```python
class Solution(object):
    def twoSum(self, nums, target):
	"""
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        seen = {}
        for i in range(len(nums)):
            remaining = target - nums[i]
            if remaining in seen:
                return [seen[remaining], i]
            seen[nums[i]] = i
```

## Add Two Numbers
```
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```
```
m, n = len(input1), len(input2)
O(max(m, n))
```
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        result = ListNode(0)
        result_tail = result
        carry = 0
        
        while l1 or l2 or carry:
            v1 = (l1.val if l1 else 0)
            v2 = (l2.val if l2 else 0)
            carry, out = divmod(v1+v2+carry, 10)
            
            result_tail.next = ListNode(out)
            result_tail = result_tail.next
            
            l1 = (l1.next if l1 else None)
            l2 = (l2.next if l2 else None)
        
        return result.next
```

## Longest Substring Without Repeating Characters
```
Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3.

Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3. 
             Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
```
```
O(n)
```
```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        start = maxLength = 0
        usedChar = {}

        for index, char in enumerate(s):
            if char in usedChar and start <= usedChar[char]:
                start = usedChar[char] + 1
            else:
                maxLength = max(maxLength, index - start + 1)
            usedChar[char] = index
        return maxLength
```

## Longest Palindromic Substring
```
Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.

Input: "cbbd"
Output: "bb"
```
```
O(n^2)
```
```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        ans = ""
        for i in range(len(s)):
            odd = self.palindromeStr(s, i, i)
            even = self.palindromeStr(s, i, i+1)
            ans = max(ans, odd, even, key=len)
        return ans
            
    def palindromeStr(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l+1:r]
```

## Container With Most Water
```
Input: [1,8,6,2,5,4,8,3,7]
Output: 49
```
```
O(n)
```
```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        start = 0
        end = len(height) - 1
        ans = 0
        
        while start != end:
            if height[start] > height[end]:
                area = (end - start) * height[end]
                end -= 1
            else:
                area = (end - start) * height[start]
                start += 1
            ans = max(ans, area)
        return ans
```

## 3Sum
```
Given array nums = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```
```
O(nlogn+n^2) ~= O(n^2)
```
```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans = []
        nums.sort()
        length = len(nums)
        for i in range(length-2):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l, r = i + 1, length -1
            while l < r:
                total = nums[i] + nums[l] + nums[r]
                if total < 0:
                    l += 1
                elif total > 0:
                    r -= 1
                else:
                    ans.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    l += 1
                    r -= 1
        return ans
```

## Letter Combinations of a Phone Number
```
Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
```
```
O(n)
```
```python
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        mappings = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
                    '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        if digits:
            ans = ['']
        else:
            ans = []
            
        for d in digits:
            ans = [p+q for p in ans for q in mappings[d]]
        return ans
```

## Remove Nth Node From End of List
```
Given linked list: 1->2->3->4->5, and n = 2.
After removing the second node from the end, the linked list becomes 1->2->3->5.
```
```
O(n)
```
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        first = second = head
        for _ in range(n):
            first = first.next
        if not first:
            return head.next
        
        while first.next:
            first = first.next
            second = second.next
        second.next = second.next.next
        return head
```

## Valid Parentheses
```
Input: "{[]}"
Output: true

Input: "([)]"
Output: false
```
```
O(n)
```
```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        pairs = {'}': '{', ')': '(', ']': '['}
        
        for char in s:
            if char in pairs:
                if stack:
                    top_element = stack.pop()
                    if pairs[char] != top_element:
                        return False
                else:
                    return False
            else:
                stack.append(char)
        return not stack
```

## Merge Two Sorted Lists
```
Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4
```
```
O(m+n)
```
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = cur = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 or l2
        return dummy.next
```

## Roman to Integer
```
Input: "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
```
```
O(n)
```
```python
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        roman = {'M': 1000, 'D': 500, 'C': 100, 
                 'L': 50, 'X': 10, 'V': 5,'I': 1}
        ans = 0
        for i in range(len(s)-1):
            if roman.get(s[i]) >= roman.get(s[i+1]):
                ans += roman.get(s[i])
            else:
                ans -= roman.get(s[i])
        return ans + roman.get(s[-1])
```

## Maximum Depth of Binary Tree
```
    3
   / \
  9  20
    /  \
   15   7
Depth: 3
```
```
O(n)
```
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

## Single Number
```
Input: [4,1,2,1,2]
Output: 4
```
```
O(n)
```
```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return 2*sum(set(nums))-sum(nums)
```

## Fizz Buzz
```
n = 15,

Return:
[
    "1",
    "2",
    "Fizz",
    "4",
    "Buzz",
    "Fizz",
    "7",
    "8",
    "Fizz",
    "Buzz",
    "11",
    "Fizz",
    "13",
    "14",
    "FizzBuzz"
]
```
```
O(n)
```
```python
class Solution(object):
    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        ans = []
        for i in range(1, n+1):
            if i % 3 == 0 and i % 5 == 0:
                ans.append('FizzBuzz')
            elif i % 3 == 0:
                ans.append('Fizz')
            elif i % 5 == 0:
                ans.append('Buzz')
            else:
                ans.append(str(i))
        return ans
```

## Binary Tree Inorder Traversal
```
Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,3,2]
```
```
O(n)
```
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```

## Permutations
```
Input: [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```
```
O(n*n!)
```
```python
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans = []
        self.DFS(nums, [], ans)
        return ans
        
    def DFS(self, nums, path, ans):
        if not nums:
            ans.append(path)
        for i in range(len(nums)):
            self.DFS(nums[:i]+nums[i+1:], path+[nums[i]], ans)
```

## Generate Parentheses
```
For example, given n = 3, a solution set is:

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```
```
O((2k)!/((k!*(k+1)!)).......
```
```python
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        if not n:
            return []
        ans = []
        self.DFS(n, n, "", ans)
        return ans
        
    def DFS(self, left_stack, right_stack, path, ans):
        if left_stack > right_stack:
            return
        if not left_stack and not right_stack:
            ans.append(path)
        if left_stack:
            self.DFS(left_stack-1, right_stack, path+"(", ans)
        if right_stack:
            self.DFS(left_stack, right_stack-1, path+")", ans)
```
