# My Leetcode 
- [Top 100 liked questions](https://leetcode.com/problemset/top-100-liked-questions/)
- [Top interview questions](https://leetcode.com/problemset/top-interview-questions/)

## Table of Contents
- Array
	- [Two Sum](#two-sum)
	- [Container With Most Water](#container-with-most-water)
	- [3Sum](#3sum)
	- [Remove Duplicates from Sorted Array](#remove-duplicates-from-sorted-array)
	- [Search in Rotated Sorted Array](#search-in-rotated-sorted-array)
	- [Find First and Last Position of Element in Sorted Array](#find-first-and-last-position-of-element-in-sorted-array)
	- [Rotate Image](#rotate-image)
	- [Maximum Subarray](#maximum-subarray)
	- [Jump Game](#jump-game)
	- [Merge Intervals](#merge-intervals)
	- [Unique Paths](#unique-paths)
	- [Set Matrix Zeroes](#set-matrix-zeroes)
	- [Subsets](#subsets)
	- [Word Search](#word-search)
	- [Pascal's Triangle](#pascals-triangle)
	- [Best Time to Buy and Sell Stock](#best-time-to-buy-and-sell-stock)
	- [Best Time to Buy and Sell Stock II](#best-time-to-buy-and-sell-stock-ii)
	- [Maximum Product Subarray](#maximum-product-subarray)
	- [Product of Array Except Self](#product-of-array-except-self)
	- [Find the Duplicate Number](#find-the-duplicate-number)
	- [Move Zeroes](#move-zeroes)
	- [Find Peak Element](#find-peak-element)
- Linked List
	- [Add Two Numbers](#add-two-numbers)
	- [Remove Nth Node From End of List](#remove-nth-node-from-end-of-list)
	- [Merge Two Sorted Lists](#merge-two-sorted-lists)
	- [Linked List Cycle](#linked-list-cycle)
	- [Intersection of Two Linked Lists](#intersection-of-two-linked-lists)
	- [Reverse Linked List](#reverse-linked-list)
- Depth-first Search
	- [Validate Binary Search Tree](#validate-binary-search-tree)
	- [Symmetric Tree](#symmetric-tree)
	- [Maximum Depth of Binary Tree](#maximum-depth-of-binary-tree)
	- [Number of Islands](#number-of-islands)
	- [Course Schedule](#course-schedule)
	- [Course Schedule II](#course-schedule-ii)
	- [Permutations](#permutations)
	- [Construct Binary Tree from Preorder and Inorder Traversal](#construct-binary-tree-from-preorder-and-inorder-traversal)
	- [Convert Sorted Array to Binary Search Tree](#convert-sorted-array-to-binary-search-tree)
	- [Populating Next Right Pointers in Each Node](#populating-next-right-pointers-in-each-node)
- Hash Table
	- [Longest Substring Without Repeating Characters](#longest-substring-without-repeating-characters)
	- [Valid Sudoku](#valid-sudoku)
	- [Group Anagrams](#group-anagrams)
	- [Binary Tree Inorder Traversal](#binary-tree-inorder-traversal)
	- [Single Number](#single-number)
	- [Top K Frequent Elements](#top-k-frequent-elements)
- Math
	- [Roman to Integer](#roman-to-integer)
	- [Pow(x, n)](#powx-n)
	- [Perfect Squares](#perfect-squares)
- Dynamic Programming
	- [Longest Palindromic Substring](#longest-palindromic-substring)
	- [Climbing Stairs](#climbing-stairs)
	- [Word Break](#word-break)
	- [House Robber](#house-robber)
	- [Longest Increasing Subsequence](#longest-increasing-subsequence)
	- [Coin Change](#coin-change)
- Others
	- [Valid Parentheses](#valid-parentheses)
	- [Letter Combinations of a Phone Number](#letter-combinations-of-a-phone-number)
	- [Generate Parentheses](#generate-parentheses)
	- [Subarray Sum Equals K](#subarray-sum-equals-k)
	- [Binary Tree Level Order Traversal](#binary-tree-level-order-traversal)
	- [Binary Tree Zigzag Level Order Traversal](#binary-tree-zigzag-level-order-traversal)

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

## Top K Frequent Elements
```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```
```
O(n)?
```
```python
from collections import defaultdict, Counter

class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        count = defaultdict(list)
        for key, c in Counter(nums).items():
            count[c].append(key)
        
        ans = []
        for target in range(len(nums), 0, -1):
            ans.extend(count.get(target, []))
            if len(ans) >= k:
                return ans[:k]
        return ans[:k]
```

## Reverse Linked List
```
Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
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
    # iteratively
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        ans = None
        while head:
            cur = head
            head = head.next
            cur.next = ans
            ans = cur
        return ans
    
    # recursively
    def reverseList(self, head, prev=None):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return prev
    
        curr, head.next = head.next, prev
        return self.reverseList(curr, head)
```

## Product of Array Except Self
```
Input:  [1,2,3,4]
Output: [24,12,8,6]
```
```
O(n)
```
```python
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        ans = []
        p = 1
        for i in range(len(nums)):
            ans.append(p)
            p *= nums[i]
            
        p = 1
        for i in range(len(nums)-1, -1, -1):
            ans[i] *= p
            p *= nums[i]
            
        return ans
```

## Remove Duplicates from Sorted Array
```
Given nums = [0,0,1,1,1,2,2,3,3,4],

Your function should return length = 5, with the first five elements of nums being modified to 0, 1, 2, 3, and 4 respectively.

It doesn't matter what values are set beyond the returned length.
```
```
O(n)
```
```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0

        newTail = 0

        for i in range(1, len(nums)):
            if nums[i] != nums[newTail]:
                newTail += 1
                nums[newTail] = nums[i]

        return newTail + 1
```

## Find First and Last Position of Element in Sorted Array
```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
```
```
O(n)
ps. there is another O(logn) solution
```
```python
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        left_idx = -1
        
        for i in range(len(nums)):
            if nums[i] == target:
                left_idx = i
                break
        
        if left_idx == -1:
            return [-1, -1]
        
        for j in range(len(nums)-1, -1, -1):
            if nums[j] == target:
                right_idx = j
                break
                
        return [left_idx, right_idx]
```

## Valid Sudoku
```
Input:
[
  ["8","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
Output: false
Explanation: Same as Example 1, except with the 5 in the top left corner being 
    modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.
```
```
O(n^2)
```
```python
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        def is_valid(data):
            nums = [d for d in data if d != '.']
            return len(nums) == len(set(nums))

        def is_valid_row(board):
            for data in board:
                if not is_valid(data):
                    return False
            return True

        def is_valid_column(board):
            for data in zip(*board):
                if not is_valid(data):
                    return False
            return True

        def is_valid_square(board):
            for i in [0, 3, 6]:
                for j in [0, 3, 6]:
                    data = [board[ii][jj] for ii in range(i, i+3) for jj in range(j, j+3)]
                    if not is_valid(data):
                        return False
            return True
        
        return is_valid_row(board) and is_valid_column(board) and is_valid_square(board)
```

## Search in Rotated Sorted Array
```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```
```
O(logn)
```
```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        l, r = 0, len(nums)-1
        
        while l <= r:
            mid = (l+r) // 2
            
            if nums[mid] == target:
                return mid
            
            if nums[l] <= nums[mid]:
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[r] >= target > nums[mid]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1
```

## Rotate Image
```
Given input matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

rotate the input matrix in-place such that it becomes:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
```
```
?
```
```python
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        matrix.reverse()
        for i in range(len(matrix)):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```

## Group Anagrams
```
Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
```
```
O(Nklogk)
```
```python
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        d = {}
        for s in strs:
            key = tuple(sorted(s))
            d[key] = d.get(key, []) + [s]
        return list(d.values())
```

## Pow(x, n)
```
Input: 2.00000, 10
Output: 1024.00000
```
```
O(n)
```
```python
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if not n:
            return 1
        
        if n < 0:
            return 1 / self.myPow(x, -n)
        
        if n % 2:
            return x * self.myPow(x, n-1)
        
        return self.myPow(x*x, n/2)
```

## Subarray Sum Equals K
```
Input:nums = [1,1,1], k = 2
Output: 2
```
```
O(n)
```
```python
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        d = {0: 1}
        cusum = 0
        counter = 0
        for n in nums:
            cusum += n
            counter += d.get(cusum-k, 0)
            d[cusum] = d.get(cusum, 0) + 1
        return counter
```

## Maximum Product Subarray
```
Input: [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
```
```
O(n)
```
```python
class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums_reverse = nums[::-1]
        
        for i in range(1, len(nums)):
            if nums[i-1] != 0:
                nums[i] *= nums[i-1]
            if nums_reverse[i-1] != 0:
                nums_reverse[i] *= nums_reverse[i-1]
        
        return max(nums + nums_reverse)
```

## Maximum Subarray
```
Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```
```
O(n)
```
```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        cusum = maxsum = nums[0]
        for n in nums[1:]:
            cusum = max(n, cusum+n)
            maxsum = max(cusum, maxsum)
        return maxsum
```

## Unique Paths
```
Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right
```
```
O(m*n)
```
```python
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        table = [[1 for _  in range(n)] for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                table[i][j] = table[i-1][j] + table[i][j-1]
        return table[-1][-1]
```

## Climbing Stairs
```
Input: 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
```
```
O(n)
```
```python
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 0
        
        if n == 1:
            return 1
        
        table = [0 for _ in range(n)]
        table[0] = 1
        table[1] = 2
        for i in range(2, n):
            table[i] = table[i-1] + table[i-2]
        return table[-1]
```

## Best Time to Buy and Sell Stock
```
Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
```
```
O(n)
```
```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        maxCur, maxAll = 0, 0
        for i in range(1, len(prices)):
            maxCur += prices[i] - prices[i-1]
            maxCur = max(0, maxCur)
            maxAll = max(maxCur, maxAll)
        return maxAll
```

## Word Break
```
Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
```
```
O(nk)
```
```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        n = len(s)
        table = [False for _ in range(n+1)]
        table[0] = True
        for i in range(1, n+1):
            for word in wordDict:
                if s[i-len(word):i] == word and table[i-len(word)]:
                    table[i] = True
        return table[-1]
```

## House Robber
```
Input: [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
             Total amount you can rob = 1 + 3 = 4.
```
```
O(n)
```
```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        
        if len(nums) == 1:
            return nums[0]
        
        table = [0 for _ in range(len(nums))]
        table[0] = nums[0]
        table[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            table[i] = max(table[i-1], table[i-2] + nums[i])
        return table[-1]
```

## Perfect Squares
```
Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.
```
```
O(n*(n^1/2))
```
```python
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [0 for _ in range(n+1)]
        for i in range(1, n+1):
            candidates = []
            j = 1
            while j*j <= i:
                candidates.append(dp[i-j*j]+1)
                j += 1
            if len(candidates):
                dp[i] = min(candidates)
        return dp[-1]
```

## Longest Increasing Subsequence
```
Input: [10,9,2,5,3,7,101,18]
Output: 4 
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4. 
```
```
O(n^2)
```
```python
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)
```

## Coin Change
```
Input: coins = [1, 2, 5], amount = 11
Output: 3 
Explanation: 11 = 5 + 5 + 1
```
```
O(S*n)
```
```python
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        dp = [0] + [float('inf') for _ in range(amount)]
        
        for i in range(1, amount+1):
            for coin in coins:
                if (i - coin) >= 0:
                    dp[i] = min(dp[i], dp[i-coin]+1)
        
        if dp[-1] == float('inf'):
            return -1
        
        return dp[-1]
```

## Merge Intervals
```
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
```
```
O(nlogn)
```
```python
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        if len(intervals) == 0:
            return []
    
        if len(intervals) == 1:
            return intervals
        
        intervals.sort(key=lambda x: x[0])
        
        merged = []
        for i, interval in enumerate(intervals):
            if i == 0:
                merged.append(interval)
            if interval[0] <= merged[-1][-1]:
                if interval[-1] > merged[-1][-1]:
                    merged[-1][-1] = interval[-1]
            else:
                merged.append(interval)
        return merged
```

## Jump Game
```
Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
```
```
O(n)
```
```python
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        i = 0
        maxReachIndex = nums[0]
        lastIndex = len(nums) - 1
        
        while i <= maxReachIndex and i <= lastIndex:
            maxReachIndex = max(maxReachIndex, nums[i]+i)
            i += 1
            
        return maxReachIndex >= lastIndex
```

## Subsets
```
Input: nums = [1,2,3]
Output:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
```
```
O(2^n)
```
```python
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = [[]]
        for num in nums:
            result += [i + [num] for i in result]
        return result
```

## Word Search
```
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
```
```
O(n*m*k)
```
```python
class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        if not board:
            return False
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.dfs(board, i, j, word):
                    return True
        return False
                
        
    def dfs(self, board, i, j, word):
        if len(word) == 0:
            return True
        
        if (i<0) or (j<0) or (i>=len(board)) or (j>=len(board[0])) or word[0] != board[i][j]:
            return False
        
        temp = board[i][j]
        board[i][j] = '#'
        ans = self.dfs(board, i-1, j, word[1:]) or \
            self.dfs(board, i, j-1, word[1:]) or \
            self.dfs(board, i+1, j, word[1:]) or \
            self.dfs(board, i, j+1, word[1:])
        board[i][j] = temp
        return ans
```

## Validate Binary Search Tree
```
    5
   / \
  1   4
     / \
    3   6

Input: [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.
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
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.check(root, float('-inf'), float('inf'))
    
    def check(self, node, left, right):
        if not node:
            return True
        
        if not left < node.val < right:
            return False
        
        return self.check(node.left, left, node.val) and self.check(node.right, node.val, right)
```

## Set Matrix Zeroes
```
Input: 
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
Output: 
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]
```
```
O(m*n)
```
```python
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        len_row, len_col = len(matrix), len(matrix[0])
        row, col = [], []
        for i in range(len_row):
            for j in range(len_col):
                if matrix[i][j] == 0:
                    row.append(i)
                    col.append(j)
                    
        for r in range(len_row):
            for c in range(len_col):
                if r in row or c in col:
                    matrix[r][c] = 0
```

## Symmetric Tree
```
    1
   / \
  2   2
 / \ / \
3  4 4  3
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
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.check(root, root)
        
    def check(self, node1, node2):
        if not node1 and not node2:
            return True
        
        if not node1 or not node2:
            return False
        
        if node1.val == node2.val:
            return self.check(node1.left, node2.right) and self.check(node1.right, node2.left)
        else:
            return False
```

## Binary Tree Level Order Traversal
```
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]
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
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        
        ans, level = [], [root]
        while level:
            ans.append([node.val for node in level])
            temp = []
            for node in level:
                temp.extend([node.left, node.right])
            level = [node for node in temp if node]
        return ans
```

## Binary Tree Zigzag Level Order Traversal
```
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]
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
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        
        order_flag = 1
        ans, level = [], [root]
        while level:
            if order_flag:
                ans.append([node.val for node in level])
                order_flag = 0
            else:
                ans.append([node.val for node in level[::-1]])
                order_flag = 1
                
            temp = []
            for node in level:
                temp.extend([node.left, node.right])
                
            level = [node for node in temp if node]
            
        return ans
```

## Course Schedule
```
Input: 2, [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0, and to take course 0 you should
             also have finished course 1. So it is impossible.
```
```
O(V+E)
```
```python
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        graph = collections.defaultdict(list)
        for u, v in prerequisites:
            graph[u].append(v)
        
        # 0: not yet, 1: visiting, 2: visited
        visited = [0 for _ in range(numCourses)]
        for i in range(numCourses):
            if not self.dfs(graph, visited, i):
                return False
        return True
    
    def dfs(self, graph, visited, i):
        # cycle is found
        if visited[i] == 1:
            return False
        # done 
        if visited[i] == 2:
            return True
        # visit all neighbors
        visited[i] = 1
        for j in graph[i]:
            if not self.dfs(graph, visited, j):
                return False
        visited[i] = 2
        return True
```

## Course Schedule II
```
Input: 4, [[1,0],[2,0],[3,1],[3,2]]
Output: [0,1,2,3] or [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both     
             courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0. 
             So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3] .
```
```
O(V+E)
```
```python
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        graph = collections.defaultdict(list)
        for u, v in prerequisites:
            graph[u].append(v)
            
        ans = []
        visited = [0 for _ in range(numCourses)]
        for i in range(numCourses):
            if not self.dfs(graph, visited, i, ans):
                return []
        return ans
            
    def dfs(self, graph, visited, x, ans):
        # cycle detected
        if visited[x] == 1:
            return False
        # finished and been added
        if visited[x] == 2:
            return True
        # go through all neighbors
        visited[x] = 1
        for j in graph[x]:
            if not self.dfs(graph, visited, j, ans):
                return False
        # finish
        visited[x] = 2
        ans.append(x)
        return True
```

## Pascal's Triangle
```
Input: 5
Output:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
```
```
O(n*2)
```
```python
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows == 0:
            return []
        
        if numRows == 1:
            return [[1]]
        
        ans = [[1]]
        for i in range(2, numRows+1):
            temp = [1]
            for j in range(i-2):
                item = ans[-1][j] + ans[-1][j+1]
                temp.append(item)
            temp.append(1)
            ans.append(temp)
            
        return ans
```

## Best Time to Buy and Sell Stock II
```
Input: [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
             Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
```
```
O(n)
```
```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        ans = 0
        for i in range(1, len(prices)):
            profit = prices[i] - prices[i-1]
            if profit > 0:
                ans += profit
        return ans
```

## Number of Islands
```
11000
11000
00100
00011

Output: 3
```
```
O(m*n)
```
```python
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid:
            return 0
        
        if not len(grid[0]):
            return 0
        
        self.grid = grid
        self.m, self.n = len(self.grid), len(self.grid[0])
        self.direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        
        count = 0
        for r in range(self.m):
            for c in range(self.n):
                if self.grid[r][c] == '1':
                    # bfs or dfs
                    self.bfs(r, c)
                    self.dfs(r, c)
                    count += 1
        return count
            
    def isValid(self, r, c):
        if r < 0 or c < 0 or r >= self.m or c >= self.n:
            return False
        return True
        
    def bfs(self, r, c):
        queue = [[r, c]]
        self.grid[r][c] = '0'
        while queue:
            r, c = queue.pop(0)
            for d in self.direction:
                nr, nc = r + d[0], c + d[1]
                if self.isValid(nr, nc) and self.grid[nr][nc] == '1':
                    queue.append([nr, nc])
                    self.grid[nr][nc] = '0'
                    
    def dfs(self, r, c):
        self.grid[r][c] = '0'
        for d in self.direction:
            nr, nc = r + d[0], c + d[1]
            if self.isValid(nr, nc) and self.grid[nr][nc] == '1':
                self.dfs(nr, nc)
```

## Linked List Cycle
```
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where tail connects to the second node.
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
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return True
        return False
```

## Intersection of Two Linked Lists
```
Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
Output: Reference of the node with value = 8
Input Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect). From the head of A, it reads as [4,1,8,4,5]. From the head of B, it reads as [5,0,1,8,4,5]. There are 2 nodes before the intersected node in A; There are 3 nodes before the intersected node in B.
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
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        A, B = headA, headB
        
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        
        return A
```

## Construct Binary Tree from Preorder and Inorder Traversal
```
For example, given

preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
Return the following binary tree:

    3
   / \
  9  20
    /  \
   15   7
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
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not preorder or not inorder:
            return None
        
        rootValue = preorder.pop(0)
        root = TreeNode(rootValue)
        rootIndex = inorder.index(rootValue)
        
        root.left = self.buildTree(preorder, inorder[:rootIndex])
        root.right = self.buildTree(preorder, inorder[rootIndex+1:])
        
        return root
```

## Convert Sorted Array to Binary Search Tree
```
Given the sorted array: [-10,-3,0,5,9],

One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:

      0
     / \
   -3   9
   /   /
 -10  5
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
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        self.nums = nums
        return self.helper(0, len(nums))
        
    def helper(self, start, end):
        if start == end:
            return None
        
        mid = (start + end) // 2
        root = TreeNode(self.nums[mid])
        root.left = self.helper(start, mid)
        root.right = self.helper(mid+1, end)
        return root
```

## Populating Next Right Pointers in Each Node
```
Input: root = [1,2,3,4,5,6,7]
Output: [1,#,2,3,#,4,5,6,7,#]
Explanation: Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
```
```
O(n)
```
```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
class Solution(object):
    # recursively
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if root and root.left and root.right:
            root.left.next = root.right
            if root.next:
                root.right.next = root.next.left
            self.connect(root.left)
            self.connect(root.right)
        return root
    
    # DFS
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return
        
        stack = [root]
        while stack:
            cur = stack.pop()
            if cur.left and cur.right:
                cur.left.next = cur.right
                if cur.next:
                    cur.right.next = cur.next.left
                stack.append(cur.left)
                stack.append(cur.right)
        return root
    
    # BFS
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return
        
        queue = [root]
        while queue:
            cur = queue.pop(0)
            if cur.left and cur.right:
                cur.left.next = cur.right
                if cur.next:
                    cur.right.next = cur.next.left
                queue.append(cur.left)
                queue.append(cur.right)
        return root
```

## Find the Duplicate Number
```
Input: [3,1,3,4,2]
Output: 3
```
```
O(n)
```
```python
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        seen = set()
        for num in nums:
            if num in seen:
                return num
            seen.add(num)
        return None
```

## Move Zeroes
```
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
```
```
O(n)
```
```python
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        lastNonZero = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[lastNonZero] = nums[i]
                lastNonZero += 1
                
        for i in range(lastNonZero, len(nums)):
            nums[i] = 0
```

## Find Peak Element
```
Input: nums = [1,2,1,3,5,6,4]
Output: 1 or 5 
Explanation: Your function can return either index number 1 where the peak element is 2, 
             or index number 5 where the peak element is 6.
```
```
O(n)
```
```python
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ans = 0
        for i, num in enumerate(nums):
            if nums[i] > nums[ans]:
                ans = i
        return ans
```
