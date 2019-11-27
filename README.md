# My Leetcode 
- [Top 100 liked questions](https://leetcode.com/problemset/top-100-liked-questions/)
- [Top interview questions](https://leetcode.com/problemset/top-interview-questions/)

## Table of Contents
- Easy
	- [Two Sum](#two-sum)
	- [Valid Parentheses](#valid-parentheses)
	- [Merge Two Sorted Lists](#merge-two-sorted-lists)
	- [Roman to Integer](#roman-to-integer)
	- [Maximum Depth of Binary Tree](#maximum-depth-of-binary-tree)
	- [Single Number](#single-number)
	- [Fizz Buzz](#fizz-buzz)
	- [Reverse Linked List](#reverse-linked-list)
	- [Remove Duplicates from Sorted Array](#remove-duplicates-from-sorted-array)
	- [Maximum Subarray](#maximum-subarray)
	- [Climbing Stairs](#climbing-stairs)
	- [Best Time to Buy and Sell Stock](#best-time-to-buy-and-sell-stock)
	- [House Robber](#house-robber)
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
	- [Top K Frequent Elements](#top-k-frequent-elements)
	- [Product of Array Except Self](#product-of-array-except-self)
	- [Find First and Last Position of Element in Sorted Array](#find-first-and-last-position-of-element-in-sorted-array)
	- [Valid Sudoku](#valid-sudoku)
	- [Search in Rotated Sorted Array](#search-in-rotated-sorted-array)
	- [Rotate Image](#rotate-image)
	- [Group Anagrams](#group-anagrams)
	- [Pow(x, n)](#powx-n)
	- [Subarray Sum Equals K](#subarray-sum-equals-k)
	- [Maximum Product Subarray](#maximum-product-subarray)
	- [Unique Paths](#unique_paths)
	- [Word Break](#word-break)
	- [Perfect Squares](#perfect-squares)
	- [Longest Increasing Subsequence](#longest-increasing-subsequence)\
	- [Coin Change](#coin-change)

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
