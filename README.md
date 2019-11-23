# Leetcode 
* top 100 liked questions ([link](https://leetcode.com/problemset/top-100-liked-questions/))
* top interview questions ([link](https://leetcode.com/problemset/top-interview-questions/))

## 1. Two Sum (Easy)
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

## 2. Add Two Numbers (Medium)
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

## 3. Longest Substring Without Repeating Characters (Medium)
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

## 4. Median of Two Sorted Arrays (Hard)

## 5. Longest Palindromic Substring (Medium)
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

## 10. Regular Expression Matching (Hard)

## 11. Container With Most Water (Medium)
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

## 15. 3Sum (Medium)
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

## 17. Letter Combinations of a Phone Number (Medium)
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

## 19. Remove Nth Node From End of List (Medium)
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

## 20. Valid Parentheses (Easy)
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

## 21. Merge Two Sorted Lists (Easy)
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
