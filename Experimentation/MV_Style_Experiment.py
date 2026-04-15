"""
Synthetic Voting Test Runner

This script tests whether models vote based on code appearance or correctness.

Setup:
- 10 tasks, each with 3 solutions
- Solution A: INCORRECT but pretty (terse, no comments)
- Solution B: INCORRECT but pretty (well-formatted, comments)
- Solution C: CORRECT but ugly (different approach, well-formatted)

Hypothesis: If models vote by appearance, they'll choose A or B.
            If models vote by correctness, they'll choose C.

Expected baseline: 0% accuracy if appearance-based, 100% if correctness-based.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from collections import Counter
import gc
from multiprocessing import Process, Queue

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODELS = {
    "DeepSeek Coder 7B Instruct": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "CodeGemma 7B Instruct": "google/codegemma-7b-it",
    "Code Llama 7B Instruct": "codellama/CodeLlama-7b-Instruct-hf"
}

# ============================================================================
# HELPER FUNCTIONS (from multimodel_voting_benchmark)
# ============================================================================

def _execute_test_in_process(code_to_test: str, test_code: str, entry_point: str, result_queue: Queue):
    """Execute test in a separate process"""
    result = {
        'passed': False,
        'error': None,
        'code_tested': code_to_test
    }
    
    try:
        namespace = {}
        exec("from typing import List, Dict, Tuple, Optional, Union, Any", namespace)
        
        try:
            exec(code_to_test, namespace)
        except SyntaxError as e:
            result['error'] = f"Syntax error: {str(e)}"
            result_queue.put(result)
            return
        except Exception as e:
            result['error'] = f"Execution error: {str(e)}"
            result_queue.put(result)
            return
        
        if entry_point not in namespace:
            result['error'] = f"Function '{entry_point}' not found"
            result_queue.put(result)
            return
        
        try:
            exec(test_code, namespace)
        except AssertionError as e:
            error_msg = str(e) if str(e) else "Assertion failed (no message)"
            result['error'] = f"Test assertion failed: {error_msg}"
            result_queue.put(result)
            return
        except Exception as e:
            result['error'] = f"Test error: {str(e)}"
            result_queue.put(result)
            return
        
        result['passed'] = True
        result_queue.put(result)
        
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
        result_queue.put(result)

def run_unit_test(code: str, test_code: str, entry_point: str, timeout: int = 40) -> Dict:
    """Run unit tests with timeout"""
    result = {
        'passed': False,
        'error': None,
        'code_tested': None
    }
    
    try:
        code_to_test = code.strip()
        
        # Extract from <solution> tags if present
        solution_match = re.search(r'<solution>(.*?)</solution>', code_to_test, re.DOTALL)
        if solution_match:
            code_to_test = solution_match.group(1).strip()
        else:
            # Try to extract if only closing tag present
            if '</solution>' in code_to_test:
                code_to_test = code_to_test.split('</solution>')[0].strip()
            else:
                result['error'] = "No <solution> tags found in model output"
                return result
        
        # Remove <end_of_turn> if model added it
        if '<end_of_turn>' in code_to_test:
            code_to_test = code_to_test.split('<end_of_turn>')[0].strip()
        
        # Remove >>> prefix if present
        if code_to_test.startswith('>>>'):
            code_to_test = code_to_test[3:].strip()
        
        # Clean markdown fences
        if code_to_test.startswith('```python'):
            code_to_test = code_to_test[9:].strip()
        elif code_to_test.startswith('```'):
            code_to_test = code_to_test[3:].strip()
        
        if code_to_test.endswith('```'):
            code_to_test = code_to_test[:-3].strip()
        
        code_to_test = re.sub(r'```python\s*\n?', '\n', code_to_test)
        code_to_test = re.sub(r'```\s*\n?', '\n', code_to_test)
        
        lines = code_to_test.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped in ['```', '```python', '```py']:
                continue
            cleaned_lines.append(line)
        code_to_test = '\n'.join(cleaned_lines)
        
        result['code_tested'] = code_to_test
        
        if not code_to_test or len(code_to_test.strip()) == 0:
            result['error'] = "Generated code is empty after extraction"
            return result
        
        # Run in subprocess with timeout
        result_queue = Queue()
        process = Process(target=_execute_test_in_process, args=(code_to_test, test_code, entry_point, result_queue))
        process.start()
        process.join(timeout=timeout)
        
        if process.is_alive():
            process.terminate()
            process.join()
            result['error'] = f"Infinite loop detected: exceeded {timeout}s timeout"
            return result
        
        if not result_queue.empty():
            result = result_queue.get()
        else:
            result['error'] = "Test process terminated unexpectedly"
        
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
    
    return result

def load_model_safe(model_name: str, model_path: str):
    """Load model with proper configuration"""
    import os
    os.environ["ACCELERATE_USE_FSDP"] = "false"
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None, None, None

def load_all_models():
    """Load all models"""
    models_loaded = {}
    
    for model_name, model_path in MODELS.items():
        st.write(f"Loading {model_name}...")
        tokenizer, model, device = load_model_safe(model_name, model_path)
        
        if tokenizer and model:
            models_loaded[model_name] = (tokenizer, model, device)
            st.success(f"✓ Loaded {model_name}")
        else:
            st.error(f"✗ Failed to load {model_name}")
    
    return models_loaded

def generate_vote(tokenizer, model, device, voting_prompt, max_tokens, temperature):
    """Generate a vote from a model"""
    try:
        with torch.no_grad():
            inputs = tokenizer(voting_prompt, return_tensors="pt").to(device)
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            
            prompt_length = inputs["input_ids"].shape[1]
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=1.0 if temperature == 0 else temperature,
                do_sample=False if temperature == 0 else True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            )
            
            generated_tokens = outputs[0][prompt_length:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            del inputs
            del outputs
            torch.cuda.empty_cache()
            
            return result.strip()
    except Exception as e:
        return f"Voting error: {e}"

def create_voting_prompt(task: str, solutions: Dict[str, str]) -> str:
    """Create voting prompt with solutions wrapped in XML tags"""
    candidate_letters = ['A', 'B', 'C']
    
    prompt = f"""<instruction>
<task>
{task}
</task>
Among the {len(solutions)} solutions below, which solution best implements the task?
Respond with ONLY the solution tag name (solutionA, solutionB, or solutionC).

Encapsulate your answer in <best> tags:
>>> <best>solutionX</best>
</instruction>

<allSolutions>
"""
    
    for idx, (model_name, solution) in enumerate(solutions.items()):
        letter = candidate_letters[idx]
        prompt += f"<solution{letter}>\n{solution}\n</solution{letter}>\n\n"
    
    prompt += "</allSolutions>\n\nYour response:"
    
    return prompt

def extract_vote(vote_text: str, valid_solutions: List[str]) -> str:
    """Extract vote from model output"""
    # Try <best>X</best>
    match = re.search(r'<best>(.*?)</best>', vote_text, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip().lower()
        if extracted in [v.lower() for v in valid_solutions]:
            return extracted
    
    # Try solutionX anywhere
    match = re.search(r'solution([ABC])', vote_text, re.IGNORECASE)
    if match:
        normalized = f"solution{match.group(1).upper()}"
        if normalized in valid_solutions:
            return normalized.lower()
    
    return None

# ============================================================================
# SYNTHETIC TASKS DATA
# ============================================================================

SYNTHETIC_TASKS = [
    {
        "task_id": "subtle_1",
        "level": "easy",
        "prompt": """Write a function that finds the index of a target value in a sorted array. Return -1 if not found.

def binary_search(arr: list, target: int) -> int:
    # Your implementation here
    pass
    
assert binary_search([1, 2, 3, 4, 5], 3) == 2
assert binary_search([1, 2, 3, 4, 5], 6) == -1
assert binary_search([], 1) == -1
assert binary_search([5], 5) == 0""",
        
        "entry_point": "binary_search",
        
        "test": """
assert binary_search([1, 2, 3, 4, 5], 3) == 2
assert binary_search([1, 2, 3, 4, 5], 6) == -1
assert binary_search([], 1) == -1
assert binary_search([5], 5) == 0
assert binary_search([1, 3, 5, 7, 9], 1) == 0
assert binary_search([1, 3, 5, 7, 9], 9) == 4
""",
        
        "solutions": {
            "A_incorrect_pretty": """def binary_search(arr: list, target: int) -> int:
    \"\"\"Binary search implementation.\"\"\"
    left = 0
    right = len(arr) - 1
    
    # BUG: Uses < instead of <=, misses single element case
    while left < right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1""",
            
            "B_incorrect_pretty": """def binary_search(arr: list, target: int) -> int:
    \"\"\"Binary search with edge case handling.\"\"\"
    if not arr:
        return -1
    
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            # BUG: Doesn't update right correctly
            right = mid
    
    return -1""",
            
            "C_correct_ugly": """def binary_search(arr: list, target: int) -> int:
    \"\"\"Search for target in sorted array.\"\"\"
    if not arr:
        return -1
    
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1"""
        }
    },
    
    {
        "task_id": "subtle_2",
        "level": "easy",
        "prompt": """Write a function that removes duplicates from a list while preserving order.

def remove_duplicates(nums: list) -> list:
    # Your implementation here
    pass
    
assert remove_duplicates([1, 2, 2, 3]) == [1, 2, 3]
assert remove_duplicates([]) == []
assert remove_duplicates([1, 1, 1]) == [1]
assert remove_duplicates([1, 2, 3]) == [1, 2, 3]""",
        
        "entry_point": "remove_duplicates",
        
        "test": """
assert remove_duplicates([1, 2, 2, 3]) == [1, 2, 3]
assert remove_duplicates([]) == []
assert remove_duplicates([1, 1, 1]) == [1]
assert remove_duplicates([1, 2, 3]) == [1, 2, 3]
assert remove_duplicates([3, 2, 1, 2, 3]) == [3, 2, 1]
assert remove_duplicates([0, 0, 1, 1, 2, 2]) == [0, 1, 2]
""",
        
        "solutions": {
            "A_incorrect_pretty": """def remove_duplicates(nums: list) -> list:
    \"\"\"Remove duplicates preserving order.\"\"\"
    seen = set()
    result = []
    
    for num in nums:
        # BUG: Checks if num NOT in seen, but logic is inverted
        if num in seen:
            continue
        result.append(num)
        seen.add(num)
    
    return result""",
            
            "B_incorrect_pretty": """def remove_duplicates(nums: list) -> list:
    \"\"\"Remove duplicates while maintaining order.\"\"\"
    if not nums:
        return []
    
    seen = {}
    result = []
    
    for num in nums:
        # BUG: Should check 'not in', but checks 'in'
        if num not in seen:
            result.append(num)
        seen[num] = True
    
    return result""",
            
            "C_correct_ugly": """def remove_duplicates(nums: list) -> list:
    \"\"\"Remove duplicates keeping order.\"\"\"
    if not nums:
        return []
    
    seen = set()
    result = []
    
    for num in nums:
        if num not in seen:
            result.append(num)
            seen.add(num)
    
    return result"""
        }
    },
    
    {
        "task_id": "subtle_3",
        "level": "easy",
        "prompt": """Write a function that checks if a string is a valid palindrome, ignoring non-alphanumeric characters and case.

def is_palindrome(s: str) -> bool:
    # Your implementation here
    pass
    
assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("race a car") == False
assert is_palindrome("") == True
assert is_palindrome("a") == True""",
        
        "entry_point": "is_palindrome",
        
        "test": """
assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("race a car") == False
assert is_palindrome("") == True
assert is_palindrome("a") == True
assert is_palindrome("ab") == False
assert is_palindrome("aba") == True
""",
        
        "solutions": {
            "A_incorrect_pretty": """def is_palindrome(s: str) -> bool:
    \"\"\"Check if string is palindrome ignoring non-alphanumeric.\"\"\"
    # Filter to alphanumeric and lowercase
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    
    # Check palindrome with two pointers
    left = 0
    right = len(cleaned) - 1
    
    # BUG: Uses < instead of <=
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    
    return True""",
            
            "B_incorrect_pretty": """def is_palindrome(s: str) -> bool:
    \"\"\"Validate palindrome with case/punctuation handling.\"\"\"
    # Extract alphanumeric characters
    cleaned = ''.join(c for c in s if c.isalnum())
    
    # BUG: Doesn't convert to lowercase
    # Compare with reverse
    return cleaned == cleaned[::-1]""",
            
            "C_correct_ugly": """def is_palindrome(s: str) -> bool:
    \"\"\"Check palindrome ignoring case and non-alphanumeric.\"\"\"
    # Clean string
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    
    # Two pointer comparison
    left = 0
    right = len(cleaned) - 1
    
    while left <= right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    
    return True"""
        }
    },
    
    {
        "task_id": "subtle_4",
        "level": "middle",
        "prompt": """Write a function that merges two sorted arrays into one sorted array.

def merge_sorted(arr1: list, arr2: list) -> list:
    # Your implementation here
    pass
    
assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
assert merge_sorted([], [1, 2]) == [1, 2]
assert merge_sorted([1, 2], []) == [1, 2]
assert merge_sorted([], []) == []""",
        
        "entry_point": "merge_sorted",
        
        "test": """
assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
assert merge_sorted([], [1, 2]) == [1, 2]
assert merge_sorted([1, 2], []) == [1, 2]
assert merge_sorted([], []) == []
assert merge_sorted([1], [2]) == [1, 2]
assert merge_sorted([2], [1]) == [1, 2]
""",
        
        "solutions": {
            "A_incorrect_pretty": """def merge_sorted(arr1: list, arr2: list) -> list:
    \"\"\"Merge two sorted arrays into one sorted array.\"\"\"
    result = []
    i = 0
    j = 0
    
    # Merge while both have elements
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    
    # BUG: Forgets to add remaining elements from arr1
    while j < len(arr2):
        result.append(arr2[j])
        j += 1
    
    return result""",
            
            "B_incorrect_pretty": """def merge_sorted(arr1: list, arr2: list) -> list:
    \"\"\"Merge two sorted lists maintaining order.\"\"\"
    if not arr1:
        return arr2
    if not arr2:
        return arr1
    
    result = []
    i = 0
    j = 0
    
    while i < len(arr1) and j < len(arr2):
        # BUG: Uses < instead of <=, changes order for duplicates
        if arr1[i] < arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    
    # Add remaining
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    
    return result""",
            
            "C_correct_ugly": """def merge_sorted(arr1: list, arr2: list) -> list:
    \"\"\"Merge two sorted arrays.\"\"\"
    if not arr1:
        return arr2
    if not arr2:
        return arr1
    
    result = []
    i = 0
    j = 0
    
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    
    # Add remaining elements
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    
    return result"""
        }
    },
    
    {
        "task_id": "subtle_5",
        "level": "middle",
        "prompt": """Write a function that finds the first non-repeating character in a string. Return None if all characters repeat.

def first_unique_char(s: str) -> str:
    # Your implementation here
    pass
    
assert first_unique_char("leetcode") == "l"
assert first_unique_char("loveleetcode") == "v"
assert first_unique_char("aabb") == None
assert first_unique_char("") == None""",
        
        "entry_point": "first_unique_char",
        
        "test": """
assert first_unique_char("leetcode") == "l"
assert first_unique_char("loveleetcode") == "v"
assert first_unique_char("aabb") == None
assert first_unique_char("") == None
assert first_unique_char("z") == "z"
assert first_unique_char("aabbcc") == None
""",
        
        "solutions": {
            "A_incorrect_pretty": """def first_unique_char(s: str) -> str:
    \"\"\"Find first non-repeating character.\"\"\"
    if not s:
        return None
    
    # Count character frequencies
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    # Find first unique
    for char in char_count:  # BUG: Iterates dict, loses order
        if char_count[char] == 1:
            return char
    
    return None""",
            
            "B_incorrect_pretty": """def first_unique_char(s: str) -> str:
    \"\"\"Return first character that appears once.\"\"\"
    if not s:
        return None
    
    char_count = {}
    
    # Count occurrences
    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 0  # BUG: Initializes to 0 instead of 1
    
    # Find first unique
    for char in s:
        if char_count[char] == 1:
            return char
    
    return None""",
            
            "C_correct_ugly": """def first_unique_char(s: str) -> str:
    \"\"\"Find first non-repeating character.\"\"\"
    if not s:
        return None
    
    # Build frequency map
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    # Find first with count 1
    for char in s:
        if char_count[char] == 1:
            return char
    
    return None"""
        }
    },
    
    {
        "task_id": "subtle_6",
        "level": "middle",
        "prompt": """Write a function that rotates an array to the right by k steps.

def rotate_array(nums: list, k: int) -> list:
    # Your implementation here
    pass
    
assert rotate_array([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]
assert rotate_array([1, 2], 3) == [2, 1]
assert rotate_array([1], 0) == [1]
assert rotate_array([], 3) == []""",
        
        "entry_point": "rotate_array",
        
        "test": """
assert rotate_array([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]
assert rotate_array([1, 2], 3) == [2, 1]
assert rotate_array([1], 0) == [1]
assert rotate_array([], 3) == []
assert rotate_array([1, 2, 3], 4) == [3, 1, 2]
""",
        
        "solutions": {
            "A_incorrect_pretty": """def rotate_array(nums: list, k: int) -> list:
    \"\"\"Rotate array to right by k positions.\"\"\"
    if not nums or k == 0:
        return nums
    
    n = len(nums)
    # BUG: Doesn't handle k > n
    k = k % n
    
    # Rotate by slicing
    return nums[-k:] + nums[:-k]""",
            
            "B_incorrect_pretty": """def rotate_array(nums: list, k: int) -> list:
    \"\"\"Rotate array right by k steps.\"\"\"
    if not nums:
        return nums
    
    n = len(nums)
    k = k % n if n > 0 else 0
    
    # BUG: When k=0 after modulo, should return original
    # but slicing nums[-0:] returns empty
    return nums[-k:] + nums[:-k]""",
            
            "C_correct_ugly": """def rotate_array(nums: list, k: int) -> list:
    \"\"\"Rotate array to right by k.\"\"\"
    if not nums or k == 0:
        return nums
    
    n = len(nums)
    k = k % n
    
    if k == 0:
        return nums
    
    return nums[-k:] + nums[:-k]"""
        }
    },
    
    {
        "task_id": "subtle_7",
        "level": "middle",
        "prompt": """Write a function that checks if two strings are anagrams of each other.

def are_anagrams(s1: str, s2: str) -> bool:
    # Your implementation here
    pass
    
assert are_anagrams("listen", "silent") == True
assert are_anagrams("hello", "world") == False
assert are_anagrams("", "") == True
assert are_anagrams("a", "a") == True""",
        
        "entry_point": "are_anagrams",
        
        "test": """
assert are_anagrams("listen", "silent") == True
assert are_anagrams("hello", "world") == False
assert are_anagrams("", "") == True
assert are_anagrams("a", "a") == True
assert are_anagrams("rat", "car") == False
assert are_anagrams("anagram", "nagaram") == True
""",
        
        "solutions": {
            "A_incorrect_pretty": """def are_anagrams(s1: str, s2: str) -> bool:
    \"\"\"Check if two strings are anagrams.\"\"\"
    # BUG: Doesn't check length first
    if not s1 and not s2:
        return True
    
    # Count characters
    char_count = {}
    
    for char in s1:
        char_count[char] = char_count.get(char, 0) + 1
    
    for char in s2:
        char_count[char] = char_count.get(char, 0) - 1
    
    # Check all counts are zero
    return all(count == 0 for count in char_count.values())""",
            
            "B_incorrect_pretty": """def are_anagrams(s1: str, s2: str) -> bool:
    \"\"\"Determine if strings are anagrams.\"\"\"
    if len(s1) != len(s2):
        return False
    
    # BUG: Uses set instead of sorted, loses duplicate info
    return set(s1) == set(s2)""",
            
            "C_correct_ugly": """def are_anagrams(s1: str, s2: str) -> bool:
    \"\"\"Check if two strings are anagrams.\"\"\"
    if len(s1) != len(s2):
        return False
    
    # Sort and compare
    return sorted(s1) == sorted(s2)"""
        }
    },
    
    {
        "task_id": "subtle_8",
        "level": "hard",
        "prompt": """Write a function that finds the longest substring without repeating characters.

def longest_unique_substring(s: str) -> int:
    # Your implementation here
    pass
    
assert longest_unique_substring("abcabcbb") == 3
assert longest_unique_substring("bbbbb") == 1
assert longest_unique_substring("") == 0
assert longest_unique_substring("pwwkew") == 3""",
        
        "entry_point": "longest_unique_substring",
        
        "test": """
assert longest_unique_substring("abcabcbb") == 3
assert longest_unique_substring("bbbbb") == 1
assert longest_unique_substring("") == 0
assert longest_unique_substring("pwwkew") == 3
assert longest_unique_substring("dvdf") == 3
assert longest_unique_substring("abcdefg") == 7
""",
        
        "solutions": {
            "A_incorrect_pretty": """def longest_unique_substring(s: str) -> int:
    \"\"\"Find length of longest substring without repeating chars.\"\"\"
    if not s:
        return 0
    
    max_length = 0
    left = 0
    char_set = set()
    
    for right in range(len(s)):
        # BUG: Doesn't handle the case when char is already in set
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        # BUG: Calculates length wrong (should be right - left + 1)
        max_length = max(max_length, right - left)
    
    return max_length""",
            
            "B_incorrect_pretty": """def longest_unique_substring(s: str) -> int:
    \"\"\"Calculate longest substring with unique characters.\"\"\"
    if not s:
        return 0
    
    max_len = 0
    left = 0
    seen = {}
    
    for right in range(len(s)):
        if s[right] in seen:
            # BUG: Should use max(left, seen[s[right]] + 1)
            left = seen[s[right]] + 1
        
        seen[s[right]] = right
        max_len = max(max_len, right - left + 1)
    
    return max_len""",
            
            "C_correct_ugly": """def longest_unique_substring(s: str) -> int:
    \"\"\"Find longest unique substring length.\"\"\"
    if not s:
        return 0
    
    max_length = 0
    left = 0
    char_index = {}
    
    for right in range(len(s)):
        if s[right] in char_index:
            # Move left pointer
            left = max(left, char_index[s[right]] + 1)
        
        char_index[s[right]] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length"""
        }
    },
    
    {
        "task_id": "subtle_9",
        "level": "hard",
        "prompt": """Write a function that finds the k-th largest element in an unsorted array.

def find_kth_largest(nums: list, k: int) -> int:
    # Your implementation here
    pass
    
assert find_kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
assert find_kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
assert find_kth_largest([1], 1) == 1""",
        
        "entry_point": "find_kth_largest",
        
        "test": """
assert find_kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
assert find_kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
assert find_kth_largest([1], 1) == 1
assert find_kth_largest([7, 6, 5, 4, 3, 2, 1], 1) == 7
assert find_kth_largest([7, 6, 5, 4, 3, 2, 1], 7) == 1
""",
        
        "solutions": {
            "A_incorrect_pretty": """def find_kth_largest(nums: list, k: int) -> int:
    \"\"\"Find k-th largest element in array.\"\"\"
    # Sort in descending order
    sorted_nums = sorted(nums, reverse=True)
    
    # BUG: Returns k-th unique element, not k-th element
    unique_nums = list(set(sorted_nums))
    unique_nums.sort(reverse=True)
    
    return unique_nums[k - 1]""",
            
            "B_incorrect_pretty": """def find_kth_largest(nums: list, k: int) -> int:
    \"\"\"Return k-th largest element.\"\"\"
    # Sort ascending
    nums.sort()
    
    # BUG: Returns from wrong end (should be -k)
    return nums[-k + 1]""",
            
            "C_correct_ugly": """def find_kth_largest(nums: list, k: int) -> int:
    \"\"\"Find k-th largest element.\"\"\"
    # Sort descending
    sorted_nums = sorted(nums, reverse=True)
    
    # Return k-th element
    return sorted_nums[k - 1]"""
        }
    },
    
    {
        "task_id": "subtle_10",
        "level": "hard",
        "prompt": """Write a function that validates if a string of brackets is balanced.

def is_valid_brackets(s: str) -> bool:
    # Your implementation here
    pass
    
assert is_valid_brackets("()") == True
assert is_valid_brackets("()[]{}") == True
assert is_valid_brackets("(]") == False
assert is_valid_brackets("([)]") == False
assert is_valid_brackets("{[]}") == True""",
        
        "entry_point": "is_valid_brackets",
        
        "test": """
assert is_valid_brackets("()") == True
            # Debug output
            if not test_res['passed']:
                with st.expander(f"Debug {sol_name}"):
                    st.code(test_res.get('code_tested', sol_code)[:500], language='python')
                    st.error(f"Error: {test_res.get('error', 'Unknown')}")
assert is_valid_brackets("()[]{}") == True
assert is_valid_brackets("(]") == False
assert is_valid_brackets("([)]") == False
assert is_valid_brackets("{[]}") == True
assert is_valid_brackets("") == True
assert is_valid_brackets("(") == False
""",
        
        "solutions": {
            "A_incorrect_pretty": """def is_valid_brackets(s: str) -> bool:
    \"\"\"Check if brackets are balanced.\"\"\"
    if not s:
        return True
    
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in s:
        if char in pairs:
            stack.append(char)
        else:
            # BUG: Doesn't check if stack is empty first
            if pairs[stack.pop()] != char:
                return False
    
    return len(stack) == 0""",
            
            "B_incorrect_pretty": """def is_valid_brackets(s: str) -> bool:
    \"\"\"Validate bracket balancing.\"\"\"
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    
    for char in s:
        if char in pairs:
            # BUG: Compares wrong way
            if stack and stack[-1] == pairs[char]:
                stack.pop()
            else:
                return False
        else:
            stack.append(char)
    
    # BUG: Should check stack is empty
    return True""",
            
            "C_correct_ugly": """def is_valid_brackets(s: str) -> bool:
    \"\"\"Check bracket balance.\"\"\"
    if not s:
        return True
    
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    
    for char in s:
        if char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
        else:
            stack.append(char)
    
    return len(stack) == 0"""
        }
    },
]

# ============================================================================
# TEST RUNNER
# ============================================================================

def run_synthetic_test(temperature_vote: float = 0.0):
    """Run synthetic voting test"""
    
    # Load models
    st.header("Loading Models...")
    models_loaded = load_all_models()
    
    if not models_loaded:
        st.error("Failed to load models")
        return []
    
    st.success(f"Loaded {len(models_loaded)} models")
    
    results = []
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, task in enumerate(SYNTHETIC_TASKS):
        status_text.text(f"Testing task {idx + 1}/{len(SYNTHETIC_TASKS)}: {task['task_id']}")
        
        task_result = {
            'task_id': task['task_id'],
            'level': task['level'],
            'solutions': {},
            'votes': {},
            'ground_truth': 'A_incorrect_pretty',  # We know A is correct
            'winner': None,
            'correct_vote': False,
        }
        
        # Store solutions (they're pre-made)
        task_result['solutions'] = task['solutions']
        
        # Test all solutions to verify our ground truth
        st.write(f"  Testing solutions...")
        test_results = {}
        for sol_name, sol_code in task['solutions'].items():
            test_res = run_unit_test(sol_code, task['test'], task['entry_point'], timeout=10)
            test_results[sol_name] = test_res['passed']
            st.write(f"    {sol_name}: {'✅ PASS' if test_res['passed'] else '❌ FAIL'}")
        
        task_result['test_results'] = test_results
        
        # Verify only C is correct
        if test_results['C_correct_ugly'] != True:
            st.error(f"⚠️ WARNING: Solution C failed for {task['task_id']}!")
        if test_results.get('A_incorrect_pretty', test_results.get('A_correct_ugly')) != False:
            st.error(f"⚠️ WARNING: Solution A passed for {task['task_id']}!")
        if test_results['B_incorrect_pretty'] != False:
            st.error(f"⚠️ WARNING: Solution B passed for {task['task_id']}!")
        
        # Create voting prompt
        # Map solutions to solutionA, solutionB, solutionC format
        solutions_for_voting = {
            'Model_A': task['solutions']['A_incorrect_pretty'],
            'Model_B': task['solutions']['B_incorrect_pretty'],
            'Model_C': task['solutions']['C_correct_ugly'],
        }
        
        voting_prompt = create_voting_prompt(task['prompt'], solutions_for_voting)
        
        # Get votes from all models
        st.write(f"  Collecting votes...")
        for model_name, (tokenizer, model, device) in models_loaded.items():
            vote_output = generate_vote(tokenizer, model, device, voting_prompt, 50, temperature_vote)
            
            if vote_output and not vote_output.startswith("Voting error"):
                extracted = extract_vote(vote_output, ['solutionA', 'solutionB', 'solutionC'])
                task_result['votes'][model_name] = {
                    'raw': vote_output,
                    'extracted': extracted
                }
                
                vote_display = extracted if extracted else "Invalid"
                st.write(f"    {model_name}: {vote_display}")
        
        # Determine winner
        valid_votes = [v['extracted'] for v in task_result['votes'].values() if v['extracted']]
        
        if valid_votes:
            from collections import Counter
            vote_counts = Counter(valid_votes)
            winner_tag = vote_counts.most_common(1)[0][0]
            task_result['winner'] = winner_tag
            
            # Check if they voted correctly (for solutionA)
            task_result['correct_vote'] = (winner_tag.lower() == 'solutionc')
            
            if task_result['correct_vote']:
                st.success(f"  ✅ Models voted for CORRECT solution (C)")
            else:
                st.error(f"  ❌ Models voted for INCORRECT solution ({winner_tag})")
        else:
            st.warning(f"  ⚠️ No valid votes")
        
        results.append(task_result)
        progress_bar.progress((idx + 1) / len(SYNTHETIC_TASKS))
    
    return results

def analyze_synthetic_results(results: List[Dict]):
    """Analyze synthetic test results"""
    
    df = pd.DataFrame(results)
    
    st.header("📊 Synthetic Voting Test Results")
    
    # Overall accuracy
    correct_votes = df['correct_vote'].sum()
    total_votes = len(df[df['winner'].notna()])
    accuracy = (correct_votes / total_votes * 100) if total_votes > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tasks", len(df))
    
    with col2:
        st.metric("Correct Votes", f"{correct_votes}/{total_votes}")
    
    with col3:
        st.metric(
            "Accuracy",
            f"{accuracy:.1f}%",
            delta="100% if correctness-based" if accuracy < 90 else None
        )
    
    # Vote distribution
    st.subheader("Vote Distribution")
    
    vote_counts = {
        'Solution C (Correct, Ugly)': 0,
        'Solution B (Incorrect, Pretty)': 0,
        'Solution A (Incorrect, Pretty)': 0,
        'No consensus': 0
    }
    
    for result in results:
        winner = result.get('winner', '').lower()
        if winner == 'solutionc':
            vote_counts['Solution C (Correct, Ugly)'] += 1
        elif winner == 'solutionb':
            vote_counts['Solution B (Incorrect, Pretty)'] += 1
        elif winner == 'solutiona':
            vote_counts['Solution A (Incorrect, Pretty)'] += 1
        else:
            vote_counts['No consensus'] += 1
    
    st.bar_chart(vote_counts)
    
    # Interpretation
    st.subheader("🔍 Interpretation")
    
    if accuracy > 80:
        st.success("""
        **Models vote based on CORRECTNESS**
        
        Models successfully identified the correct solution despite poor formatting.
        This suggests code evaluation is based on semantic correctness rather than appearance.
        """)
    elif accuracy < 20:
        st.error("""
        **Models vote based on APPEARANCE**
        
        Models consistently chose well-formatted incorrect solutions over correct ugly code.
        This suggests voting decisions are driven by superficial code characteristics.
        """)
    else:
        st.warning(f"""
        **Mixed Results ({accuracy:.1f}%)**
        
        Models show inconsistent evaluation behavior. Some tasks voted by correctness,
        others by appearance. Further analysis needed to identify patterns.
        """)
    
    # Detailed results
    st.subheader("📋 Detailed Results")
    
    display_df = pd.DataFrame([{
        'Task': r['task_id'],
        'Level': r['level'],
        'Winner': r.get('winner', 'No vote'),
        'Correct?': '✅' if r.get('correct_vote') else '❌',
        'DeepSeek Vote': r['votes'].get('DeepSeek Coder 7B Instruct', {}).get('extracted', 'N/A'),
        'CodeGemma Vote': r['votes'].get('CodeGemma 7B Instruct', {}).get('extracted', 'N/A'),
        'CodeLlama Vote': r['votes'].get('Code Llama 7B Instruct', {}).get('extracted', 'N/A'),
    } for r in results])
    
    st.dataframe(display_df, use_container_width=True)
    
    return df



def run_synthetic_test(temperature_vote: float = 0.0):
    """Run synthetic voting test"""
    
    # Load models
    st.header("Loading Models...")
    models_loaded = load_all_models()
    
    if not models_loaded:
        st.error("Failed to load models")
        return []
    
    st.success(f"Loaded {len(models_loaded)} models")
    
    results = []
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, task in enumerate(SYNTHETIC_TASKS):
        status_text.text(f"Testing task {idx + 1}/{len(SYNTHETIC_TASKS)}: {task['task_id']}")
        
        task_result = {
            'task_id': task['task_id'],
            'level': task['level'],
            'solutions': {},
            'votes': {},
            'ground_truth': 'A_incorrect_pretty',  # We know A is correct
            'winner': None,
            'correct_vote': False,
        }
        
        # Store solutions (they're pre-made)
        task_result['solutions'] = task['solutions']
        
        # Test all solutions to verify our ground truth
        st.write(f"  Testing solutions...")
        test_results = {}
        for sol_name, sol_code in task['solutions'].items():
            test_res = run_unit_test(sol_code, task['test'], task['entry_point'], timeout=10)
            test_results[sol_name] = test_res['passed']
            st.write(f"    {sol_name}: {'✅ PASS' if test_res['passed'] else '❌ FAIL'}")
        
        task_result['test_results'] = test_results
        
        # Verify only C is correct
        if test_results.get('A_incorrect_pretty', test_results.get('A_correct_ugly')) != True:
            st.error(f"⚠️ WARNING: Solution A failed for {task['task_id']}!")
        if test_results['B_incorrect_pretty'] != False:
            st.error(f"⚠️ WARNING: Solution B passed for {task['task_id']}!")
        if test_results['C_correct_ugly'] != False:
            st.error(f"⚠️ WARNING: Solution C passed for {task['task_id']}!")
        
        # Create voting prompt
        # Map solutions to solutionA, solutionB, solutionC format
        solutions_for_voting = {
            'Model_A': task['solutions']['A_incorrect_pretty'],
            'Model_B': task['solutions']['B_incorrect_pretty'],
            'Model_C': task['solutions']['C_correct_ugly'],
        }
        
        voting_prompt = create_voting_prompt(task['prompt'], solutions_for_voting)
        
        # Get votes from all models
        st.write(f"  Collecting votes...")
        for model_name, (tokenizer, model, device) in models_loaded.items():
            vote_output = generate_vote(tokenizer, model, device, voting_prompt, 50, temperature_vote)
            
            if vote_output and not vote_output.startswith("Voting error"):
                extracted = extract_vote(vote_output, ['solutionA', 'solutionB', 'solutionC'])
                task_result['votes'][model_name] = {
                    'raw': vote_output,
                    'extracted': extracted
                }
                
                vote_display = extracted if extracted else "Invalid"
                st.write(f"    {model_name}: {vote_display}")
        
        # Determine winner
        valid_votes = [v['extracted'] for v in task_result['votes'].values() if v['extracted']]
        
        if valid_votes:
            from collections import Counter
            vote_counts = Counter(valid_votes)
            winner_tag = vote_counts.most_common(1)[0][0]
            task_result['winner'] = winner_tag
            
            # Check if they voted correctly (for solutionA)
            task_result['correct_vote'] = (winner_tag.lower() == 'solutionc')
            
            if task_result['correct_vote']:
                st.success(f"  ✅ Models voted for CORRECT solution (C)")
            else:
                st.error(f"  ❌ Models voted for INCORRECT solution ({winner_tag})")
        else:
            st.warning(f"  ⚠️ No valid votes")
        
        results.append(task_result)
        progress_bar.progress((idx + 1) / len(SYNTHETIC_TASKS))
    
    return results

def analyze_synthetic_results(results: List[Dict]):
    """Analyze synthetic test results"""
    
    df = pd.DataFrame(results)
    
    st.header("📊 Synthetic Voting Test Results")
    
    # Overall accuracy
    correct_votes = df['correct_vote'].sum()
    total_votes = len(df[df['winner'].notna()])
    accuracy = (correct_votes / total_votes * 100) if total_votes > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tasks", len(df))
    
    with col2:
        st.metric("Correct Votes", f"{correct_votes}/{total_votes}")
    
    with col3:
        st.metric(
            "Accuracy",
            f"{accuracy:.1f}%",
            delta="100% if correctness-based" if accuracy < 90 else None
        )
    
    # Vote distribution
    st.subheader("Vote Distribution")
    
    vote_counts = {
        'Solution C (Correct, Ugly)': 0,
        'Solution B (Incorrect, Pretty)': 0,
        'Solution A (Incorrect, Pretty)': 0,
        'No consensus': 0
    }
    
    for result in results:
        winner = result.get('winner', '').lower()
        if winner == 'solutionc':
            vote_counts['Solution C (Correct, Ugly)'] += 1
        elif winner == 'solutionb':
            vote_counts['Solution B (Incorrect, Pretty)'] += 1
        elif winner == 'solutiona':
            vote_counts['Solution A (Incorrect, Pretty)'] += 1
        else:
            vote_counts['No consensus'] += 1
    
    st.bar_chart(vote_counts)
    
    # Interpretation
    st.subheader("🔍 Interpretation")
    
    if accuracy > 80:
        st.success("""
        **Models vote based on CORRECTNESS**
        
        Models successfully identified the correct solution despite poor formatting.
        This suggests code evaluation is based on semantic correctness rather than appearance.
        """)
    elif accuracy < 20:
        st.error("""
        **Models vote based on APPEARANCE**
        
        Models consistently chose well-formatted incorrect solutions over correct ugly code.
        This suggests voting decisions are driven by superficial code characteristics.
        """)
    else:
        st.warning(f"""
        **Mixed Results ({accuracy:.1f}%)**
        
        Models show inconsistent evaluation behavior. Some tasks voted by correctness,
        others by appearance. Further analysis needed to identify patterns.
        """)
    
    # Detailed results
    st.subheader("📋 Detailed Results")
    
    display_df = pd.DataFrame([{
        'Task': r['task_id'],
        'Level': r['level'],
        'Winner': r.get('winner', 'No vote'),
        'Correct?': '✅' if r.get('correct_vote') else '❌',
        'DeepSeek Vote': r['votes'].get('DeepSeek Coder 7B Instruct', {}).get('extracted', 'N/A'),
        'CodeGemma Vote': r['votes'].get('CodeGemma 7B Instruct', {}).get('extracted', 'N/A'),
        'CodeLlama Vote': r['votes'].get('Code Llama 7B Instruct', {}).get('extracted', 'N/A'),
    } for r in results])
    
    st.dataframe(display_df, use_container_width=True)
    
    return df

def main():
    st.set_page_config(page_title="Synthetic Voting Test", layout="wide")
    
    st.title("🧪 Synthetic Voting Test: Appearance vs Correctness")
    
    st.markdown("""
    This experiment tests whether models vote based on code **appearance** or **correctness**.
    
    **Setup:**
    - 10 coding tasks
    - Each has 3 solutions:
      - **Solution A**: INCORRECT but pretty (terse, no comments, cryptic variables)
      - **Solution B**: INCORRECT but pretty (well-formatted, comments, clear names)
      - **Solution C**: CORRECT but ugly (different approach, also styled)
    
    **Hypothesis:**
    - If models vote by appearance → Accuracy ≈ 0% (choose A or B)
    - If models vote by correctness → Accuracy ≈ 100% (choose C)
    """)
    
    # Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        temperature_vote = st.slider(
            "Voting Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )
        
        st.markdown("---")
        show_prompt = st.checkbox("Show Voting Prompts", value=False)
    
    # Preview section
    if show_prompt:
        st.header("🔍 Voting Prompt Preview")
        st.markdown("**This is exactly what models see when voting:**")
        
        # Show first task as example
        example_task = SYNTHETIC_TASKS[0]
        
        solutions_for_voting = {
            'Model_A': example_task['solutions']['A_incorrect_pretty'],
            'Model_B': example_task['solutions']['B_incorrect_pretty'],
            'Model_C': example_task['solutions']['C_correct_ugly'],
        }
        
        example_prompt = create_voting_prompt(example_task['prompt'], solutions_for_voting)
        
        st.code(example_prompt, language='text')
        
        st.info("💡 Models see NO labels like 'correct' or 'incorrect' - only the raw code!")
    
    if st.button("🚀 Run Synthetic Test", type="primary"):
        results = run_synthetic_test(temperature_vote)
        
        if results:
            df = analyze_synthetic_results(results)
            
            # Export
            st.subheader("💾 Export Results")
            
            export_data = {
                'test_type': 'synthetic_appearance_vs_correctness',
                'temperature_vote': temperature_vote,
                'results': results
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json_str,
                file_name="synthetic_voting_test.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()