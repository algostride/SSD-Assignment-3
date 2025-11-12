"""
This script generates a file named 'input_data.txt' containing N random integers.
Each integer is within the 32-bit signed range.
"""

import random

N = 2000
MIN_INT = -2**31
MAX_INT = 2**31 - 1

# Use explicit encoding for best practice
with open("input_data.txt", "w", encoding="utf-8") as f:
    for _ in range(N):
        f.write(f"{random.randint(MIN_INT, MAX_INT)}\n")

print(f"Created input_data.txt with {N} random integers.")
