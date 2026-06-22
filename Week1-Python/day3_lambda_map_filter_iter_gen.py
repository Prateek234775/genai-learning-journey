# ============================================
# DAY 3 - Lambda + Map + Filter
#          + Iterators + Generators
# Author: PRATEEK KUMAR KUNTAL
# Date: 7 May 2026
# ============================================


# ──────────────────────────────────────────
# PART 1 - LAMBDA FUNCTIONS
# (Small one line functions)
# ──────────────────────────────────────────

print("===== PART 1: Lambda Functions =====")

# Normal function vs Lambda
def square_normal(x):
    return x * x

square_lambda = lambda x: x * x

print(f"Normal   : {square_normal(5)}")
print(f"Lambda   : {square_lambda(5)}")

# Lambda with 2 inputs
add      = lambda a, b: a + b
subtract = lambda a, b: a - b
multiply = lambda a, b: a * b

print(f"\nAdd      : {add(10, 5)}")
print(f"Subtract : {subtract(10, 5)}")
print(f"Multiply : {multiply(10, 5)}")

# Lambda with condition
check_even = lambda x: "Even" if x % 2 == 0 else "Odd"
print(f"\n4 is : {check_even(4)}")
print(f"7 is : {check_even(7)}")


# ──────────────────────────────────────────
# PART 2 - MAP FUNCTION
# (Apply a function to every item in list)
# ──────────────────────────────────────────

print("\n===== PART 2: Map Function =====")

numbers = [1, 2, 3, 4, 5]

# Square every number
squared = list(map(lambda x: x ** 2, numbers))
print(f"Original : {numbers}")
print(f"Squared  : {squared}")

# Double every number
doubled = list(map(lambda x: x * 2, numbers))
print(f"Doubled  : {doubled}")

# Convert names to uppercase
names = ["raj", "priya", "amit", "sneha"]
upper = list(map(lambda x: x.upper(), names))
print(f"\nOriginal : {names}")
print(f"Upper    : {upper}")

# Add rupee symbol to prices
prices = [100, 250, 399, 899]
with_symbol = list(map(lambda x: f"₹{x}", prices))
print(f"\nPrices   : {prices}")
print(f"Formatted: {with_symbol}")


# ──────────────────────────────────────────
# PART 3 - FILTER FUNCTION
# (Keep only items matching condition)
# ──────────────────────────────────────────

print("\n===== PART 3: Filter Function =====")

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Keep only even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"All     : {numbers}")
print(f"Evens   : {evens}")

# Keep only numbers greater than 5
big = list(filter(lambda x: x > 5, numbers))
print(f"Above 5 : {big}")

# Keep only passing students
students = [
    {"name": "Raj",   "marks": 85},
    {"name": "Priya", "marks": 92},
    {"name": "Amit",  "marks": 45},
    {"name": "Sneha", "marks": 78},
    {"name": "Jo",    "marks": 38},
]

passing = list(filter(lambda s: s["marks"] >= 50, students))
failing = list(filter(lambda s: s["marks"] < 50, students))

print(f"\nPassing students:")
for s in passing:
    print(f"  {s['name']} - {s['marks']} marks")

print(f"Failing students:")
for s in failing:
    print(f"  {s['name']} - {s['marks']} marks")


# ──────────────────────────────────────────
# PART 4 - ITERATORS
# (Objects you can loop through one by one)
# ──────────────────────────────────────────

print("\n===== PART 4: Iterators =====")

# Every list is iterable
numbers = [10, 20, 30, 40]

# Create an iterator from list
my_iter = iter(numbers)

# Get items one by one using next()
print(next(my_iter))   # 10
print(next(my_iter))   # 20
print(next(my_iter))   # 30
print(next(my_iter))   # 40

# What happens after all items are done?
try:
    print(next(my_iter))
except StopIteration:
    print(" No more items in iterator!")

# Build your own Iterator class
class CountUp:
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

print("\nCustom Iterator - Count 1 to 5:")
counter = CountUp(1, 5)
for num in counter:
    print(num, end=" ")
print()


# ──────────────────────────────────────────
# PART 5 - GENERATORS
# (Functions that produce values one by one)
# (Much more memory efficient than lists!)
# ──────────────────────────────────────────

print("\n===== PART 5: Generators =====")

# Normal function — returns all at once
def squares_list(n):
    result = []
    for i in range(1, n+1):
        result.append(i ** 2)
    return result

# Generator function — yields one at a time
def squares_generator(n):
    for i in range(1, n+1):
        yield i ** 2        # yield instead of return!

print("Normal function output:")
print(squares_list(5))

print("\nGenerator output:")
gen = squares_generator(5)
for val in gen:
    print(val, end=" ")
print()

# Generator is lazy — produces only when asked
print("\nGenerator one by one:")
gen2 = squares_generator(3)
print(next(gen2))    # 1
print(next(gen2))    # 4
print(next(gen2))    # 9

# Generator Expression (like list comprehension)
print("\nGenerator Expression:")
gen_exp = (x ** 2 for x in range(1, 6))
for val in gen_exp:
    print(val, end=" ")
print()

# Real use — infinite sequence generator
def infinite_counter(start=1):
    while True:
        yield start
        start += 1

print("\nInfinite Generator (first 5 values):")
counter = infinite_counter()
for _ in range(5):
    print(next(counter), end=" ")
print()


# ──────────────────────────────────────────
# MINI PROJECT - AI Dataset Generator
# Simulate generating a large dataset
# one record at a time using Generator
# ──────────────────────────────────────────

print("\n===== MINI PROJECT: AI Dataset Generator =====")

import random

# Generator that creates student data one by one
# (memory efficient - doesn't load all at once)
def student_data_generator(count):
    names    = ["Yukta", "Priya", "Sanvi", "Sneha", "Aarya",
                "Neha", "Ravi", "Pooja", "Vikram", "Ananya"]
    branches = ["AIML", "CSE", "ECE", "IT", "Mechanical"]

    for i in range(count):
        yield {
            "id"    : i + 1,
            "name"  : random.choice(names),
            "branch": random.choice(branches),
            "marks" : random.randint(35, 100)
        }

# Generate 10 students
gen = student_data_generator(10)

# Filter only passing (marks >= 50) using filter + lambda
all_students = list(gen)
passing = list(filter(lambda s: s["marks"] >= 50, all_students))
toppers = list(filter(lambda s: s["marks"] >= 80, all_students))

# Format marks with map + lambda
formatted = list(map(
    lambda s: f"{s['name']} ({s['branch']}) - {s['marks']} marks",
    all_students
))

print("All Students:")
for s in formatted:
    print(f"  {s}")

print(f"\n Total    : {len(all_students)}")
print(f" Passing  : {len(passing)}")
print(f" Toppers  : {len(toppers)}")
print(f" Failing  : {len(all_students) - len(passing)}")

