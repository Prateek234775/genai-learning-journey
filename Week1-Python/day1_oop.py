# ============================================
# DAY 1 - Python OOP Practice
# Topic: Classes, Objects, Inheritance
# Author: PRATEEK KUMAR KUNTA
# Date: 5 May 2026
# ============================================


# ──────────────────────────────────────────
# CONCEPT 1 - Basic Class & Object
# ──────────────────────────────────────────

class Student:
    # Class variable - shared by all students
    college = "AIML College"

    # Constructor - runs when object is created
    def __init__(self, name, branch, year):
        self.name = name        # instance variable
        self.branch = branch
        self.year = year

    # Method
    def introduce(self):
        print(f"Hi! I am {self.name}")
        print(f"Branch: {self.branch}")
        print(f"Year: {self.year}")
        print(f"College: {self.college}")

    def study(self, subject):
        print(f"{self.name} is studying {subject}")


# Creating objects
s1 = Student("Raj", "AIML", "2nd Year")
s2 = Student("Priya", "CSE", "3rd Year")

print("===== CONCEPT 1: Basic Class =====")
s1.introduce()
print()
s2.introduce()
print()
s1.study("Python OOP")
print()


# ──────────────────────────────────────────
# CONCEPT 2 - Class vs Instance Variables
# ──────────────────────────────────────────

class Phone:
    # Class variable
    brand = "Samsung"

    def __init__(self, model, price):
        # Instance variables
        self.model = model
        self.price = price

    def show_details(self):
        print(f"Brand: {Phone.brand}")
        print(f"Model: {self.model}")
        print(f"Price: ₹{self.price}")


print("===== CONCEPT 2: Class vs Instance Variables =====")
p1 = Phone("Galaxy S23", 75000)
p2 = Phone("Galaxy A54", 38000)
p1.show_details()
print()
p2.show_details()
print()


# ──────────────────────────────────────────
# CONCEPT 3 - Inheritance
# (Child class gets all features of Parent)
# ──────────────────────────────────────────

# Parent class
class Animal:
    def __init__(self, name, sound):
        self.name = name
        self.sound = sound

    def speak(self):
        print(f"{self.name} says {self.sound}!")

    def eat(self):
        print(f"{self.name} is eating.")


# Child class - inherits from Animal
class Dog(Animal):
    def __init__(self, name):
        # Calling parent constructor
        super().__init__(name, "Woof")

    # Extra method only Dog has
    def fetch(self):
        print(f"{self.name} is fetching the ball! 🎾")


# Child class
class Cat(Animal):
    def __init__(self, name):
        super().__init__(name, "Meow")

    # Extra method only Cat has
    def purr(self):
        print(f"{self.name} is purring... 😺")


print("===== CONCEPT 3: Inheritance =====")
dog = Dog("Bruno")
cat = Cat("Whiskers")

dog.speak()       # from Animal
dog.eat()         # from Animal
dog.fetch()       # only Dog has this
print()
cat.speak()       # from Animal
cat.eat()         # from Animal
cat.purr()        # only Cat has this
print()


# ──────────────────────────────────────────
# CONCEPT 4 - Real World Mini Project
# Bank Account System
# ──────────────────────────────────────────

class BankAccount:
    bank_name = "Python Bank"
    total_accounts = 0  # tracks how many accounts created

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        BankAccount.total_accounts += 1
        self.account_number = BankAccount.total_accounts

    def deposit(self, amount):
        if amount <= 0:
            print(" Deposit amount must be positive!")
        else:
            self.balance += amount
            print(f" ₹{amount} deposited successfully!")
            print(f" New Balance: ₹{self.balance}")

    def withdraw(self, amount):
        if amount <= 0:
            print(" Withdrawal amount must be positive!")
        elif amount > self.balance:
            print(" Insufficient balance!")
            print(f" Available Balance: ₹{self.balance}")
        else:
            self.balance -= amount
            print(f" ₹{amount} withdrawn successfully!")
            print(f" Remaining Balance: ₹{self.balance}")

    def show_details(self):
        print(f" Bank: {BankAccount.bank_name}")
        print(f" Account Holder: {self.owner}")
        print(f" Account Number: ACC{self.account_number:03d}")
        print(f" Balance: ₹{self.balance}")


# SavingsAccount inherits from BankAccount
class SavingsAccount(BankAccount):
    def __init__(self, owner, balance=0, interest_rate=5):
        super().__init__(owner, balance)
        self.interest_rate = interest_rate

    def add_interest(self):
        interest = (self.balance * self.interest_rate) / 100
        self.balance += interest
        print(f" Interest of ₹{interest} added at {self.interest_rate}%")
        print(f" New Balance: ₹{self.balance}")


print("===== CONCEPT 4: Real World Bank System =====")

# Normal account
acc1 = BankAccount("Raj", 5000)
acc1.show_details()
print()
acc1.deposit(2000)
print()
acc1.withdraw(1000)
print()
acc1.withdraw(10000)
print()

# Savings account
acc2 = SavingsAccount("Priya", 10000, interest_rate=6)
print()
acc2.show_details()
print()
acc2.add_interest()
print()

print(f" Total Accounts Created: {BankAccount.total_accounts}")
print()

