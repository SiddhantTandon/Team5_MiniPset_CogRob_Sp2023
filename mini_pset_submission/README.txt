# Team 5: Learning Models by Exploiting Structure and Fundamental Knowledege
# Notes for the mini-pset implementation

# FILES
# TO BE RELEASED
1. utils_pset.py: Implementation for generating data sets for the minipset - 
input vector (image frames) along with actions, rewards and true agent coordinates
to help make test cases and examples.

2. checker.py: Implements an assertion to match student's responses with the true
solutions present in solution_vals.py

3. solution_vals.py: True solutions for various test cases. Would be nice to
hide this file, from student's view, if we can. Otherwise the checker
will look for an input function

4. LearningWithExploitationPset.ipynb: The python notebook that has formatted
cells and tutorial for the students to work with.


# NOT TO BE RELEASED
1. LearningWithExploitationPset_SOLUTIONS.ipynb: This file contains the solution
implementation for the problems to be used for TAs reference.


# AUTOGRADING
The test cells are below the answer cells and have:
1. checker function for asserting the loss value
2. test_ok() function to give a visual feedback on successfull implementation
If possible the answer cells should be checked if students have not used the 
true answer from the solution_vals file instead of implementing the algorithm


# PROBLEMS
There are 3 problems which we have designed for the students. A tutorial
on both an actual implementation of a similar problem and taking gradient
has been provided for reference. 
The points are divided as follows:
Problem A: Temporal Cohesion - 15 points
Problem B: Causality Prior - 15 points
Problem C: Repeatability Prior - 20 points


# LIBRARY USED
1. NUMPY


