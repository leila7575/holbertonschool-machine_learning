#!/usr/bin/env python3


import sys


exit_cases = {'exit', 'quit', 'goodbye', 'bye'}
while True:
    question = input("Q: ")
    if question.lower() in exit_cases:
        print('A: Goodbye')
        sys.exit()
    else:
        print('A: ')
