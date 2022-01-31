from collections import deque
from factorizetangle import normal_form
import numpy as np

def pairs(dot_bracket):
    dot_bracket_cleaned = dot_bracket.replace(".","")
    N_2 = len(dot_bracket_cleaned)
    N = N_2 // 2
    stack1 = deque() #()
    stack2 = deque() #[]
    stack3 = deque() #{}
    stacks = [(stack1, "(", ")"),(stack2, "[", "]"),(stack3, "{", "}")]
    pairs = []
    for i,b in enumerate(dot_bracket_cleaned):
        if b not in ["(",")","[","]","{","}"]:
            raise Exception("The only parentheses allowed are () , [] and {}")

        for stack, op, cl in stacks:
            if b == op:
                stack.append(i)
            elif b == cl:
                try:
                    j = stack.pop()
                except:
                    raise Exception("Invalid Dot-Bracket.")
                pairs.append([j+1,i+1])
    
    if len(stack1) + len(stack2) + len(stack3) != 0:
        raise Exception("Invalid Dot-Bracket.")

    return pairs

def abbreviate(pairs):
    abbreviated_pairs = pairs[:]


    for pair1 in pairs:
        for pair2 in pairs:
            if pair1 == pair2:
                continue
                
            a1,a2 = pair1
            b1,b2 = pair2
            if pair2 in abbreviated_pairs and a1 == b1 - 1 and a2 == b2 + 1:
                abbreviated_pairs.remove(pair2)
    return abbreviated_pairs

def dot_bracket_to_tangle(dot_bracket):
    p = pairs(dot_bracket)
    p = abbreviate(p)
    indexes = np.array(p).flatten()
    indexes = np.sort(indexes)
    for pair in p:
        pair[0] = np.where(indexes == pair[0])[0][0]+1
        pair[1] = np.where(indexes == pair[1])[0][0]+1


    N = len(p)

    for pair in p:
        if pair[0] > N:
            pair[0] -= (2*(pair[0] - N)-1)
            pair[0] = str(pair[0])
        else:
            pair[0] = str(pair[0]) + "'"

        if pair[1] > N:
            pair[1] -= (2*(pair[1] - N)-1)
            pair[1] = str(pair[1])
        else:
            pair[1] = str(pair[1]) + "'"

    return normal_form(p)