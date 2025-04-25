import numpy as np
from itertools import permutations, product

uint32_max = 4_294_967_295
uint32_half = 65_535

block = np.array([uint32_max], dtype=np.uint32)
block2 = np.array([0], dtype=np.uint32)
mask = np.array([uint32_half], dtype=np.uint32)

def OR(block): return np.bitwise_or(block, mask)
def XOR(block): return np.bitwise_xor(block, mask)
def AND(block): return np.bitwise_and(block, mask)
def NOR(block): return np.invert(np.bitwise_or(block, mask))
def NXOR(block): return np.invert(np.bitwise_xor(block, mask))
def NAND(block): return np.invert(np.bitwise_and(block, mask))
# XOR | AND = NXOR, removed redundants to lower answer space


length = 4


def opt(func):
    output = []
    thing = list(product(func, repeat=length))
    for iter in thing:
        output.extend(permutations(iter))

    return set(output)

functions = (NOR, NAND, AND, XOR, NXOR, OR)
arr = list(opt(functions))




working = []
working2 = []



"""
Constraints:
    - must loop back to START
    - must change every step
    - work with starting MAX and MIN with HALF* mask
        - HALF meaning any mask at all
"""

for i in arr:
    val, val2 = block, block2
    j = 0
    for step in i:
        j += 1
        prev, prev2 = np.copy(val), np.copy(val2)
        val, val2 = step(val), step(val2)
        #if val[0] == prev[0] or val2[0] == prev2[0]: break
        if (j != length and val2[0] == uint32_max) or \
        (j != length and val[0] == 0): break
        #if not((j == length and val[0] == uint32_max) and \
        #(j == length and val2[0] == 0)): break
    else:
        working.append(i)


print(len(working))
for i in working:
    if len(set(i)) == len(i):
        print([step.__name__ for step in i])
'''
"""
['XOR', 'NAND', 'NXOR', 'NOR'] stuck
['NXOR', 'NOR', 'OR', 'NAND'] stuck
['NXOR', 'NOR', 'OR', 'XOR'] negitive
['XOR', 'NAND', 'AND', 'NXOR'] positive 
['NXOR', 'NOR', 'XOR', 'NAND'] stuck 
['XOR', 'NAND', 'AND', 'NOR'] stuck
"""
print(block, block2)
print((block := XOR(block)), (block2 := XOR(block2)))
print((block := NAND(block)), (block2 := NAND(block2)))
print((block := AND(block)), (block2 := AND(block2)))
print((block := NXOR(block)), (block2 := NXOR(block2)))
#'''
