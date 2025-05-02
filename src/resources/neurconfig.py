import json
import time
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


length = 6



def opt(func):
    thing = list(product(func, repeat=length))# skip exploding the list into factorials and ultra pain
    count = len(thing)
    # commented out because... optionally if you hate storage space you can write all to a json. i fking LOVE doing that uselessly
    # pp = [[fn.__name__ for fn in perm] for perm in thing]
    # with open("tonywpm_fixed.json", "w") as fp:
    #     json.dump(pp, fp)
    # print(f"opt generated {count} combos")
    return set(thing)
    
functions = (NOR, NAND, AND, XOR, NXOR, OR)


print("Running opt()")
start = time.time()
arr = list(opt(functions))
print(f"tony took {time.time()-start} seconds. no greggs sadly")

working = [] 
working2 = [] #unused currently but i dont want to delete it if it's structural or something glueless

# Constraints:
#    - must loop back to START
#    - must change every step
#    - work with starting MAX and MIN with HALF* mask
#        - HALF meaning any mask at all


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

# dump again if u want after removing any duplicates in a cool 200000 years
# deduped = [[fn.__name__ for fn in seq] for seq in set(working)]
# with open("deduped.json", "w") as wf:
#     json.dump(deduped, wf)
# print(f"Deduped saved to deduped.json nowaying ({len(deduped)} sequences)")


# ['XOR', 'NAND', 'NXOR', 'NOR'] stuck
# ['NXOR', 'NOR', 'OR', 'NAND'] stuck
# ['NXOR', 'NOR', 'OR', 'XOR'] negitive
# ['XOR', 'NAND', 'AND', 'NXOR'] positive 
# ['NXOR', 'NOR', 'XOR', 'NAND'] stuck 
# ['XOR', 'NAND', 'AND', 'NOR'] stuck

# print(block, block2)
# print((block := XOR(block)), (block2 := XOR(block2)))
# print((block := NAND(block)), (block2 := NAND(block2)))
# print((block := AND(block)), (block2 := AND(block2)))
# print((block := NXOR(block)), (block2 := NXOR(block2)))

# linter h8s ''' ''' style comments so i changed it but feel free to tell me you LOVE ~'''*comments*'''~
