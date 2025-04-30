import numpy as np
from time import time

from neurray2 import Neurray, unpackbits

if __name__ == "__main__":
    args = (1, 1, np.uint8, np.uint8)
    #assert args[1] > 1
    eliv = Neurray(*args)
    given = np.empty((args[0],args[1]), dtype=args[2])
    target = np.empty((args[0],args[1]), dtype=args[3])
    
    prev_fitness = -1
    generations = 0
    mutations = 0

    given[0] = 42 
    #given[1] = 69

    target[0] = 69
    #target[1] = 42

    eliv.init_array()
    eliv.allocate_training()

    max_fitness = np.iinfo(args[3]).bits * args[0] * args[1]

    start = time()
    while 1:
        # NOTE flawed once we merge between 2+ target results

        result = eliv.forward(given)
        fitness = np.sum(unpackbits(~np.bitwise_xor(result, target)))
        generations += 1
        if fitness == max_fitness:
            mutations += 1
            break
        if fitness > prev_fitness:
            prev_fitness = fitness
            mutations += 1
            #print(eliv.forward(given))
        eliv.backward(target)

    end = time()
    print(f"{generations}/{mutations}")
    print(f"{round(end - start, 4)} seconds")
    #print(eliv.arrays)
    results = {}

    





