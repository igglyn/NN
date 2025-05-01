import numpy as np
from time import time

from neurray2 import Neurray, unpackbits

if __name__ == "__main__":
    args = (2, 2, np.uint8, np.uint8)
    eliv = Neurray(*args)
    given = np.empty((args[0],args[1]), dtype=args[2])
    target = np.empty((args[0],args[1]), dtype=args[3])
    
    prev_fitness = -1
    generations = 0
    mutations = 0

    given[0] = 42, 69
    given[1] = 24, 96

    target[0] = 69, 42
    target[1] = 96, 24

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
            break
        if fitness > prev_fitness:
            prev_fitness = fitness
        eliv.backward(target)

    end = time()
    print(f"""{args[0]} neuron{'s' if args[0]-1 else ''} | {args[1]} target{'s' if args[1]-1 else ''} | {generations} pass{'es' if generations-1 else ''} | {round((end - start) * 10_000, 1)} µs""")
    #print(eliv.forward(given))
