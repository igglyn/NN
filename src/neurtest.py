import numpy as np

from neurray import Neurray, unpackbits

if __name__ == "__main__":
    # the np.uint8 is hardcoded within the neuron array in a few places
    # 1 neuron as that is current test, this section breaks down if it more than one
    args = (1000, np.uint8, np.uint8)
    eliv = Neurray(*args)
    start = np.empty(args[0], dtype=args[1])
    target = np.empty(args[0], dtype=args[2])
    
    prev_fitness = -1
    result = None
    generations = 0
    mutations = 0

    start[...] = 42
    target[...] = 69

    eliv.init_array()


    while 1:
        # NOTE flawed once we merge between 2+ target results
        result = eliv.forward(start)
        fitness = np.sum(unpackbits(~np.bitwise_xor(result, target)))
        generations += 1
        # greater than, equal to to prevent getting stuck
        if fitness == np.iinfo(args[2]).bits * args[0]:
            mutations += 1
            break
        if fitness > prev_fitness:
            prev_fitness = fitness
            eliv.effect()
            mutations += 1
        # gambling on correct improvement
        eliv.backward(start, result, target)

    print(f"{generations}/{mutations}")
    #print(eliv.arrays)
    #results = {}

    #for i in range(0,256):
    #    res = int(eliv.forward(np.array((i,), dtype=args[1]))[0])
    #    if not res in results:
    #        results[res] = 1
    #    else:
    #        results[res] += 1
    #print(results)





