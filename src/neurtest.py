import numpy as np
from time import time

from neurray import Neurray

def unpackbits(x):
    """
    Reshapes the np.unpackbits function from a 1d binary array into the orginal dims + bits
    """
    bits = np.iinfo(x.dtype).bits
    bytes = bits // 8
    rshape = (*x.shape, bits)
    output = np.unpackbits(x.astype(f">u{bytes}").view("u1")).reshape(rshape)
    return output

if __name__ == "__main__":
    args = (10000, 2, np.uint8, np.uint8)
    eliv = Neurray(*args)
    given = np.empty((args[0],args[1]), dtype=args[2])
    target = np.empty((args[0],args[1]), dtype=args[3])
    
    prev_fitness = -1
    generations = 0
    mutations = 0

    given[...,0] = 69
    given[...,1] = 42
    target[...,0] = 42
    target[...,1] = 69

    eliv.init_array()
    eliv.allocate_training()

    max_fitness = np.iinfo(args[3]).bits * args[0] * args[1]

    start = time()
    while 1:
        result = eliv.forward(given)
        fitness = np.sum(unpackbits(~np.bitwise_xor(result, target)))
        generations += 1
        # The NN should actually stop mutating once it reaches this point
        if fitness == max_fitness:
            break
        #if fitness > prev_fitness:
        #    # debugging
        #    prev_fitness = fitness
        eliv.backward(target)

    end = time()
    print(f"""{args[0]} neuron{'s' if args[0]-1 else ''} | {args[1]} target{'s' if args[1]-1 else ''} | {generations} pass{'es' if generations-1 else ''} | {round((end - start) * 1_000, 2)} ms""")
   
    
    results = {}
    for i in range(np.iinfo(args[2]).max):
        index = int(eliv.forward(np.array((i,), dtype=args[2]))[0,0])
        if index in results:
            results[index] += 1
        else:
            results[index] = 0
    #print(len(results))
