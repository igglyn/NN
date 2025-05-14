import numpy as np
from time import time

from neurray import Neurray

if __name__ == "__main__":
    args = (1, np.uint8, np.uint8)
    args2 = (2,)
    eliv = Neurray(*args)
    given = np.empty((args[0],args2[0]), dtype=args[1])
    target = np.empty((args[0],args2[0]), dtype=args[2])
    
    prev_result = None
    generations = 0
    mutations = 0

    given[...,0] = 69
    if args2[0] >= 2:
        given[...,1] = 42
    if args2[0] >= 3:
        given[...,2] = 0
    target[...,0] = 42
    if args2[0] >= 2:
        target[...,1] = 69
    if args2[0] >= 3:
        target[...,2] = 255

    eliv.init_array()
    eliv.set_training()

    #print(given, target)

    start = time()
    for i in range(50):
    #while 1:
    #if 1:
        result = eliv.forward(given)
        generations += 1
        if not prev_result is None and np.equal(result, target).all():
            #pass
            break
        else:
            prev_result = result
        eliv.backward(target)

    end = time()
    print(f"""{args[0]} neuron{'s' if args[0]-1 else ''} | {args2[0]} target{'s' if args2[0]-1 else ''} | {generations} pass{'es' if generations-1 else ''} | {round((end - start) * 1_000, 2)} ms""")
    print(eliv.forward(given))
    print(np.sum(np.unpackbits(eliv.forward(given) ^ target).reshape(args2[0],np.iinfo(args[2]).bits), 1))
   
    
