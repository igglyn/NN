import numpy as np
from typing import Optional, Union

# NOTE torch doesn't support what is needed for this to function in the same capacity, locked into uint8


gen = np.random.default_rng()



# unpack into useful shapes
def unpackbits(x):
    xshape = list(x.shape)
    bits = np.iinfo(x.dtype).bits
    bytes = bits // 8
    output = np.unpackbits(x.astype(f">u{bytes}").view("u1")).reshape(xshape + [bits])
    return output


# Neuron Array
class Neurray:
    def __init__(self, width:int, input_dtype=np.uint8, output_dtype=np.uint8):
        self.hidden = np.empty((4,1,width), dtype=output_dtype)

        self.width = width
        self.itype = input_dtype
        self.otype = output_dtype

    def init_array(self, *args):
        if not len(args):
            self.match = gen.integers(np.iinfo(self.itype).max, size=(4, self.width), dtype=self.itype)
            self.emit = gen.integers(np.iinfo(self.otype).max, size=(4, self.width), dtype=self.otype)
        else:
            self.match = args[0]
            self.emit = args[1]

    @property
    def zeros(self) -> np.ndarray: return np.zeros((self.width,), dtype=self.otype)

    def forward(self, lop:np.ndarray) -> np.ndarray:
        # handle each of the 4 opts
        ## NOTE decide on what ops to use here I'm glueless
        self.hidden[0] = np.where(np.bitwise_and(lop, self.match[0]) == 0, self.emit[0], ~self.emit[0])
        self.hidden[1] = np.where(np.bitwise_and(lop, self.match[1]) == 0, self.emit[1], ~self.emit[1])
        self.hidden[2] = np.where(np.bitwise_and(lop, self.match[2]) == 0, self.emit[2], ~self.emit[2])
        self.hidden[3] = np.where(np.bitwise_and(lop, self.match[3]) == 0, self.emit[3], ~self.emit[3])

        # hardcoded zeros are biting me
        mask = np.copy(self.zeros)

        # apply each part of the mask (we are always stuck atm)
        mask ^= self.hidden[0, 0]
        mask &= self.hidden[1, 0]
        mask = ~mask
        mask ^= self.hidden[2, 0]
        mask = ~mask
        mask |= self.hidden[3, 0]
        mask = ~mask
        
        self.hidden2 = mask

        # NOTE the mask is a (self.width) sized array, the result should be merged together (XOR)
        # it is not being done here to remain flexable
        return mask


    def backward(self) -> None:
        ## This is seperate from effect to allow the skipping of calling unwanted backward passes
        # calculate via stimulation which should be edited
        # NOTE `- 1` is used because it can get stuck
        neuron_mask = np.where((self._effect == np.min(self._effect)).any() or (self._effect == np.max(self._effect)).any(), True, False)
        sub_neuron_mask = np.where((self._effects == np.min(self._effects, 0)).any() or (self._effects == np.max(self._effects, 0)).any(), True, False)
        emit_mask = np.bitwise_and(neuron_mask, sub_neuron_mask)
        match_mask = emit_mask
        # decay each of the final masks (unused as unsure of effect)
        #match_mask = np.bitwise_and(np.copy(emit_mask), gen.choice((True,False), size=(4,self.width))).astype(bool)
        #emit_mask = np.bitwise_and(emit_mask, gen.choice((True,False), size=(4,self.width))).astype(bool)



        match_hurl = gen.integers(np.iinfo(self.itype).max, size=(4, self.width), endpoint=True, dtype=self.itype)
        emit_hurl = gen.integers(np.iinfo(self.otype).max, size=(4, self.width), endpoint=True, dtype=self.otype)

        # TODO test if I can mutate the match and emit
        # TODO is it possible to figure out which bits to invert in emit instead of randomness
        self.match = np.where(match_mask, ~np.bitwise_xor(self.match, match_hurl), self.match)
        self.emit = np.where(emit_mask, ~np.bitwise_xor(self.emit, emit_hurl), self.emit)


    def effect(self) -> np.ndarray:
        self._effects = np.sum(unpackbits(np.bitwise_xor(self.hidden, np.reshape(self.hidden, (1,4,self.width)))), (1,3)) // 3
        # Above is considered the effects within the array, below is the effects of the array
        self._effect = np.sum(unpackbits(self.hidden2), 1)
        return self._effect

    @property
    def arrays(self):
        return self.match, self.emit




if __name__ == "__main__":
    # the np.uint8 is hardcoded within the neuron array in a few places
    # 1 neuron as that is current test, this section breaks down if it more than one
    args = (1, np.uint8, np.uint8)
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
    #if 1:
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
            #print(f"{generations}/{mutations}")
        # gambling on correct improvement
        eliv.backward()

    print(f"{generations}/{mutations}")
    print(eliv.arrays)





