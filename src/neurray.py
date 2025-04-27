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
    def __init__(self, width:int, batch_size:int, input_dtype=np.uint8, output_dtype=np.uint8):
        self.hidden = np.empty((4,1,width,batch_size), dtype=output_dtype)

        self.width = width
        self.batch_size = batch_size
        self.itype = input_dtype
        self.otype = output_dtype

    def init_array(self, *args):
        if not len(args):
            self.match = gen.integers(np.iinfo(self.itype).max, size=(4, self.width, 1), dtype=self.itype)
            self.emit = gen.integers(np.iinfo(self.otype).max, size=(4, self.width, 1), dtype=self.otype)
        else:
            self.match = args[0]
            self.emit = args[1]

    @property
    def zeros(self) -> np.ndarray: return np.zeros((self.width, self.batch_size), dtype=self.otype)

    def forward(self, lop:np.ndarray) -> np.ndarray:
        # handle each of the 4 opts
        ## NOTE only using OR and AND, the max states is equal to 2^numAND, (1,2,4,8,16)
        self.hidden[0] = np.where(np.bitwise_and(lop, self.match[0]) == 0, self.emit[0], ~self.emit[0])
        self.hidden[1] = np.where(np.bitwise_and(lop, self.match[1]) == 0, self.emit[1], ~self.emit[1])
        self.hidden[2] = np.where(np.bitwise_and(lop, self.match[2]) == 0, self.emit[2], ~self.emit[2])
        self.hidden[3] = np.where(np.bitwise_and(lop, self.match[3]) == 0, self.emit[3], ~self.emit[3])

        mask = np.copy(self.zeros)

        mask ^= self.hidden[0,0]
        mask &= self.hidden[1,0]
        mask = ~mask
        mask ^= self.hidden[2,0]
        mask = ~mask
        mask |= self.hidden[3,0]
        mask = ~mask
        
        # NOTE the mask is a (self.width) sized array, the result should be merged together (XOR)
        # it is not being done here to remain flexable
        self.hidden2 = mask

        return mask

    def backward(self, lop:np.ndarray, target:np.ndarray) -> None:
        self._effects = np.sum(np.sum(unpackbits(np.bitwise_xor(self.hidden, np.reshape(self.hidden, (1,4,self.width,self.batch_size)))), (1,4)) // 3, (2)) // self.batch_size
        # Above is considered the effects within the array, below is the effects of the array
        self._effect = np.sum(unpackbits(self.hidden2), (1,2))

        neuron_mask = (self._effect == np.min(self._effect)) | (self._effect == np.max(self._effect))
        sub_neuron_mask = (self._effects == np.min(self._effects, 0)) | (self._effects == np.max(self._effects, 0))
        mask = (neuron_mask | sub_neuron_mask).reshape(4,self.width,1)

        def XNOR_across(x:np.ndarray) -> int:
            result = np.array((x[0],), dtype=x.dtype)
            for val in x[0:]:
                result = ~(result ^ val)
            return result

        match_lop = np.apply_along_axis(XNOR_across, 2, ~(lop ^ self.match))
        emit_target = np.apply_along_axis(XNOR_across, 1, ~(self.hidden2 ^ target))

        # decay each of the final masks
        match_mask = np.bitwise_and(np.copy(mask), gen.choice((True,False), size=(4,self.width,1))).astype(bool)
        emit_mask = np.bitwise_and(mask, gen.choice((True,False), size=(4,self.width,1))).astype(bool)

        self.match[...] = np.where(match_mask, ~np.bitwise_xor(self.match, match_lop), self.match)
        self.emit[...] = np.where(emit_mask, ~np.bitwise_xor(self.emit, emit_target), self.match)





    @property
    def arrays(self):
        return self.match, self.emit

