import numpy as np

gen = np.random.default_rng()
randarray = lambda size, dtype: gen.integers(np.iinfo(dtype).max, size=size, dtype=dtype)

def unpackbits(x):
    bits = np.iinfo(x.dtype).bits
    bytes = bits // 8
    rshape = (*x.shape, bits)
    output = np.unpackbits(x.astype(f">u{bytes}").view("u1")).reshape(rshape)
    return output

class Neurray:
    def __init__(self, neurons:int, batch_size:int, idtype=np.uint8, odtype=np.uint8):
        self.neurons = neurons
        self.batch_size = batch_size
        self.idtype = idtype
        self.odtype = odtype

        self.can_train = False
        self.training = False

        self.match = np.empty((4, neurons), dtype=idtype)
        self.emit = np.empty((4, neurons), dtype=odtype)

        self.hidden_states = np.empty((4, neurons, batch_size), dtype=odtype)
        self.choices = np.empty((4, neurons, batch_size), dtype=bool)

    def allocate_training(self):
        self.can_train = True
        self.training = True

        self.givens = np.empty((self.neurons, self.batch_size), dtype=self.idtype)
        self.results = np.empty((self.neurons, self.batch_size), dtype=self.odtype)

        self.match_window = np.empty((4, self.neurons), dtype=self.idtype)
        self.emit_window = np.empty((4, self.neurons), dtype=self.odtype)

    def init_array(self, *args):
        if not len(args):
            self.match[...] = randarray((4, self.neurons), self.idtype)
            self.emit[...] = randarray((4, self.neurons), self.odtype)
        else:
            self.match[...] = args[0]
            self.emit[...] = args[1]

    @property
    def base(self) -> np.ndarray:
        return np.zeros((self.neurons, self.batch_size), dtype=self.odtype)

    def forward(self, inputs:np.ndarray) -> np.ndarray:

        # self.choices doesn't have to be stored for only inference
        # but the array is going to be created over and over again

        # NOTE could be collapsed if desired, but kept for ease of testing atm
        ## Something is preventing this from being collapsed atm

        self.choices[0] = (inputs & self.match[0]) == 0
        self.choices[1] = (inputs & self.match[1]) == 0
        self.choices[2] = (inputs & self.match[2]) == 0
        self.choices[3] = (inputs & self.match[3]) == 0

        self.hidden_states = np.where(self.choices, (emits := np.reshape(self.emit, (4,self.batch_size,1))), ~emits)


        outputs = np.copy(self.base)

        outputs ^= self.hidden_states[0]

        outputs &= self.hidden_states[1]
        outputs[...] = ~outputs

        outputs ^= self.hidden_states[2]
        outputs[...] = ~outputs

        outputs |= self.hidden_states[3]
        outputs[...] = ~outputs


        if self.training:
            self.givens[...] = inputs
            self.results[...] = outputs

        # Outputs are not merged into a single value and should be treated as masks on previous steps
        return outputs

    def backward(self, target:np.ndarray) -> None:
        assert self.can_train, "Properties needed for the backward pass have not been initalized yet!"
        assert self.training, "NN Inputs and NN Outputs are frozen, place this class in training mode first!"

        # (4, 4, neurons, batch_size, bits) -> (4, neurons, batch_size) // 3 -> (4, neurons) // batch_size
        states_effects = np.sum(np.sum(unpackbits(np.reshape(self.hidden_states, (4, 1, self.neurons, self.batch_size)) ^ np.reshape(self.hidden_states, (1, 4, self.neurons, self.batch_size))), (1,4)) // 3, 2) // self.batch_size

        # (neurons, batch_size, bits) -> (neurons) // batch_size
        neuron_effects = np.sum(unpackbits(self.results), (1,2)) // self.batch_size

        
        # Calculate all parts of the masks
        states_mask = (states_effects == np.min(states_effects, 0)) | (states_effects == np.max(states_effects, 0))
        neuron_mask = (neuron_effects == np.min(neuron_effects)) | (states_effects == np.max(neuron_effects))

        emit_decay  = gen.choice((True,False), size=(4,self.neurons))
        match_decay = gen.choice((True,False), size=(4,self.neurons))

        emit_mask  = states_mask & neuron_mask & emit_decay
        match_mask = states_mask & neuron_mask & match_decay

        # invert effects of matches that got inverted during forward pass
        match_states = np.where(self.choices, ~(matches := self.match.reshape(4,self.neurons,1)), matches)

        # Calculate diffs between expected and actual outputs
        expected_match = ~(self.givens ^ match_states)
        expected_emit  = ~(self.results ^ target)

        self.match_window[...] = expected_match[...,0]
        self.emit_window[...]  = expected_emit[...,0]
        for batch in range(self.batch_size-1):
            self.match_window[...] = ~(self.match_window ^ expected_match[...,batch+1])
            self.emit_window[...] = ~(self.emit_window ^ expected_emit[...,batch+1])

        self.match[...] = np.where(match_mask, ~(self.match ^ self.match_window), self.match)
        self.emit[...]  = np.where(emit_mask,  ~(self.emit ^ self.emit_window), self.emit)
