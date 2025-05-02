import numpy as np

gen = np.random.default_rng()
randarray = lambda size, dtype: gen.integers(np.iinfo(dtype).max, size=size, dtype=dtype)

def unpackbits(x):
    """
    Reshapes the np.unpackbits function from a 1d binary array into the orginal dims + bits
    """
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

        self.training = False

        self.match = np.empty((4, neurons), dtype=idtype)
        self.emit = np.empty((4, neurons), dtype=odtype)

        # Could be local if no backwards pass, but decided to always handle it here
        self.choices = np.empty((4, neurons, batch_size), dtype=bool)

    def allocate_training(self):
        self.training = True

        # inputs and outputs
        self.givens = np.empty((self.neurons, self.batch_size), dtype=self.idtype)
        self.results = np.empty((self.neurons, self.batch_size), dtype=self.odtype)

        # batch_size convergance, opted to just keep these around
        self.match_window = np.empty((4, self.neurons), dtype=self.idtype)
        self.emit_window = np.empty((4, self.neurons), dtype=self.odtype)

    def init_array(self, *args):
        # either pass in the arrays or generate some
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

        match = np.reshape(self.match, (4, self.neurons, 1))

        # check for resonance
        self.choices[...] = (inputs & match) == 0

        # invert mask based on choice made
        hidden_states = np.where(self.choices, (emits := np.reshape(self.emit, (4,self.neurons,1))), ~emits)

        # start from zero, applying each part
        ## updating in place when possible
        outputs = np.copy(self.base)

        outputs ^= hidden_states[0]
        outputs[...] = ~outputs

        outputs |= hidden_states[1]
        outputs[...] = ~outputs

        outputs |= hidden_states[2]

        outputs ^= hidden_states[3]

        # capture for backwards pass
        if self.training:
            self.givens[...] = inputs
            self.results[...] = outputs

        # Outputs are not merged into a single value and should be treated as masks on previous steps
        return outputs

    def backward(self, target:np.ndarray) -> None:
        assert self.training, "NN Inputs and Outputs are non existant, place this class in training mode first!"

        # (neurons, batch_size, bits) -> (neurons) 
        neuron_convergance = np.reshape(np.sum(unpackbits(self.results ^ target), (1,2)) != 0, (1, self.neurons))

        # select parts to mutate
        emit_decay  = gen.choice((True,False), size=(4,self.neurons))
        match_decay = gen.choice((True,False), size=(4,self.neurons))

        # select parts to add noise to
        emit_hurl_decay  = gen.choice((True,False), size=(4,self.neurons))
        match_hurl_decay = gen.choice((True,False), size=(4,self.neurons))
        
        # mask out completed neurons
        emit_loss_mask  = emit_decay & neuron_convergance
        match_loss_mask = match_decay & neuron_convergance

        emit_hurl_mask = emit_decay & emit_hurl_decay & neuron_convergance
        match_hurl_mask = match_decay & emit_hurl_decay & neuron_convergance

        # invert effects of matches that got inverted during forward pass
        match_states = np.where(self.choices, ~(matches := self.match.reshape(4,self.neurons,1)), matches)

        # Calculate diffs between expected and actual outputs
        expected_match = ~(self.givens ^ match_states)
        expected_emit  = ~(self.results ^ target)

        # handle batch size (this whole thing exists when it can't converge to more than 2 values)
        self.match_window[...] = expected_match[...,0]
        self.emit_window[...]  = expected_emit[...,0]
        for batch in range(self.batch_size-1):
            self.match_window[...] = (self.match_window & expected_match[...,batch+1])
            self.emit_window[...] = (self.emit_window & expected_emit[...,batch+1])

        # apply weight decay
        self.match_window[...] = np.where(match_hurl_decay, (self.match_window | randarray((4, self.neurons), self.idtype)), self.match_window)
        self.emit_window[...] = np.where(emit_hurl_decay, (self.emit_window | randarray((4, self.neurons), self.odtype)), self.emit_window)

        # update the neurons
        self.match[...] = np.where(match_loss_mask, ~(self.match ^ self.match_window), self.match)
        self.emit[...]  = np.where(emit_loss_mask,  ~(self.emit ^ self.emit_window), self.emit)
