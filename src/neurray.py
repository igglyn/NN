import numpy as np
from typing import Optional

gen = np.random.default_rng()
randarray = lambda size, dtype: gen.integers(np.iinfo(dtype).max, size=size, dtype=dtype)


class Neurray:
    def __init__(self, neurons:int, idtype=np.uint8, odtype=np.uint8):
        self.neurons = neurons
        self.idtype = idtype
        self.odtype = odtype

        self.training = False

        self.match = np.empty((4, neurons, 1), dtype=idtype)
        self.emit = np.empty((4, neurons, 1), dtype=odtype)

    def set_training(self):
        self.training = True

    def init_array(self, *args):
        # either pass in the arrays or generate some
        # as if anyone else besides myself knows what a better than random array is
        if not len(args):
            self.match[...] = randarray((4, self.neurons, 1), self.idtype)
            self.emit[...] = randarray((4, self.neurons, 1), self.odtype)
        else:
            self.match[...] = args[0]
            self.emit[...] = args[1]

    @property
    def base(self) -> np.ndarray:
        return np.zeros((self.neurons, 1), dtype=self.odtype)

    def forward(self, inputs:np.ndarray) -> np.ndarray:

        # check for resonance (saving for backwards)
        self.choices = (inputs & self.match) == 0

        # invert mask based on choice made
        hidden_states = np.where(self.choices, self.emit, ~self.emit)

        # start from zero, applying each part
        outputs = \
                (((np.copy(self.base)
                ^ ~hidden_states[0])
                | ~hidden_states[1])
                | hidden_states[2]) \
                ^ hidden_states[3]

        # capture for backwards pass
        if self.training:
            self.givens = inputs
            self.results = outputs

        # Outputs are not merged into a single value and should be treated as masks on previous steps
        return outputs

    def backward(self, target:np.ndarray, neurons:Optional[np.ndarray]=None) -> None:
        assert self.training, "NN Inputs and Outputs are non existant, place this class in training mode first!"
        
        batch_size = self.results.shape[1]
        # check if neurons have already converged, inverting the result to use as a mask
        output_diff = self.results ^ target
        target = np.copy(target)
        
        # TODO isolate search to neurons if present

        # check if neurons already are converged
        ## useless with max batch size of 2
        for batch in range(batch_size-1):
            output_diff[...,0] |= output_diff[...,batch+1]


        neuron_convergance = (output_diff[...,0] != 0)

        # apply overide
        if not neurons is None:
            neuron_convergance |= neruons

        # filter out only the parts that need to be edited
        filtered_emit = self.emit[:,neuron_convergance]
        filtered_match = self.match[:,neuron_convergance]
        filtered_choices = self.choices[:,neuron_convergance]

        filtered_givens = self.givens[neuron_convergance]
        filtered_results = self.results[neuron_convergance]

        filtered_output_diff = output_diff[neuron_convergance, 0, None]

        filtered_target = target[neuron_convergance]
        filtered_inputs = self.givens[neuron_convergance]

        filtered_length = filtered_results.shape[0]


        # invert effects of matches that got inverted during forward pass
        match_states = np.where(filtered_choices, ~filtered_match, filtered_match)
        emit_states = np.where(filtered_choices, ~filtered_emit, filtered_emit)
        
        # Calculate diffs between expected and actual outputs
        expected_match = ~(filtered_givens ^ match_states)
        expected_emit  = (~(filtered_results ^ filtered_target)).reshape(1, *filtered_results.shape)

        ## NOTE None stops numpy from eating the dim / regerates the dim
        match_window = expected_match[...,0,None]
        emit_window = expected_emit[...,0,None]

        #states_window = emit_states[...,0,None]
        #target_window = filtered_target[...,0,None]

        input_window = filtered_inputs[...,0,None]
        matching_window = match_states[...,0,None]

        # handle batch size (this whole thing exists when it can't converge to more than 2 values)
        for batch in range(batch_size-1):
            next_match = expected_match[...,batch+1,None]
            next_emit = expected_emit[...,batch+1,None]

            #next_state = emit_states[...,batch+1,None]
            #next_target = filtered_target[...,batch+1,None]

            next_input = filtered_inputs[...,batch+1,None]
            next_matching = match_states[...,batch+1,None]
            # NOTE match_window will probably be replaced, at the moment it is matching emit_window because of no tetests for it

            # Don't touch (needed for correct output)
            match_window = (match_window & next_match) ^ ~(match_window | ~next_match)
            emit_window = (emit_window & next_emit) ^ ~(match_window | ~next_emit)

            # Edit these two (abandoned for now, sanity toll)
            #states_window = ~(states_window | ~next_state) | ~(states_window ^ next_state)
            #target_window = ~(target_window | ~next_target) | ~(target_window ^ next_target)
            
            # Don't touch (want to abuse this behavior later)
            input_window = (input_window & next_input) ^ ~(input_window | ~next_input)
            matching_window = (matching_window & next_matching) ^ ~(matching_window | ~next_matching)



        # select parts to mutate
        # NOTE this has no real test at the moment so we are just picking something
        
        # Works with batch_size 1, doesn't with higher
        #emit_loss_mask = (~(states_window ^ emit_window) ^ target_window) < (states_window ^ target_window)
        emit_loss_mask = gen.choice((True,False), size=(4,filtered_length,1))
        
        match_loss_mask = (~(matching_window ^ match_window) ^ input_window) < (matching_window ^ input_window)

        # update the neurons
        self.match[...,neuron_convergance,0,None] = np.where(match_loss_mask, ~(filtered_match ^ match_window), filtered_match)
        self.emit[...,neuron_convergance,0,None]  = np.where(emit_loss_mask,  ~(filtered_emit ^ emit_window), filtered_emit)



class Elivrge:
    def __init__(self, step_count:int):
        self.steps = None 

    @property
    def base(self) -> np.ndarray:
        return np.zeros((self.neurons, 1), dtype=self.dtype)

    def forward(self, array):
        batch_size = array.shape[1]


