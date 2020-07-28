import numpy as np

class Minerva2:
    def __init__(self, features_per_trace):
        self.features_per_trace = features_per_trace
        self.model = None
        
    def get_activation(self, probe, trace):
        '''
        also returns similarity in 2nd return value
        '''
        # the @ symbol denotes a "dot product," which is the same as 
        # sum([probe[i] * self.model[trace_idx][i] for i in range(len(probe))])
        similarity = (probe @ trace) / len(probe)
        return similarity**3, similarity
        
    def get_activation_by_idx(self, probe, trace_idx):
        return self.get_activation(probe, self.model[trace_idx], return_sim)
    
    def get_echo_intensity(self, probe, return_all=False):
        activations = []
        similarities = []
        for trace in self.model:
            activation, similarity = self.get_activation(probe, trace)
            activations.append(activation)
            similarities.append(similarity)
        
        if return_all:
            return sum(activations) / len(activations), activations, similarities
        else:
            return sum(activations) / len(activations)
        
    def get_echo_intensities(self, probes):
        if type(probes) != np.ndarray:
            raise Exception("Probes are not of type numpy array, fail.")
        model_arr = np.array(self.model)
        similarities = (probes @ model_arr.T)/probes.shape[1]
        activations = similarities**3
        intensities = np.mean(activations, axis=1)
        return intensities
    
    def add_trace(self, trace):
        if type(trace) != np.ndarray:
            raise Exception("Trace is not of type numpy array, fail.")
        if trace.shape != (self.features_per_trace,):
            raise Exception("Trace is not a one-dimensional array of length", self.features_per_trace, ", fail.")
        if len([x for x in trace if x not in (-1, 0, 1)]) > 0:
            raise Exception("Trace contains values besides -1, 0, or 1, fail.")
        reshaped = np.reshape(trace, (-1, self.features_per_trace))
        if self.model is not None:
            self.model = np.append(self.model, reshaped, axis=0)
        else:
            self.model = reshaped
    
    def add_traces(self, traces):
        if type(traces) != np.ndarray:
            raise Exception("Trace is not of type numpy array, fail.")
        if traces.shape[1] != self.features_per_trace:
            raise Exception("Trace is not a two-dimensional array of width", self.features_per_trace, ", fail.")
        if len([x for x in traces.flatten() if x not in (-1, 0, 1)]) > 0:
            raise Exception("Trace contains values besides -1, 0, or 1, fail.")
        if self.model is not None:
            self.model = np.append(self.model, traces, axis=0)
        else:
            self.model = traces

    def pretty_print_echo_intensity(self, probe):
        '''
        Illustrates the calculation of echo intensity
        '''
        echo_intensity, activations, similarities = self.get_echo_intensity(probe, return_all=True)

        print('PROBE:', list(probe))
        for i in range(len(activations)):
            print('TRACE {}:'.format(i), list(self.model[i]), '->', '{:>6.3f}^3 = {:>8.3f}'.format(similarities[i], activations[i]))
        print('-'*80)
        print('{:>80.3f}'.format(echo_intensity))