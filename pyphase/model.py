import numpy as np


class phase_model(object):
    
    def __init__(self, phase_type_generator, initial_state):
        self.phases = initial_state.shape[1]
        self.number_of_parameters = 2 * self.phases - 1
        self.phase_type_generator = phase_type_generator
        self.initial_state = initial_state
        self.e = np.ones((self.phases , 1))
        self.exit_rate_vector = - phase_type_generator.dot(self.e)
        
    def _pdf(self, y):  
        return self.initial_state.dot(expm(self.phase_type_generator * y)).dot(self.exit_rate_vector).reshape(-1)
    
    def pdf(self, data):
        return np.array([self._pdf(i) for i in data])
    
    def log_likelyhood(self, data):
        probs = self.pdf(data)
        log_prob = np.log(probs)
        return log_prob.sum()

    def aic(self, data):
        return 2 * self.number_of_parameters - 2 * self.log_likelyhood(data)
