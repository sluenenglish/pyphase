import numpy as np


class phase_model(object):
    
    def __init__(self, T, pi):
        self.phases = pi.shape[1]
        self.k = 2 * self.phases - 1
        self.T = T
        self.pi = pi
        self.e = np.ones((T.shape[0] , 1))
        self.t = -T.dot(self.e)
        
    def _pdf(self, y):  
        return self.pi.dot(expm(self.T*y)).dot(self.t).reshape(-1)
    
    def pdf(self, data):
        return np.array([self._pdf(i) for i in data])
    
    def log_likelyhood(self, data):
        probs = self.pdf(data)
        log_prob = np.log(probs)
        return log_prob.sum()

    def aic(self, data):
        return 2 * self.k - 2 * self.log_likelyhood(data)
