import numpy as np

class Uniform():
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def sample(self, n):
        return np.random.uniform(self.min, self.max, n)
    
class Sine():
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def sample(self, n):
        return np.arcsin(np.random.uniform(np.sin(self.min), np.sin(self.max), n))

# ra = eval(sky_distribution['ra']).sample(len(injections))
# dec = eval(sky_distribution['dec']).sample(len(injections))
# sky_locations = [{'ra': ra[i], 'dec': dec[i]} for i in range(len(injections))]