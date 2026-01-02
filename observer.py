class Observer:
    def __init__(self):
        self.total_time = 0
        self.total_reward = 0
    
    def observe(self, time, state, reward):
        self.total_time = time
        self.total_reward += reward

    def empirical_gain(self):
        return self.total_reward / self.total_time
