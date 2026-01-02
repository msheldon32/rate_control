class Observer:
    def __init__(self):
        self.total_time = 0
        self.total_reward = 0

        self.step_rewards = []
        self.step_times = []
    
    def observe(self, time, state, reward):
        self.step_rewards.append(reward)
        self.step_times.append(time - self.total_time)

        self.total_time = time
        self.total_reward += reward

    def empirical_gain(self):
        return self.total_reward / self.total_time

    def trailing_gain(self, trailing_steps):
        t = sum(self.step_times[-trailing_steps:])
        r = sum(self.step_rewards[-trailing_steps:])

        return r/t
