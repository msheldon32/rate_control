import matplotlib.pyplot as plt

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
    
    def get_regret(self, ideal_gain):
        regret = [0]

        for t, x in zip(self.step_times, self.step_rewards):
            g = (ideal_gain*t)
            regret.append(regret[-1] + (g-x))

        return regret
    
    def plot_regret(self, ideal_gain, color="r"):
        regret = self.get_regret()

        plt.plot(regret, color=color)

    def summarize(self, ideal_gain, timestep=10000):
        regret = self.get_regret()

        cum_reward = [0]

        for x in self.step_rewards:
            cum_reward.append(cum_reward[-1] + x)

        t = 1
        reward_tstep = []
        regret_tstep = []

        while t < len(cum_reward):
            reward_tstep.append(cum_reward[t])
            regret_tstep.append(regret[t])
            t += timestep

        return {
                "regret": regret_tstep,
                "reward": reward_tstep
                }
