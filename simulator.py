import model
import agent
import observer

import numpy as np

class Simulator:
    def __init__(self, model, agent, observer, rng : np.random._generator.Generator):
        self.model = model
        self.agent = agent
        self.observer = observer
        self.rng = rng


        self.state = 0
        self.t = 0


    def step(self):
        action = self.agent.get_action(self.state)

        customer_rate = self.model.get_customer_rate(self.state, action[0])
        server_rate = self.model.get_server_rate(self.state, action[1])

        total_rate = customer_rate + server_rate

        sojourn_time = self.rng.exponential(scale = 1/total_rate)

        step = self.rng.choice([-1,1], p=[server_rate/total_rate, customer_rate/total_rate])

        holding_reward = self.model.generate_holding_reward(self.state)
        transition_reward = self.model.generate_transition_reward(self.state, action, step)

        self.state += step
        self.t += sojourn_time

        total_reward = holding_reward*sojourn_time + transition_reward

        self.agent.observe(holding_reward, transition_reward, sojourn_time, step)

        self.observer.observe(self.t, self.state, total_reward)

if __name__ == "__main__":
    rng = np.random.default_rng()
    model = model.generate_random_model((10,10), (5,5), 1, 10, rng)
    model.print_rates()

    policy = model.get_optimal_policy()

    optimal_agent = agent.PolicyAgent((10,10), 5,5, policy, rng)

    optimal_observer = observer.Observer()

    sim = Simulator(model, optimal_agent, optimal_observer, rng)

    for i in range(1000000):
        sim.step()

    print(f"empirical gain: {optimal_observer.empirical_gain()}")
    print(f"estimated gain: {optimal_agent.evaluate(model)}")
