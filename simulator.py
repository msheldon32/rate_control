import model
import agent
import observer

from learners.KL_UCRL import KL_UCRL

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

        total_reward = holding_reward*sojourn_time + transition_reward

        self.agent.observe(self.t + sojourn_time, self.state, action, holding_reward, transition_reward, sojourn_time, step)

        self.state += step
        self.t += sojourn_time

        self.observer.observe(self.t, self.state, total_reward)

if __name__ == "__main__":
    rng = np.random.default_rng()
    model_bounds = model.ModelBounds((50,50), (3,3), 1, 5)
    model = model.generate_random_model(model_bounds, rng)
    model.print_rates()
    model.print_rewards()

    policy, _ = model.get_optimal_policy(n_iterations=1000)
    input("continue...")
    

    optimal_agent = agent.PolicyAgent((50,50), 3,3, policy, rng)
    rc_agent = agent.RC_Agent((50, 50),3,3, model_bounds, rng)
    kl_ucrl = KL_UCRL(model_bounds.n_states, model_bounds.n_actions, 0.05)
    kl_agent = agent.LearnersAgent((50,50), 3, 3, 1/2, kl_ucrl, model_bounds, rng)

    optimal_observer = observer.Observer()
    rc_observer = observer.Observer()
    kl_observer = observer.Observer()

    sim = Simulator(model, optimal_agent, optimal_observer, rng)
    rc_sim = Simulator(model, rc_agent, rc_observer, rng)
    kl_sim = Simulator(model, kl_agent, kl_observer, rng)

    for i in range(1000000000):
        if i % 1000 == 0 and i != 0:
            print(f"(optimal) trailing gain: {optimal_observer.trailing_gain(1000)}")
            print(f"(rc) trailing gain: {rc_observer.trailing_gain(1000)}")
            #print(f"(kl) trailing gain: {kl_observer.trailing_gain(1000)}")
        sim.step()
        rc_sim.step()
        #kl_sim.step()

    print(f"empirical gain: {optimal_observer.empirical_gain()}")
    print(f"learning gain: {rc_observer.empirical_gain()}")
    print(f"estimated gain: {optimal_agent.evaluate(model)}")
