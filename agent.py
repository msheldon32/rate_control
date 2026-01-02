import numpy as np

import random

class Agent:
    def __init__(self, capacities, n_customer_levels, n_server_levels, rng : np.random._generator.Generator):
        self.capacities = capacities
        self.n_customer_levels = n_customer_levels
        self.n_server_levels = n_server_levels


        self.rng = rng

    def get_action(self, state):
        return [random.randrange(self.n_customer_levels), random.randrange(self.n_server_levels)]
    
    def observe(self, holding_r, trans_r, sojourn_time, transition):
        pass

class PolicyAgent(Agent):
    def __init__(self, capacities, n_customer_levels, n_server_levels, policy, rng : np.random._generator.Generator):
        super().__init__(capacities, n_customer_levels, n_server_levels, rng)
        self.policy = policy

    def get_action(self, state):
        return self.policy.get_action(state)
    
    def observe(self, holding_r, trans_r, sojourn_time, transition):
        pass

    def evaluate(self, model):
        return model.evaluate_policy(self.policy)
