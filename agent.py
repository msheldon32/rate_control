import rc
import optimism

import numpy as np

import random

INITIAL_CONFIDENCE_PARAM = 10

class Agent:
    def __init__(self, capacities, n_customer_levels, n_server_levels, rng : np.random._generator.Generator):
        self.capacities = capacities
        self.n_customer_levels = n_customer_levels
        self.n_server_levels = n_server_levels


        self.rng = rng

    def get_action(self, state):
        return [random.randrange(self.n_customer_levels), random.randrange(self.n_server_levels)]
    
    def observe(self, state, holding_r, trans_r, sojourn_time, transition):
        pass

class PolicyAgent(Agent):
    def __init__(self, capacities, n_customer_levels, n_server_levels, policy, rng : np.random._generator.Generator):
        super().__init__(capacities, n_customer_levels, n_server_levels, rng)
        self.policy = policy

    def get_action(self, state):
        return self.policy.get_action(state)
    
    def observe(self, state, holding_r, trans_r, sojourn_time, transition):
        pass

    def evaluate(self, model):
        return model.evaluate_policy(self.policy)

class RC_Agent(Agent):
    def __init__(self, capacities, n_customer_levels, n_server_levels, model_bounds, rng : np.random._generator.Generator):
        super().__init__(capacities, n_customer_levels, n_server_levels, rng)
        self.model_bounds = model_bounds

        self.parameter_estimator = rc.ParameterEstimator(model_bounds)
        self.exploration = rc.Exploration(model_bounds)

        self.initial_confidence_param = INITIAL_CONFIDENCE_PARAM

        self.model = optimism.build_optimistic_model(self.parameter_estimator, self.model_bounds, self.initial_confidence_param)
        self.policy = self.model.get_optimal_policy()

    def get_action(self, state):
        ext_action = self.policy.get_action(state)

        return (ext_action[0]//2, ext_action[1]//2)

    def observe(self, state, action, holding_r, trans_r, sojourn_time, transition):
        self.parameter_estimator.observe(state, action, transition, sojourn_time, holding_r, trans_r)

        level = action[0] if transition == 1 else action[1]

        new_episode = self.exploration.observe(state, level, transition == 1)

        if new_episode:
            self.exploration.new_episode()
            confidence_param = self.initial_confidence_param / self.exploration.steps_before_episode
            self.policy = optimism.build_optimistic_model(self.parameter_estimator, self.model_bounds, confidence_param)
