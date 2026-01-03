import rc
import optimism
import ucrl

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
    
    def observe(self, time, state, action, holding_r, trans_r, sojourn_time, transition):
        pass

class PolicyAgent(Agent):
    def __init__(self, capacities, n_customer_levels, n_server_levels, policy, rng : np.random._generator.Generator):
        super().__init__(capacities, n_customer_levels, n_server_levels, rng)
        self.policy = policy

    def get_action(self, state):
        return self.policy.get_action(state)
    
    def observe(self, time, state, action, holding_r, trans_r, sojourn_time, transition):
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

        self.model = optimism.build_optimistic_model(self.parameter_estimator, self.model_bounds, self.initial_confidence_param, self.rng)
        self.policy,_ = self.model.get_optimal_policy()

    def state_idx(self, state):
        return state + self.model_bounds.capacities[1]

    def get_action(self, state):
        ext_action = self.policy.get_action(state)

        return (ext_action[0]//2, ext_action[1]//2)

    def observe(self, time, state, action, holding_r, trans_r, sojourn_time, transition):
        self.parameter_estimator.observe(self.state_idx(state), action, transition, sojourn_time, holding_r, trans_r)

        level = action[0] if transition == 1 else action[1]

        new_episode = self.exploration.observe(state, level, transition == 1)

        if new_episode:
            self.exploration.new_episode()
            confidence_param = self.initial_confidence_param / self.exploration.steps_before_episode
            self.model = optimism.build_optimistic_model(self.parameter_estimator, self.model_bounds, confidence_param, self.rng)
            self.policy, gain = self.model.get_optimal_policy()
            print(f"new policy, optimistic gain: {gain}")
            #print("----------------------------------------------")
            #self.parameter_estimator.print_rate_bounds(confidence_param) 
            #print("----------------------------------------------")
            #self.model.print_rates()
            #self.model.print_rewards()

class LearnersAgent(Agent):
    def __init__(self, capacities, n_customer_levels, n_server_levels, uni_constant, learner, model_bounds, rng : np.random._generator.Generator):
        super().__init__(capacities, n_customer_levels, n_server_levels, rng)

        self.uni_constant = uni_constant
        self.model_bounds = model_bounds
        self.learner = learner
        self.learner.reset(self.model_bounds.capacities[1])

        self.exploration = ucrl.Exploration(model_bounds)

        self.reward_norm = 5

    def get_action_from_idx(self, action_idx):
        cust_level = action_idx // self.model_bounds.n_levels[1]
        serv_level = action_idx %  self.model_bounds.n_levels[1]

        return (cust_level, serv_level)
    
    def get_idx_from_action(self, action):
        return (action[0]*self.model_bounds.n_levels[1]) + action[1]
        

    def get_action(self, state):
        return self.get_action_from_idx(self.learner.play(state))

    def normalize_reward(self, reward):
        reward = reward/self.reward_norm

        reward = max(reward, -self.reward_norm)
        reward = min(reward, self.reward_norm)

        return reward
    
    def observe(self, time, state, action, holding_r, trans_r, sojourn_time, transition):
        n_next_transitions = 1
        n_self_transitions = max(round(sojourn_time/self.uni_constant)-1,0)

        final_st = sojourn_time - self.uni_constant*n_self_transitions

        final_reward = (final_st*holding_r) + trans_r
        final_reward = self.normalize_reward(final_reward)
        sojourn_reward = self.normalize_reward(holding_r*self.uni_constant)

        new_episode = False

        state_idx = state + self.capacities[1]

        for i in range(n_self_transitions):
            self.learner.update(state_idx, action_idx, sojourn_reward, state)
            new_episode = self.exploration.observe(state_idx, action_idx) or new_episode

        action_idx = self.get_idx_from_action(action)

        self.learner.update(state_idx, action_idx, final_reward, state + transition)
        new_episode = self.exploration.observe(state_idx, action_idx) or new_episode
        
        if new_episode:
            self.exploration.new_episode()
            self.learner.new_episode()
