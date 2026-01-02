import math
import copy
import random

import model

import numpy as np
from scipy.stats import chi2

class ParameterEstimator:
    def __init__(self, model_bounds):
        self.model_bounds = model_bounds

        self.positive_sojourn_times = [[[] for j in range(model_bounds.n_levels[0])] for i in range(model_bounds.n_states)]
        self.negative_sojourn_times = [[[] for j in range(model_bounds.n_levels[1])] for i in range(model_bounds.n_states)]

        self.positive_clock = [[0 for j in range(model_bounds.n_levels[0])] for i in range(model_bounds.n_states)]
        self.negative_clock = [[0 for j in range(model_bounds.n_levels[1])] for i in range(model_bounds.n_states)]

        self.level_ct = sum(self.model_bounds.n_levels)

        self.state_counts = [0 for j in range(model_bounds.n_states)]

        self.cum_h_reward = [0 for i in range(model_bounds.n_states)]

        self.cum_cust_rewards = [[0 for j in range(model_bounds.n_levels[0])] for i in range(model_bounds.n_states)]
        self.cum_serv_rewards = [[0 for j in range(model_bounds.n_levels[1])] for i in range(model_bounds.n_states)]

    def observe(self, state, action, step, time_elapsed, holding_reward, transition_reward):
        self.positive_clock[state] += time_elapsed
        self.negative_clock[state] += time_elapsed

        self.state_counts[state] += 1

        is_positive = (step == 1)

        self.cum_h_reward[state] += holding_reward

        if is_positive:
            self.positive_sojourn_times[state][action[0]].append(self.positive_clock[state])
            self.positive_clock[state] = 0
            
            self.cum_cust_rewards[state][action[0]] += transition_reward
        else:
            self.negative_sojourn_times[state][action[1]].append(self.negative_clock[state])
            self.negative_clock[state] = 0

            self.cum_serv_rewards[state][action[1]] += transition_reward

    def get_count(self, state, level, is_positive):
        if is_positive:
            return len(self.positive_sojourn_times[state][level])
        return len(self.negative_sojourn_times[state][level])

    def transition_reward_bounds(self, state, level, confidence_param, is_positive):
        ct = self.get_count(state, level, is_positive)
        if ct == 0:
            return [-1, 1]
        if is_positive:
            point_estimate = self.cum_cust_rewards[state][level]/ct
        else:
            point_estimate = self.cum_cust_rewards[state][level]/ct

        epsilon = math.sqrt((math.log((self.model_bounds.n_states*self.level_ct)/confidence_param))/(2*max(1,ct)))

        return [max(-1, point_estimate-epsilon), min(1, point_estimate+epsilon)]
    
    def holding_reward_bounds(self, state, confidence_param):
        ct = self.state_counts[state]

        if ct == 0:
            return [-1, 1]
        point_estimate = self.cum_h_reward[state]/ct

        epsilon = math.sqrt((math.log((self.model_bounds.n_states*self.level_ct)/confidence_param))/(2*max(1,ct)))

        return [max(-1, point_estimate-epsilon), min(1, point_estimate+epsilon)]

    def sojourn_time_estimate(self, state, level, confidence_param, is_positive):
        acc = 0
        min_rate = self.model_bounds.rate_lb

        ct = self.get_count(state, level, is_positive)
        if ct == 0:
            return min_rate

        times = self.positive_sojourn_times[state][level] if is_positive else self.negative_sojourn_times[state]

        for i, stime in enumerate(times):
            truncation = math.sqrt(2*(i+1)/(math.pow(min_rate,2)*max(math.log((self.model_bounds.n_states*self.level_ct)/confidence_param),0.00001)))
            if stime <= truncation:
                acc += stime
        return acc/ct

    def sojourn_time_epsilon(self, state, level, confidence_param, is_positive):
        ct = self.get_count(state, level, is_positive)
        if ct == 0:
            return float("inf")

        inner_term = (2/max(1, ct))*max(math.log((self.model_bounds.n_states*self.level_ct)/confidence_param),0.00001)

        min_rate = self.model_bounds.rate_lb

        return (4/min_rate)*math.sqrt(inner_term)
    
    def get_naive_rate_bounds(self, state, level, is_positive):
        lbound = self.model_bounds.rate_lb
        rbound = self.model_bounds.rate_ub

        return [lbound, rbound]

    def transition_rate_bounds(self, state, level, confidence_param, is_positive):
        ct = self.get_count(state, level, is_positive)

        if ct == 0:
            return self.get_naive_rate_bounds(state, level, is_positive)

        st = self.sojourn_time_estimate(state, confidence_param, is_positive)
        ste = self.sojourn_time_epsilon(state, confidence_param, is_positive)

        min_rate, max_rate = self.get_naive_rate_bounds(state, is_positive)

        stime_lb = min(max(st-ste, 1/max_rate), 1/min_rate)
        stime_ub = min(max(st+ste, 1/max_rate), 1/min_rate)

        return [1/stime_ub, 1/stime_lb]

class Exploration:
    def __init__(self, model_bounds: model.ModelBounds):
        self.model_bounds = model_bounds
        self.level_ct = sum(self.model_bounds.n_levels)
        self.state_visit_counts = [[0 for j in range(self.level_ct)] for i in range(self.model_bounds.n_states)]
        self.state_visit_counts_in_episode = [[0 for j in range(self.level_ct)] for i in range(self.model_bounds.n_states)]
        self.steps_before_episode = 1
        self.n_episodes = 0

    def observe(self, state: int, level, is_positive) -> bool:
        level_idx = level
        if not is_positive:
            level_idx += self.model_bounds.n_levels[0]
        self.state_visit_counts[state][level_idx] += 1
        self.state_visit_counts_in_episode[state][level_idx] += 1

        return (2*self.state_visit_counts_in_episode[state][level_idx]) >= self.state_visit_counts[state][level_idx]

    def new_episode(self):
        self.state_visit_counts_in_episode = [[0 for j in range(self.level_ct)] for i in range(self.model_bounds.n_states)]

        self.steps_before_episode = sum([sum(x) for x in self.state_visit_counts])
        self.n_episodes += 1
