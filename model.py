import policy

import numpy as np
import matplotlib.pyplot as plt
import random

from util import *

class ModelRewards:
    def __init__(self, holding_rewards, customer_rewards, server_rewards, capacities, noise=2, deterministic=False, service_process =None):
        self.holding_rewards = holding_rewards
        self.customer_rewards = customer_rewards
        self.server_rewards = server_rewards
        self.capacities = capacities
        self.noise = noise
        self.deterministic = deterministic
        self.service_process = service_process

    def beta_gen(self, mean, rng):
        beta_mean = (mean+1)/2

        # find alpha such that alpha/(alpha+beta) = beta_mean
        # alpha = (alpha+beta)*beta_mean
        # (1-beta_mean)*alpha = beta*beta_mean
        # alpha = (beta/(1-beta_mean))*beta_mean

        alpha = (self.noise/(1-beta_mean))*beta_mean

        beta_val = rng.beta(alpha, self.noise)

        return (2*beta_val)-1

    def generate_customer_reward(self, state_idx, level, rng):
        #return self.customer_rewards[state_idx][level]+ rng.normal(0, self.noise)
        if self.deterministic:
            return self.customer_rewards[state_idx][level]

        return self.beta_gen(self.customer_rewards[state_idx][level], rng)

    def generate_server_reward(self, state_idx, level, rng):
        #return self.server_rewards[state_idx][level]+ rng.normal(0, self.noise)
        #return self.beta_gen(self.server_rewards[state_idx][level], rng)
        if self.deterministic:
            if self.service_process is not None:
                return (self.service_process(rng))(state_idx, level)
            return self.server_rewards[state_idx][level]
        return self.beta_gen(self.server_rewards[state_idx][level], rng)

    def generate_holding_reward(self, state_idx, rng):
        #return self.beta_gen(self.holding_rewards[state_idx], rng)
        if self.deterministic:
            return self.holding_rewards[state_idx]
        return self.beta_gen(self.holding_rewards[state_idx], rng)

    def print_rewards(self):
        for state in range(0, sum(self.capacities)+1):
            print(f"({state-self.capacities[1]}): holding_reward: {self.holding_rewards[state]}, customer_rewards: {self.customer_rewards[state]}, server_rewards: {self.server_rewards[state]}")

class ModelBounds:
    def __init__(self, capacities, n_levels, rate_lb, rate_ub, customer_ub=None, server_ub=None):
        self.capacities = capacities
        self.n_levels = n_levels

        self.rate_lb = rate_lb
        self.rate_ub = rate_ub
        self.n_states = sum(self.capacities)+1
        self.n_actions = n_levels[0]*n_levels[1]

        if not customer_ub:
            self.customer_ub = rate_ub
        else:
            self.customer_ub = customer_ub

        if not server_ub:
            self.server_ub = rate_ub
        else:
            self.server_ub = server_ub

class Model:
    def __init__(self, customer_levels, server_levels, rewards, capacities, rng : np.random._generator.Generator):
        self.customer_levels = customer_levels
        self.server_levels = server_levels
        self.capacities = capacities
        self.rewards = rewards

        self.n_states = sum(self.capacities)+1

        self.rng = rng

    def get_state_idx(self, state):
        return state + self.capacities[1] # server capacities are after customer ones

    def get_customer_rate(self, state, level):
        state_idx = self.get_state_idx(state)

        return self.customer_levels[state_idx][level]

    def get_server_rate(self, state, level):
        state_idx = self.get_state_idx(state)

        return self.server_levels[state_idx][level]
    
    def generate_transition_reward(self, state, action, transition):
        state_idx = self.get_state_idx(state)
        if transition == -1:
            return self.rewards.generate_server_reward(state_idx, action[1], self.rng)
        elif transition == 1:
            return self.rewards.generate_customer_reward(state_idx, action[0], self.rng)
        else:
            raise Exception("Invalid transition")
    
    def get_transition_reward(self, state, action, transition):
        state_idx = self.get_state_idx(state)
        if transition == -1:
            return self.rewards.server_rewards[state_idx][action[1]]
        elif transition == 1:
            return self.rewards.customer_rewards[state_idx][action[0]]
        else:
            raise Exception("Invalid transition")
    
    def generate_holding_reward(self, state):
        state_idx = self.get_state_idx(state)
        return self.rewards.generate_holding_reward(state_idx, self.rng)
    
    def get_holding_reward(self, state):
        state_idx = self.get_state_idx(state)
        return self.rewards.holding_rewards[state_idx]
    
    
    def get_state_reward(self, state_idx, action):
        cust_reward = self.customer_levels[state_idx][action[0]] * self.rewards.customer_rewards[state_idx][action[0]]
        serv_reward = self.server_levels[state_idx][action[1]] * self.rewards.server_rewards[state_idx][action[1]]

        state_reward = cust_reward + serv_reward + self.rewards.holding_rewards[state_idx]
        return state_reward

    def get_mean_rewards(self, policy_):
        # get the mean reward in each state (indexed)
        mean_rewards = []
        
        for state_idx in range(self.n_states):
            state_reward = self.get_state_reward(state_idx, policy_.get_action_idx(state_idx))
            
            mean_rewards.append(state_reward)

        return mean_rewards


    def get_distribution(self, policy_):
        # get the stationary distribution of the policy

        unnorm_probs = [1]

        for state_idx in range(1, self.n_states):
            action = policy_.get_action_idx(state_idx)
            cust_rate = self.customer_levels[state_idx-1][policy_.get_action_idx(state_idx-1)[0]]
            serv_rate = self.server_levels[state_idx][action[1]]

            if serv_rate == 0:
                if state_idx <= self.capacities[1]:
                    unnorm_probs = [0 for x in unnorm_probs]
                    unnorm_probs.append(1)
                    continue
                
                raise Exception("distribution is not unichain with recurrent 0")

            
            new_prob = (cust_rate/serv_rate) * unnorm_probs[-1]
            unnorm_probs.append(new_prob)

        norm = sum(unnorm_probs)
        return [x/norm for x in unnorm_probs]


    def get_unnorm_distribution(self, policy_):
        # get the stationary distribution of the policy

        unnorm_probs = [1]

        for state_idx in range(1, self.n_states):
            action = policy_.get_action_idx(state_idx)
            cust_rate = self.customer_levels[state_idx-1][policy_.get_action_idx(state_idx-1)[0]]
            serv_rate = self.server_levels[state_idx][action[1]]

            if serv_rate == 0:
                if state_idx <= self.capacities[1]:
                    unnorm_probs = [0 for x in unnorm_probs]
                    unnorm_probs.append(1)
                    continue
                
                raise Exception("distribution is not unichain with recurrent 0")

            
            new_prob = (cust_rate/serv_rate) * unnorm_probs[-1]
            unnorm_probs.append(new_prob)

        return unnorm_probs, sum(unnorm_probs)
    
    def evaluate_policy(self, policy_):
        # return the gain and bias of this policy
        mean_rewards = self.get_mean_rewards(policy_)
        #print(f"mean_rewards: {mean_rewards}")
        distribution, norm = self.get_unnorm_distribution(policy_)

        gain = sum([x * y for x,y in zip(mean_rewards, distribution)])
        gain /= norm

        bias_diff = []
        bias_acc = 0


        """
        # find the bias via forward induction
        # start with the base case, i.e. the lowest state. Lambda must be positive here (except in the trivial case)
        cust_rate = self.customer_levels[0][policy_.get_action_idx(0)[0]]
        bias_diff.append((gain - mean_rewards[0])/cust_rate)
        for state_idx in range(1, self.n_states-1):
            cust_rate = self.customer_levels[state_idx][policy_.get_action_idx(state_idx)[0]]
            serv_rate = self.server_levels[state_idx][policy_.get_action_idx(state_idx)[1]]

            if cust_rate == 0:
                # this will be added in the backward pass
                raise Exception("this should not happen")
                bias_diff.append(None)
                continue

            bd = (serv_rate/cust_rate)*bias_diff[-1] + (gain - mean_rewards[state_idx])/cust_rate
            bias_diff.append(bd)

        serv_rate = self.server_levels[-1][policy_.get_action_idx(self.n_states-1)[1]]
        bias_diff[-1] = (mean_rewards[-1] - gain)/serv_rate

        for state_idx in range(self.n_states-2, (self.n_states)//2, -1):
            # backward pass to fix numerical blowup from the first.
            cust_rate = self.customer_levels[state_idx][policy_.get_action_idx(state_idx)[0]]
            serv_rate = self.server_levels[state_idx][policy_.get_action_idx(state_idx)[1]]

            if serv_rate == 0:
                break

            bias_diff[state_idx-1] = (cust_rate/serv_rate)*bias_diff[state_idx] + (mean_rewards[state_idx]-gain)/serv_rate
        """

        # find the bias using the time-reversed equation
        midpoint = self.n_states//2
        bias_diff = [0 for i in range(self.n_states-1)]
        for state_idx in range(0, self.n_states-1):
            acc_diff = 0
            for att_state in range(state_idx+1, self.n_states):
                cust_rate = self.customer_levels[att_state-1][policy_.get_action_idx(att_state-1)[0]]
                serv_rate = self.server_levels[att_state][policy_.get_action_idx(att_state)[1]]

                if serv_rate == 0:
                    continue

                if cust_rate == 0:
                    break

                acc_diff += distribution[att_state]*(mean_rewards[att_state] - gain)
            cust_rate = self.customer_levels[state_idx][policy_.get_action_idx(state_idx)[0]]
            bias_diff[state_idx] = acc_diff/(cust_rate*distribution[state_idx])


        bias = [0]

        for bd in bias_diff:
            bias.append(bias[-1] + bd)

        # normalize 
        #bnorm = 0
        #for p, h in zip(distribution, bias):
        #    bnorm += p * h
        #bias = [h - bnorm for h in bias]

        #print(f"bias: {bias}")
        
        return gain, bias

    def improve_policy(self, policy_):
        gain, bias = self.evaluate_policy(policy_)

        new_mapping = []

        changed = False

        for state_idx in range(self.n_states):
            #max_bias = float("-inf")
            #argmax = 0
            #max_bias = bias[state_idx]
            argmax = policy_.get_action_idx(state_idx)

            cust_reward = self.rewards.customer_rewards[state_idx][argmax[0]]
            serv_reward = self.rewards.server_rewards[state_idx][argmax[1]]
            cust_rate = self.customer_levels[state_idx][argmax[0]]
            serv_rate = self.server_levels[state_idx][argmax[1]]
            bias_nom = 0
            total_rate = 0
            if cust_rate > 0:
                bias_nom += cust_rate * bias[state_idx+1]
                total_rate += cust_rate
            if serv_rate > 0:
                bias_nom += serv_rate * bias[state_idx-1]
                total_rate += serv_rate

            bias_nom += self.get_state_reward(state_idx, argmax)
            bias_nom -= gain

            max_bias = bias_nom/total_rate


            for cust_level, cust_rate in enumerate(self.customer_levels[state_idx]):
                cust_reward = self.rewards.customer_rewards[state_idx][cust_level]

                for serv_level, serv_rate in enumerate(self.server_levels[state_idx]):
                    serv_reward = self.rewards.server_rewards[state_idx][serv_level]
                    
                    bias_nom = 0
                    total_rate = 0
                    if cust_rate > 0:
                        bias_nom += cust_rate * bias[state_idx+1]
                        total_rate += cust_rate

                    if serv_rate > 0:
                        bias_nom += serv_rate * bias[state_idx-1]
                        total_rate += serv_rate

                    bias_nom += self.get_state_reward(state_idx, (cust_level, serv_level))
                    bias_nom -= gain

                    if (bias_nom/total_rate) > max_bias:
                        max_bias = bias_nom/total_rate
                        argmax = (cust_level, serv_level)
                #if state_idx >= self.n_states-2:
                #    print(f"({state_idx}) old mapping: {policy_.get_action_idx(state_idx)}, new mapping: {argmax}")
                #    print(f"({state_idx}) old bias: {bias[state_idx]}, new bias: {max_bias}")
            new_mapping.append(argmax)
            if argmax != policy_.get_action_idx(state_idx):
                changed = True

        return policy.Policy(new_mapping, self.capacities), gain, changed

    def evaluate_policy2(self, policy_):
        # Solve directly for the bias differences delta(s) = h(s+1) - h(s)
        # and the gain g. These are uniquely determined by the Bellman
        # equations -- no normalization / anchor row is needed, so no
        # implicit choice of "which bias is small" is baked into the system.
        # Bellman at state s:
        #   g = r(s) + cust_rate(s)*delta(s) - serv_rate(s)*delta(s-1)
        # gives n equations in n unknowns x = [delta(0), ..., delta(n-2), g].
        # Boundary states drop the term whose rate is zero (serv_rate(0) = 0
        # and cust_rate(n-1) = 0). Solved with np.linalg.solve (LU with
        # partial pivoting) -- no truncation, no min-norm shrinkage.
        n = self.n_states
        A = np.zeros((n, n))
        b = np.zeros(n)
        g_col = n - 1

        # State 0: -cust_rate * delta(0) + g = r(0)
        action = policy_.get_action_idx(0)
        cust_rate = self.customer_levels[0][action[0]]
        A[0, 0] = -cust_rate
        A[0, g_col] = 1.0
        b[0] = self.get_state_reward(0, action)

        # Interior: serv_rate * delta(s-1) - cust_rate * delta(s) + g = r(s)
        for state_idx in range(1, n - 1):
            action = policy_.get_action_idx(state_idx)
            cust_rate = self.customer_levels[state_idx][action[0]]
            serv_rate = self.server_levels[state_idx][action[1]]
            A[state_idx, state_idx - 1] = serv_rate
            A[state_idx, state_idx] = -cust_rate
            A[state_idx, g_col] = 1.0
            b[state_idx] = self.get_state_reward(state_idx, action)

        # State n-1: serv_rate * delta(n-2) + g = r(n-1)
        action = policy_.get_action_idx(n - 1)
        serv_rate = self.server_levels[n - 1][action[1]]
        A[n - 1, n - 2] = serv_rate
        A[n - 1, g_col] = 1.0
        b[n - 1] = self.get_state_reward(n - 1, action)

        x = np.linalg.solve(A, b)

        deltas = [float(v) for v in x[:n - 1]]
        gain = float(x[g_col])

        # Reconstruct h from the bias differences. The integration constant
        # (h(0) = 0) is arbitrary -- improve_policy2 compares actions within
        # a single state, and any constant shift in h drops out of that
        # comparison, so this anchor has no algorithmic effect.
        bias = [0.0]
        for d in deltas:
            bias.append(bias[-1] + d)

        return gain, bias

    def improve_policy2(self, policy_):
        gain, bias = self.evaluate_policy2(policy_)

        new_mapping = []
        changed = False

        for state_idx in range(self.n_states):
            argmax = policy_.get_action_idx(state_idx)

            cust_rate = self.customer_levels[state_idx][argmax[0]]
            serv_rate = self.server_levels[state_idx][argmax[1]]
            bias_nom = 0
            total_rate = 0
            if cust_rate > 0:
                bias_nom += cust_rate * bias[state_idx + 1]
                total_rate += cust_rate
            if serv_rate > 0:
                bias_nom += serv_rate * bias[state_idx - 1]
                total_rate += serv_rate

            bias_nom += self.get_state_reward(state_idx, argmax)
            bias_nom -= gain

            max_bias = bias_nom / total_rate if total_rate > 0 else float("-inf")

            for cust_level, cust_rate in enumerate(self.customer_levels[state_idx]):
                for serv_level, serv_rate in enumerate(self.server_levels[state_idx]):
                    bias_nom = 0
                    total_rate = 0
                    if cust_rate > 0:
                        bias_nom += cust_rate * bias[state_idx + 1]
                        total_rate += cust_rate
                    if serv_rate > 0:
                        bias_nom += serv_rate * bias[state_idx - 1]
                        total_rate += serv_rate

                    bias_nom += self.get_state_reward(state_idx, (cust_level, serv_level))
                    bias_nom -= gain

                    if total_rate > 0 and (bias_nom / total_rate) > max_bias + 1e-2:
                        max_bias = bias_nom / total_rate
                        argmax = (cust_level, serv_level)

            new_mapping.append(argmax)
            if argmax != policy_.get_action_idx(state_idx):
                changed = True

        return policy.Policy(new_mapping, self.capacities), gain, changed

    def relative_value_iteration(self, original_policy=None, tolerance=0.01):
        default_mapping = [(0,0) for i in range(self.n_states)]
        if not original_policy:
            new_policy = policy.Policy(default_mapping, self.capacities)
        else:
            new_policy = original_policy

        values = [0 for i in range(self.n_states)]

        while True:
            new_mapping = [(0,0) for i in range(self.n_states)]

            min_delta = float("inf")
            max_delta = float("-inf")

            for state_idx in range(self.n_states):
                max_val = float("-inf")
                argmax = (0,0)
                for cust_level, cust_rate in enumerate(self.customer_levels[state_idx]):
                    for serv_level, serv_rate in enumerate(self.server_levels[state_idx]):
                        val = 0

                        if cust_rate > 0:
                            val += (cust_rate/(cust_rate+serv_rate))*values[state_idx+1]
                        if serv_rate > 0:
                            val += (serv_rate/(cust_rate+serv_rate))*values[state_idx-1]

                        val += self.get_state_reward(state_idx, (cust_level, serv_level))

                        if val > max_val:
                            max_val = val
                            argmax = (cust_level, serv_level)
                new_values[state_idx] = max_val
                new_mapping[state_idx] = argmax

                min_delta = min(min_delta, new_values[state_idx]-values[state_idx])
                max_delta = max(max_delta, new_values[state_idx]-values[state_idx])

            # update
            new_policy = policy.Policy(new_mapping, self.capacities)

            # check for convergence
            if max_delta - min_delta < tolerance:
                break

        return new_policy

    def plot_distribution(self, policy_, title=None, savepath=None, dpi=200):
        dist_raw = self.get_distribution(policy_)
        rew_raw  = self.get_mean_rewards(policy_)

        dist = [float(v) for v in dist_raw]
        rewards = [float(v) for v in rew_raw]

        if len(dist) != len(rewards):
            raise ValueError(f"dist and rewards must have same length, got {len(dist)} vs {len(rewards)}")
        S = len(dist)
        if S == 0:
            raise ValueError("Empty dist/rewards")

        # --- capacity-scaled x locations ---
        lo = -float(self.capacities[1])
        hi =  float(self.capacities[0])
        if S == 1:
            x = [0.5 * (lo + hi)]
            dx = 1.0
        else:
            dx = (hi - lo) / (S - 1)
            x = [lo + i * dx for i in range(S)]

        # --- nice ticks (keep endpoints visible) ---
        tick_count = (S//5)+1
        tick_idx = [i*5 for i in range(tick_count)] if tick_count > 1 else [0]
        tick_vals = [x[i] for i in tick_idx]

        def fmt(v):
            if abs(v) < 5e-10:
                v = 0.0
            iv = int(round(v))
            return str(iv) if abs(v - iv) < 1e-10 else f"{v:g}"

        tick_labels = [fmt(v) for v in tick_vals]

        # --- figure ---
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(6.4, 4.2), sharex=True,
            gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.08}
        )

        for ax in (ax0, ax1):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="y", linewidth=0.6, alpha=0.25)
            ax.set_axisbelow(True)

        # --- top: distribution ---
        bar_w = 0.85 * dx if S > 1 else 0.8
        ax0.bar(x, dist, width=bar_w, linewidth=0.0, alpha=0.9)
        ax0.set_ylabel("State probability")

        s = sum(dist)
        if s > 0 and abs(s - 1.0) > 1e-3:
            ax0.text(0.99, 0.92, f"sum={s:.2f}", transform=ax0.transAxes,
                     ha="right", va="top", fontsize=9, alpha=0.7)

        # (optional) highlight probability concentration: top-3 states
        #k = min(3, S)
        #top_idx = sorted(range(S), key=lambda i: dist[i])[-k:]
        #ax0.scatter([x[i] for i in top_idx], [dist[i] for i in top_idx], s=25, zorder=3)

        # --- bottom: rewards ---
        ax1.plot(x, rewards, linewidth=2.0)
        ax1.axhline(0.0, linewidth=1.0, alpha=0.4)
        ax1.set_ylabel("Mean reward")
        ax1.set_xlabel("State")

        # x ticks & limits (ensure endpoints readable)
        ax1.set_xticks(tick_vals)
        ax1.set_xticklabels(tick_labels)
        if S > 1:
            ax1.set_xlim(x[0] - 0.5 * dx, x[-1] + 0.5 * dx)

        ax1.set_ylim([0, 11])

        if title:
            fig.suptitle(title, y=0.98)

        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=dpi)

        return fig, (ax0, ax1)

    def get_optimal_policy(self, original_policy=None, n_iterations=500):
        default_mapping = [(0,0) for i in range(self.n_states)]
        if not original_policy:
            new_policy = policy.Policy(default_mapping, self.capacities)
        else:
            new_policy = original_policy

        for i in range(n_iterations):
            new_policy, gain, changed = self.improve_policy2(new_policy)
            if not changed:
                break
        gain, bias = self.evaluate_policy(new_policy)
        #input("continue?")
        return new_policy, gain

    def print_rates(self):
        for state in range(0, self.n_states):
            print(f"({state-self.capacities[1]}): server_rates: {self.server_levels[state]}, customer_rates: {self.customer_levels[state]}")

    def print_rewards(self):
        self.rewards.print_rewards()

def generate_random_model(model_bounds, rng : np.random._generator.Generator):
    capacities = model_bounds.capacities
    n_levels = model_bounds.n_levels
    rate_lb = model_bounds.rate_lb
    rate_ub = model_bounds.rate_ub

    n_states = sum(capacities)+1
    #holding_rewards = list(rng.uniform(-1,1,n_states))
    holding_rewards = []
    for state_idx in range(n_states):
        state = state_idx-capacities[1]
        if state >= 0:
            holding_rewards.append(-state/(capacities[0]+1))
        else:
            holding_rewards.append(state/(capacities[1]+1))
        #holding_rewards.append(0.04 * abs(state))
    customer_rewards = [list(rng.uniform(-1,1,n_levels[0])) for i in range(n_states)]
    server_rewards = [list(rng.uniform(-1,1,n_levels[1])) for i in range(n_states)]
    rewards = ModelRewards(holding_rewards, customer_rewards, server_rewards, capacities)

    customer_levels = []
    server_levels = []

    rate_midpoint = (rate_ub+rate_lb)/2

    upper_c = sorted(list(rng.uniform(rate_midpoint, rate_ub, n_states)), reverse=True)
    upper_s = sorted(list(rng.uniform(rate_midpoint, rate_ub, n_states)))

    lower_c = sorted(list(rng.uniform(rate_lb, rate_midpoint, n_states)), reverse=True)
    lower_s = sorted(list(rng.uniform(rate_lb, rate_midpoint, n_states)))

    for state in range(n_states):
        other_c = list(rng.uniform(lower_c[state], upper_c[state], n_levels[0]-2))
        other_s = list(rng.uniform(lower_s[state], upper_s[state], n_levels[1]-2))

        customer_levels.append(other_c + [lower_c[state], upper_c[state]])
        server_levels.append(other_s + [lower_s[state], upper_s[state]])

    customer_levels[-1] = [0 for x in customer_levels[-1]]
    server_levels[0] = [0 for x in server_levels[0]]

    model = Model(customer_levels, server_levels, rewards, capacities, rng)
    return model

def generate_random_model_2(model_bounds, rng : np.random._generator.Generator):
    capacities = model_bounds.capacities
    n_levels = model_bounds.n_levels
    rate_lb = model_bounds.rate_lb
    rate_ub = model_bounds.rate_ub

    n_states = sum(capacities)+1
    #holding_rewards = list(rng.uniform(-1,1,n_states))
    holding_rewards = []
    for state_idx in range(n_states):
        state = state_idx-capacities[1]
        if state >= 0:
            holding_rewards.append(-state/(capacities[0]+1))
        else:
            holding_rewards.append(state/(capacities[1]+1))
        #holding_rewards.append(0.04 * abs(state))

    customer_rewards = [list(rng.uniform(-1,1,n_levels[0])) for i in range(n_states)]
    server_rewards = [list(rng.uniform(-1,1,n_levels[1])) for i in range(n_states)]
    rewards = ModelRewards(holding_rewards, customer_rewards, server_rewards, capacities)

    customer_levels = []
    server_levels = []

    rate_midpoint = (rate_ub+rate_lb)/2
    rate_midmidpoint = (rate_ub-rate_lb)*(1/4) + rate_lb

    upper_c = sorted(list(rng.uniform(rate_midmidpoint, rate_ub, n_states)), reverse=True)
    upper_s = sorted(list(rng.uniform(rate_midmidpoint, rate_ub, n_states)))

    lower_c = sorted(list(rng.uniform(rate_lb, rate_midpoint, n_states)), reverse=True)
    lower_s = sorted(list(rng.uniform(rate_lb, rate_midpoint, n_states)))


    for state in range(n_states):
        c_lo = min(lower_c[state], upper_c[state])
        c_hi = max(lower_c[state], upper_c[state])
        s_lo = min(lower_s[state], upper_s[state])
        s_hi = max(lower_s[state], upper_s[state])
        other_c = list(rng.uniform(c_lo, c_hi, n_levels[0]-2))
        other_s = list(rng.uniform(s_lo, s_hi, n_levels[1]-2))

        customer_levels.append([lower_c[state], upper_c[state]] + other_c)
        server_levels.append([lower_s[state], upper_s[state]] + other_s)

    customer_levels[-1] = [0 for x in customer_levels[-1]]
    server_levels[0] = [0 for x in server_levels[0]]

    model = Model(customer_levels, server_levels, rewards, capacities, rng)
    return model

def generate_server_model(model_bounds, rng): 
    capacities = model_bounds.capacities

    n_states = sum(capacities)+1

    abandonment_rate = 2.5
    max_server_rate = 2.5
    arrival_rate = 2.5

    def generate_sensitivities(rng):
        # first difference
        fd = [rng.uniform(0,1) for i in range(n_states)]
        unnorm = [0]
        for x in fd:
            unnorm.append(unnorm[-1] + x)
        unnorm = unnorm[:n_states]
        return [(x/unnorm[-1]) for x in unnorm]


    sensitivities = generate_sensitivities(rng)
    abandonments = [(i/n_states)*y*abandonment_rate for i, y in enumerate(sensitivities)]

    clipgauss = lambda m, s, l, u: max(min(rng.normal(loc=m, scale=s), u),l)

    service_cost = sorted([clipgauss(-0.25, 0.25, -1, 0) for i in range(3)])
    server_rates = sorted([clipgauss(2, 0.5, 1.5, max_server_rate) for i in range(3)], reverse=True)

    
    #server_rewards = [[get_service_cost(i)] for i in range(n_states)]
    server_levels = []
    server_rewards = []
    p_abandon = []
    
    for s in range(n_states):
        server_levels.append([abandonments[s] + x for x in server_rates])
        rew = []
        prob = []

        for i, x in enumerate(server_rates):
            p = abandonments[s]/(abandonments[s] + x)
            prob.append(p)
            rew.append(-p + (1-p) * service_cost[i])

        p_abandon.append(prob)
        server_rewards.append(rew)

    server_levels[0] = [0 for x in server_levels[0]]

    generate_sc = lambda rng: (lambda state, level: -1 if rng.uniform() < p_abandon[state][level] else service_cost[level])
        

    holding_rewards = [-1.0*(i/(capacities[0]+1)) for i in range(n_states)]
    customer_rewards = [[1] for i in range(n_states)]
    rewards = ModelRewards(holding_rewards, customer_rewards, server_rewards, capacities, deterministic=True, service_process=generate_sc)
    

    customer_levels = []

    for state in range(n_states):
        rate = arrival_rate - sensitivities[state]

        customer_levels.append([rate])

    customer_levels[-1] = [0 for x in customer_levels[-1]]

    model = Model(customer_levels, server_levels, rewards, capacities, rng)

    return model

def generate_pricing_model(model_bounds, rng): 
    capacities = model_bounds.capacities

    n_states = sum(capacities)+1

    prices = [i*(1/3)*0.5 for i in range(0,3)]

    n_servers = n_states//10

    service_rate = 3/n_servers
    abandonment_rate = 1

    def generate_sensitivities(rng):
        # first difference
        fd = [rng.uniform(0,1) for i in range(n_states)]
        unnorm = [0]
        for x in fd:
            unnorm.append(unnorm[-1] + x)
        unnorm = unnorm[:n_states]
        return [(x/unnorm[-1]) for x in unnorm]


    sensitivities = generate_sensitivities(rng)
    abandonments = [(i/n_states)*y*abandonment_rate for i, y in enumerate(sensitivities)]

    get_service_rate = lambda i: min(i*service_rate, n_servers*service_rate)
    get_abandonment_rate = lambda i: abandonments[i]
    sc = lambda i, x, y: (0.5*x(i)-y(i))/(x(i) + y(i)) if i != 0 else 0
    abandon_bind = lambda i, x, y: y(i)/(x(i) + y(i)) if i != 0 else 0
    p_abandon = lambda i: abandon_bind(i, get_service_rate, get_abandonment_rate)
    get_service_cost = lambda i: sc(i, get_service_rate, get_abandonment_rate)

    generate_sc = lambda rng: (lambda i, _: -1 if rng.uniform() < p_abandon(i) else 0.5)
    
    server_rewards = [[get_service_cost(i)] for i in range(n_states)]
    holding_rewards = [-1.0*(i/(capacities[0]+1)) for i in range(n_states)]
    customer_rewards = [list(prices) for i in range(n_states)]
    rewards = ModelRewards(holding_rewards, customer_rewards, server_rewards, capacities, deterministic=True, service_process=generate_sc)
    
    est_rewards = []
    for i in range(n_states):
        csum = 0
        for j in range(1000):
            csum += generate_sc(rng)(i, 0)
        est_rewards.append(csum/1000)

    arrival_rate = 2 
    server_levels = [[get_service_rate(i) + get_abandonment_rate(i)] for i in range(n_states)]
    server_levels[0] = [0]

    customer_levels = []

    price_elasticity = rng.uniform(0.8,1.2)

    default_p = 1
    default_w = 1/service_rate

    for state in range(n_states):
        #est_waiting_time = sum([1/x[0] for x in server_levels[1:state+1]])
        #est_waiting_time = max((state+1)*service_rate
        service_time = 1/service_rate
        queue_waiting_time = (1/(n_servers*service_rate)) * max((state+1)-n_servers, 0)
        est_waiting_time = service_time + queue_waiting_time

        rates = []

        for p in prices:
            rate = arrival_rate
            rate *= 1-(price_elasticity * ((p-default_p)/(default_p+0.5)))
            rate -= 0.5*sensitivities[state]

            rates.append(rate)

        customer_levels.append(rates)

    customer_levels[-1] = [0 for x in customer_levels[-1]]

    model = Model(customer_levels, server_levels, rewards, capacities, rng)

    return model


def generate_path_model(model_bounds, rng : np.random._generator.Generator):
    capacities = model_bounds.capacities

    n_states = sum(capacities)+1

    customer_levels = [[2,1] for i in range(n_states)]
    customer_levels[-1] = [0,0]

    n_servers = n_states//5
    service_rate = 3/n_servers

    server_levels = [[min(i*service_rate, n_servers*service_rate)] for i in range(n_states)]
    server_levels[0] = [0]

    customer_rewards = [[0,1] for i in range(n_states)]
    server_rewards = [[0] for i in range(n_states)]
    holding_rewards = [0 for i in range(n_states)]
    rewards = ModelRewards(holding_rewards, customer_rewards, server_rewards, capacities, deterministic=True)

    model = Model(customer_levels, server_levels, rewards, capacities, rng)

    return model

def generate_special_model(model_bounds, rng : np.random._generator.Generator):
    capacities = model_bounds.capacities

    n_states = sum(capacities)+1

    customer_levels = [[1.0, 2.0, 3.0] for i in range(n_states)]
    customer_levels[-1] = [0 for i in range(3)]

    server_levels = [[1.0, 2.0, 3.0] for i in range(n_states)]
    server_levels[0] = [0 for i in range(3)]

    customer_rewards = [[0.66, 0.33, 0.0] for i in range(n_states)]
    server_rewards = [[0.0, -0.33, -0.66] for i in range(n_states)]
    holding_rewards = [abs(i-10)*0.1 for i in range(n_states)]
    rewards = ModelRewards(holding_rewards, customer_rewards, server_rewards, capacities)

    model = Model(customer_levels, server_levels, rewards, capacities, rng)

    return model
