import model
import agent
import observer
import simulator

from learners.UCRL2 import UCRL2
from learners.UCRL3 import UCRL3
from learners.KL_UCRL import KL_UCRL

import pickle
import numpy as np
import sys


def create_pricing_model(max_service_rate, n_states, rng):
    capacities = (n_states - 1, 0)
    n_servers = 1#max(1, n_states // 10)
    service_rate = max_service_rate/n_servers

    prices = [0, 0.5, 1.0]
    p_min, p_max = prices[0], prices[-1]

    # FIFO sojourn time in an M/M/c with c=n_servers, rate=service_rate.
    # State i = customers in system; the arrival becomes the (i+1)th customer.
    #   i <  n_servers: a server is free, sojourn = 1/μ
    #   i >= n_servers: queue wait for (i - n_servers + 1) completions, then service
    def expected_waiting_time(i):
        if i < n_servers:
            return 1.0 / service_rate
        return (i - n_servers + 1) / (n_servers * service_rate) + 1.0 / service_rate

    W = [expected_waiting_time(i) for i in range(n_states)]

    # State 0 spans [1.5, 3] across price levels via f(p) = 1 - u^2,
    # u = (p - p_min)/(p_max - p_min). Lowest rate (highest price) is 2.
    def demand_factor(p):
        u = (p - p_min) / (p_max - p_min)
        return (1.0 - u * u)*1.5

    state0_rates = [1.5 + demand_factor(p) for p in prices]

    # delta(s) grows linearly in W from 0 (state 0) to 1 (last state).
    # Subtracting it uniformly decays the lowest rate from 2 down to 1 and
    # shifts every other level by the same amount, keeping all rates in [1, 3].
    if W[-1] > W[0]:
        delta_slope = (1/3) / (W[-1] - W[0])
    else:
        delta_slope = 0.0
    delta = [delta_slope * (W[i] - W[0]) for i in range(n_states)]

    customer_levels = []
    for i in range(n_states):
        rates = [r * (1-delta[i]) for r in state0_rates]
        customer_levels.append(rates)
    customer_levels[-1] = [0.0 for _ in prices]

    server_levels = [[min(i, n_servers) * service_rate] for i in range(n_states)]
    server_levels[0] = [0.0]

    price_scale = 10.0 / n_states
    customer_rewards = [[p * price_scale for p in prices] for _ in range(n_states)]
    server_rewards = [[price_scale] for _ in range(n_states)]
    holding_rewards = [-1.0 * (i / (capacities[0] + 1)) for i in range(n_states)]

    rewards = model.ModelRewards(holding_rewards, customer_rewards, server_rewards, capacities, deterministic=True)

    model_ = model.Model(customer_levels, server_levels, rewards, capacities, rng)

    model_.print_rates()
    model_.print_rewards()
    input()

    return model_


class BaselineRun:
    def __init__(self, model_, model_bounds, rng, max_step_count, uni_constant=0.05, rand_rew=True):
        self.model = model_
        self.model_bounds = model_bounds
        self.rng = rng
        self.max_step_count = max_step_count

        self.optimal_policy, self.max_gain = self.model.get_optimal_policy()

        self.baseline_learners = {
            #"KL": KL_UCRL(self.model_bounds.n_states, self.model_bounds.n_actions, 0.05),
            "UCRL2": UCRL2(self.model_bounds.n_states, self.model_bounds.n_actions, 0.05),
            "UCRL3": UCRL3(self.model_bounds.n_states, self.model_bounds.n_actions, 0.05),
        }

        self.agents = {
            baseline: agent.LearnersAgent(model_bounds.capacities, model_bounds.n_levels[0], model_bounds.n_levels[1], uni_constant, learner, model_bounds, rng)
                for baseline, learner in self.baseline_learners.items()
        }

        self.observers = {
            baseline: observer.Observer() for baseline in self.baseline_learners.keys()
        }

        self.simulators = {
            baseline: simulator.Simulator(model_, ag, self.observers[baseline], self.rng, rand_rew=rand_rew)
                for baseline, ag in self.agents.items()
        }

        _, self.ideal_gain = self.model.get_optimal_policy(n_iterations=100000)

    def run(self, verbose=False):
        for i in range(self.max_step_count):
            for sim in self.simulators.values():
                sim.step()

            if verbose and i > 0 and i % 10000 == 0:
                print(f"After {i} steps")
                for k, v in self.observers.items():
                    print(f"Trailing gain ({k}): ", v.trailing_gain(10000))
                    print(f"Total regret ({k}): ", v.empirical_regret(self.ideal_gain))
                print(f"Ideal gain: ", self.ideal_gain)

    def summarize(self, timestep=10000):
        out_dict = {"ideal_gain": self.ideal_gain}
        for k, v in self.observers.items():
            out_dict[k] = v.summarize(self.ideal_gain)
        return out_dict


class RCRun:
    def __init__(self, model_, model_bounds, rng, max_step_count, rand_rew=True):
        self.model = model_
        self.model_bounds = model_bounds
        self.rng = rng
        self.max_step_count = max_step_count

        self.optimal_policy, self.max_gain = self.model.get_optimal_policy()

        self.agent = agent.RC_Agent(model_bounds.capacities, model_bounds.n_levels[0], model_bounds.n_levels[1], model_bounds, rng, False)
        self.ablation_agent = agent.RC_Agent(model_bounds.capacities, model_bounds.n_levels[0], model_bounds.n_levels[1], model_bounds, rng, True)

        self.agent_observer = observer.Observer()
        self.ablation_observer = observer.Observer()

        self.agent_sim = simulator.Simulator(model_, self.agent, self.agent_observer, self.rng, rand_rew=rand_rew)
        self.ablation_sim = simulator.Simulator(model_, self.ablation_agent, self.ablation_observer, self.rng, rand_rew=rand_rew)

        self.ideal_policy, self.ideal_gain = self.model.get_optimal_policy(n_iterations=1000000)

    def run(self, verbose=False):
        for i in range(self.max_step_count):
            self.agent_sim.step()
            self.ablation_sim.step()

            if verbose and i > 0 and i % 10000 == 0:
                print(f"After {i} steps")
                print(f"Trailing gain (rc): ", self.agent_observer.trailing_gain(10000))
                print(f"Trailing regret (rc): ", self.agent_observer.empirical_regret(self.ideal_gain))
                print(f"Trailing gain (ablation): ", self.ablation_observer.trailing_gain(10000))
                print(f"Trailing regret (ablation): ", self.ablation_observer.empirical_regret(self.ideal_gain))
                print(f"Ideal gain: ", self.ideal_gain)
                #self.agent.model.print_rates()
                print(f"Optimal policy: ", self.ideal_policy.policy_mapping)

    def summarize(self, timestep=10000):
        return {
            "rc": self.agent_observer.summarize(self.ideal_gain, timestep),
            "ablation": self.ablation_observer.summarize(self.ideal_gain, timestep),
            "ideal_gain": self.ideal_gain,
        }


def seed_for(service_rate, n_states):
    # deterministic seed so multiple invocations line up across modes
    return int(round(service_rate * 1000)) * 1000000 + n_states * 1000


def run_experiment(service_rate, n_states, mode, max_step_count=50000000, starting_no=0, ending_no=10):
    starting_seed = seed_for(service_rate, n_states)
    model_bounds = model.ModelBounds((n_states - 1, 0), (3, 1), 1, 5)

    rngs = []
    models = []
    for run_no in range(starting_no, ending_no):
        rng = np.random.default_rng(seed=(starting_seed + run_no))
        rngs.append(rng)
        models.append(create_pricing_model(service_rate, n_states, rng))

    for run_no in range(starting_no, ending_no):
        model_ = models[run_no - starting_no]
        rng = rngs[run_no - starting_no]

        if mode == "baselines":
            run = BaselineRun(model_, model_bounds, rng, max_step_count)
            out_path = f"exp_out/{n_states}_states_{service_rate}_pricing/baselines_{run_no}"
        else:
            run = RCRun(model_, model_bounds, rng, max_step_count)
            out_path = f"exp_out/{n_states}_states_{service_rate}_pricing/run_{run_no}"

        run.run(verbose=True)
        with open(out_path, "wb") as f:
            pickle.dump(run.summarize(), f)


if __name__ == "__main__":
    service_rate = float(sys.argv[1])
    n_states = int(sys.argv[2])
    mode = sys.argv[3]  # "baselines" or "rc"
    run_experiment(service_rate, n_states, mode)
