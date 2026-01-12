import model
import agent
import observer
import simulator

from learners.UCRL2 import UCRL2
from learners.UCRL3 import UCRL3
from learners.KL_UCRL import KL_UCRL

import pickle
import numpy as np
import traceback
import sys

class ExperimentRun:
    def __init__(self, model_, model_bounds, rng, max_step_count):
        self.model = model_
        self.model_bounds = model_bounds
        self.rng = rng
        self.max_step_count = max_step_count

        self.optimal_policy, self.max_gain = self.model.get_optimal_policy()

        self.baseline_learners = {
            "KL": KL_UCRL(self.model_bounds.n_states, self.model_bounds.n_actions, 0.05),
            "UCRL2": UCRL2(self.model_bounds.n_states, self.model_bounds.n_actions, 0.05),
            "UCRL3": UCRL3(self.model_bounds.n_states, self.model_bounds.n_actions, 0.05),
                }
        self.ablation_agent = agent.RC_Agent(model_bounds.capacities, model_bounds.n_levels[0], model_bounds.n_levels[1], model_bounds, rng, True)

        self.agents = {
                baseline: agent.LearnersAgent(model_bounds.capacities, model_bounds.n_levels[0], model_bounds.n_levels[1], 0.01, learner, model_bounds, rng) 
                    for baseline, learner in self.baseline_learners.items()
                }
        
        self.observers = {
                baseline: observer.Observer() for baseline in self.baseline_learners.keys()
                }
        
        self.simulators = {
                baseline: simulator.Simulator(model_, agent, self.observers[baseline], self.rng)
                    for baseline, agent in self.agents.items()
                }

        _, self.ideal_gain = self.model.get_optimal_policy(n_iterations=10000)

    def run(self, verbose=False):
        for i in range(self.max_step_count):
            for sim in self.simulators.values():
                sim.step()

            if i % 1000 == 0:
                print(f"{i} steps")

            if verbose and i > 0 and i % 10000 == 0:
                print(f"After {i} steps")
                for k, v in self.observers.items():
                    print(f"Trailing gain ({k}): ", v.trailing_gain(10000))
                print(f"Ideal gain: ", self.ideal_gain)

    def summarize(self, timestep=10000):
        return {
            "ideal_gain": self.ideal_gain
                } + {
            k: v.summarize() for k, v in self.observers.items()
            }


class Experiment:
    def __init__(self, model_bounds, max_step_count, starting_seed=0, starting_no=0, ending_no=50):
        self.model_bounds = model_bounds
        self.starting_seed = 0
        self.max_step_count = max_step_count
        self.starting_no = starting_no
        self.ending_no = ending_no
        self.starting_seed = starting_seed

    def run(self):
        for run_no in range(self.starting_no, self.ending_no):
            rng = np.random.default_rng(seed=(self.starting_seed + run_no))
            model_ = model.generate_random_model(self.model_bounds, rng)

            run = ExperimentRun(model_, self.model_bounds, rng, self.max_step_count)
            #def __init__(self, model_, model_bounds, rng, max_step_count):
            run.run(verbose=True)
            """try:
                run.run(verbose=True)
            except Exception as e:
                print(f"Run {run_no} failed, skipping...")
                traceback.print_exc()
                continue"""
            with open(f"exp_out/{self.model_bounds.n_states}_states/no_ucrl3_baselines_{run_no}", "wb") as f:
                pickle.dump(run.summarize(), f)

if __name__ == "__main__":
    # seed 1000: (3,3), (5,5)
    # seed 2000: (3,3), (10,10)
    # seed 3000: (3,3), (25,25)
    bounds = {
        1: ((5,5),1000),
        2: ((10,10),2000),
        3: ((25,25),3000),
            }
    
    capacities, seed = bounds[int(sys.argv[1])]
    model_bounds = model.ModelBounds(capacities,(3,3), 1, 5)
    exp = Experiment(model_bounds, 10000000, starting_seed = seed, starting_no=0, ending_no=50)

    exp.run()
