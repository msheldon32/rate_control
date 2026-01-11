import pickle

class ExperimentRun:
    def __init__(self, model_, model_bounds, rng, max_step_count):
        self.model = model_
        self.model_bounds = model_bounds
        self.rng = rng
        self.max_step_count = max_step_count

        self.optimal_policy, self.max_gain = self.model.get_optimal_policy()

        self.agent = agent.RC_Agent(model_bounds.capacities, model_bounds.n_levels[0], model_bounds.n_levels[1], model_bounds, rng, False)
        self.ablation_agent = agent.RC_Agent(model_bounds.capacities, model_bounds.n_levels[0], model_bounds.n_levels[1], model_bounds, rng, True)

        self.agent_observer = observer.Observer()
        self.ablation_observer = observer.observer()

        self.agent_sim = simulator.Simulator(model, self.agent, self.agent_observer, self.rng)
        self.ablation_sim = simulator.Simulator(model, self.ablation_agent, self.ablation_observer, self.rng)


        _, self.ideal_gain = self.model.get_optimal_policy(n_iterations=10000)

    def run(self, verbose=False):
        for i in range(self.max_step_count):
            self.agent_sim.step()
            self.ablation_sim.step()

            if verbose and i > 0 and i % 10000 == 0:
                print(f"After {i} steps")
                print(f"Trailing gain (rc): ", self.agent_observer.get_past_n_gain(10000))
                print(f"Trailing gain (ablation): ", self.ablation_observer.get_past_n_gain(10000))

    def summarize(self, timestep=10000):
        return {
            "rc": self.agent_observer.summarize(self.ideal_gain, timestep),
            "ablation": self.ablation_observer.summarize(self.ideal_gain, timestep)
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
            rng = np.random.default_rng(self.starting_seed + run_no)
            model_ = model.generate_random_model(self.model_bounds, rng)

            run = ExperimentRun(model_, self.model_bounds, rng, self.max_step_count)
            #def __init__(self, model_, model_bounds, rng, max_step_count):
            try:
                run.run(verbose=True)
            except Exception as e:
                print(f"Run {run_no} failed, skipping...")
                continue
            with open(f"exp_out/{self.model_bounds.n_states}_states/run_{run_no}", "wb") as f:
                pickle.dump(run.summarize(), f)

if __name__ == "__main__":
    input("Generate random rewards first.")
    input("Fix holding rewards.")
