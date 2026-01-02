class Exploration:
    def __init__(self, model_bounds):
        self.model_bounds = model_bounds

        self.n_actions = self.model_bounds.n_levels[0] * self.model_bounds.n_levels[1]
        self.n_states = sum(self.model_bounds.capacities)+1

        self.sa_visit_counts = [[0 for j in range(self.n_actions)] for i in range(self.n_states)]
        self.sa_visit_counts_in_episode = [[0 for j in range(self.n_actions)] for i in range(self.n_states)]

        self.steps_before_episode = 0
        self.n_episodes = 0

    def observe(self, state, action):
        state_idx = state + self.model_bounds.capacities[1]

        action_idx = (action[0]*self.model_bounds.n_levels[1]) + action[1]

        self.sa_visit_counts[state_idx][action_idx] += 1
        self.sa_visit_counts_in_episode[state_idx][action_idx] += 1

        return (2*self.sa_visit_counts_in_episode[state_idx][action_idx]) >= self.sa_visit_counts[state_idx][action_idx]

    def new_episode(self):
        self.sa_visit_counts_in_episode = [[0 for j in range(self.n_actions)] for i in range(self.n_states)]

        self.steps_before_episode = sum([sum(x) for x in self.sa_visit_counts])
        self.n_episodes += 1
