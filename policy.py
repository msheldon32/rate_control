class Policy:
    def __init__(self, policy_mapping, capacities):
        self.policy_mapping = policy_mapping
        self.capacities = capacities

    def get_state_idx(self, state):
        return state + self.capacities[1]

    def get_action_idx(self, state_idx):
        return self.policy_mapping[state_idx]

    def get_action(self, state):
        state_idx = self.get_state_idx(state)
        return self.get_action_idx(state_idx)
