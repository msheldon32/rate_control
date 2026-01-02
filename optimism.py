import copy

import model

def build_optimistic_model(parameter_estimator, model_bounds, confidence_param, rng):
    n_opt_levels = (model_bounds.n_levels[0]*2, model_bounds.n_levels[1]*2)
    optimistic_bounds = model.ModelBounds(model_bounds.capacities, n_opt_levels, model_bounds.rate_lb, model_bounds.rate_ub)

    naive_customer_rates = []
    naive_server_rates = []
    for state_idx in range(model_bounds.n_states):
        naive_customer_rates.append([])
        naive_server_rates.append([])
        for customer_level in range(model_bounds.n_levels[0]):
            bounds = parameter_estimator.transition_rate_bounds(state_idx, customer_level, confidence_param, True)
            naive_customer_rates[-1].append(bounds[0])
            naive_customer_rates[-1].append(bounds[1])

        for server_level in range(model_bounds.n_levels[1]):
            bounds = parameter_estimator.transition_rate_bounds(state_idx, server_level, confidence_param, False)
            naive_server_rates[-1].append(bounds[0])
            naive_server_rates[-1].append(bounds[1])

    # truncate customer/server rates appropriately
    customer_rates = copy.deepcopy(naive_customer_rates)
    server_rates = copy.deepcopy(naive_server_rates)

    max_cust = max(customer_rates[0])
    min_serv = min(server_rates[0])

    for state in range(1, model_bounds.n_states):
        # truncate appropriately
        customer_rates[state] = [min(x, max_cust) for x in customer_rates[state]]
        max_cust = max(customer_rates[state])

        server_rates[state] = [max(x, min_serv) for x in server_rates[state]]
        min_serv = min(server_rates[state])
    
    min_cust = min(customer_rates[-1])
    max_serv = max(server_rates[-1])

    for state in range(model_bounds.n_states-1, -1, -1):
        customer_rates[state] = [max(x, min_cust) for x in customer_rates[state]]
        min_cust = min(customer_rates[state])

        server_rates[state] = [min(x, max_serv) for x in server_rates[state]]
        max_serv = max(server_rates[state])
    

    # do the reward estimates
    holding_rewards = [parameter_estimator.holding_reward_bounds(state_idx, confidence_param)[1] for state_idx in range(model_bounds.n_states)]
    customer_rewards = []
    server_rewards = []

    for state_idx in range(model_bounds.n_states):
        custr = []
        servr = []
        for level in range(model_bounds.n_levels[0]):
            custr += [parameter_estimator.transition_reward_bounds(state_idx, level, confidence_param, True)[1]]*2
        for level in range(model_bounds.n_levels[1]):
            servr += [parameter_estimator.transition_reward_bounds(state_idx, level, confidence_param, False)[1]]*2
        customer_rewards.append(custr)
        server_rewards.append(servr)

    rewards = model.ModelRewards(holding_rewards, customer_rewards, server_rewards, model_bounds.capacities)

    # remove rates at the top
    customer_rates[-1] = [0 for x in customer_rates[-1]]
    server_rates[0] = [0 for x in server_rates[0]]
    
    # create the model
    ext_model = model.Model(customer_rates, server_rates, rewards, model_bounds.capacities, rng)
    
    return ext_model
