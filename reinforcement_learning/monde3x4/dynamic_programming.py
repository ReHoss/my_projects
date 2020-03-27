import numpy as np
import transition_matrix
from RL_maze import Agent, Environment

# La verif a proposer
# Ne pas oublier de fournir tous les scripts

LIST_STATES = [(i, j) for i in range(4) for j in range(3)]
LIST_STATES.remove((1, 1))
LIST_ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
LIST_TRANSITION_MATRIX = [transition_matrix.transition_matrix_up,
                          transition_matrix.transition_matrix_down,
                          transition_matrix.transition_matrix_left,
                          transition_matrix.transition_matrix_right]

DICT_STATES = {key: value for (key, value) in enumerate(LIST_STATES)}
DICT_STATES.update(dict([reversed(i) for i in DICT_STATES.items()]))

LIST_NON_FINAL_STATES = list(LIST_STATES)
LIST_NON_FINAL_STATES.remove((3, 1))
LIST_NON_FINAL_STATES.remove((3, 2))


def value_iteration(array_policy,
                    array_v,
                    epsilon=0.01,
                    gamma=0.9,
                    r=-0.04):
    array_policy = array_policy.copy()
    while True:
        delta = 0
        for tuple_state in LIST_NON_FINAL_STATES:
            state = DICT_STATES[tuple_state]
            v = array_v[state]

            list_expectation = []

            for action in range(len(LIST_ACTIONS)):
                expectation = 0
                funct_transition_matrix = LIST_TRANSITION_MATRIX[action]
                for tuple_new_state in LIST_STATES:
                    new_state = DICT_STATES[tuple_new_state]
                    proba_transit = funct_transition_matrix(tuple_state,
                                                            tuple_new_state)
                    expectation += proba_transit * (
                            r + gamma * array_v[new_state])

                list_expectation.append(expectation)

            array_v[state] = np.max(list_expectation)
            delta = np.max([delta, np.abs(v - array_v[state])])

        if delta < epsilon:
            break

        for tuple_state in LIST_NON_FINAL_STATES:
            state = DICT_STATES[tuple_state]

            list_expectation = []

            for action in range(len(LIST_ACTIONS)):
                expectation = 0
                funct_transition_matrix = LIST_TRANSITION_MATRIX[action]
                for tuple_new_state in LIST_STATES:
                    new_state = DICT_STATES[tuple_new_state]
                    proba_transit = funct_transition_matrix(tuple_state,
                                                            tuple_new_state)
                    expectation += proba_transit * (
                            r + gamma * array_v[new_state])

                list_expectation.append(expectation)

            array_policy[state] = np.argmax(list_expectation)
    return array_policy


def policy_evaluation(array_policy,
                      array_v,
                      epsilon=0.01,
                      gamma=0.9,
                      r=-0.04):
    while True:
        delta = 0
        for tuple_state in LIST_NON_FINAL_STATES:
            state = DICT_STATES[tuple_state]
            v = array_v[state]

            action = array_policy[state]
            funct_transition_matrix = LIST_TRANSITION_MATRIX[action]

            expectation = 0
            for tuple_new_state in LIST_STATES:
                new_state = DICT_STATES[tuple_new_state]
                proba_transit = funct_transition_matrix(tuple_state,
                                                        tuple_new_state)
                expectation += proba_transit * (r + gamma * array_v[new_state])

            array_v[state] = expectation
            delta = np.max([delta, np.abs(v - array_v[state])])

        if delta < epsilon:
            break


def policy_improvement(array_policy,
                       array_v,
                       gamma=0.9,
                       r=-0.04):
    bool_flag = True
    for tuple_state in LIST_NON_FINAL_STATES:
        state = DICT_STATES[tuple_state]
        old_action = array_policy[state]
        list_expectation = []

        for action in range(len(LIST_ACTIONS)):
            expectation = 0
            funct_transition_matrix = LIST_TRANSITION_MATRIX[action]
            for tuple_new_state in LIST_STATES:
                new_state = DICT_STATES[tuple_new_state]
                proba_transit = funct_transition_matrix(tuple_state,
                                                        tuple_new_state)
                expectation += proba_transit * (
                        r + gamma * array_v[new_state])

            list_expectation.append(expectation)

        array_policy[state] = np.argmax(list_expectation)
        if old_action != array_policy[state]:
            bool_flag = False

    return bool_flag


def policy_iteration(array_policy,
                     array_v,
                     epsilon=0.01,
                     gamma=0.9,
                     r=-0.04):
    array_policy = array_policy.copy()
    bool_flag = False
    i = 0
    while not bool_flag:
        policy_evaluation(array_policy,
                          array_v,
                          epsilon=epsilon,
                          gamma=gamma,
                          r=r)
        bool_flag = policy_improvement(array_policy,
                                       array_v,
                                       gamma=gamma,
                                       r=r)
        # print(i)
        # print(bool_flag)
        # print_policy(array_policy)
    return array_policy


def print_policy(array_policy):
    matrix_grid = np.empty((3, 4), dtype=object)
    matrix_grid[1, 1] = 'X'
    matrix_grid[1, 3] = ' '
    matrix_grid[0, 3] = ' '

    # list_symbol = ['^', 'v', '<', '>']
    list_symbol = [u'\u2191', u'\u2193', u'\u2190', u'\u2192']

    for i, action in enumerate(array_policy):
        x, y = DICT_STATES[i]
        matrix_grid[2 - y, x] = list_symbol[action]

    for arr in matrix_grid:
        print('|{}|{}|{}|{}|'.format(*arr))

    # print(matrix_grid)


def print_values(array_values):
    matrix_grid = np.empty((3, 4), dtype=object)
    matrix_grid[1, 1] = 'XXXXX'
    matrix_grid[1, 3] = ' '
    matrix_grid[0, 3] = ' '

    for i, value in enumerate(array_values):
        x, y = DICT_STATES[i]
        # value = round(value, 2)

        matrix_grid[2 - y, x] = '{:0.2f}'.format(value)

    for arr in matrix_grid:
        print('|{}|{}|{}|{}|'.format(*arr))


# policy_iteration(r=-0.04)


def episode_simulation(array_policy,
                       verbose=False):
    env = Environment()
    agent = Agent()
    agent.update_state_list(env.tuple_actual_state)
    if verbose:
        print(f'Initial Environment:')
        env.print_board()
    while not env.check_end():
        action = array_policy[DICT_STATES[env.tuple_actual_state]]
        str_action = LIST_ACTIONS[action]
        if verbose:
            print(f'-----------------------')
            print(f'Action choosed by agent: {str_action}')
        str_action = agent.act(str_action)
        if verbose:
            print(f'Action after perturbation: {str_action}')

        agent.update_action_list(str_action)
        env.update_state_from_action(str_action)
        agent.update_state_list(env.tuple_actual_state)
        if verbose:
            print(f'Environment:')
            env.print_board()
        reward = env.reward_from_state()
        agent.update_rewards_list(reward)
        if verbose:
            print(f'Reward received: {reward}')
            print(f'-----------------------')

    return agent


def mc_state_evaluation(array_policy,
                        array_v,
                        str_mode,
                        n_step=100,
                        gamma=0.9):
    array_n = np.zeros(len(LIST_STATES), dtype=int)
    array_acc = np.zeros(len(LIST_STATES), dtype=float)

    for i in range(n_step):
        agent = episode_simulation(array_policy)

        if str_mode == 'first':
            array_visited_state = np.zeros(len(LIST_NON_FINAL_STATES),
                                           dtype=int)

        for t, tuple_state in enumerate(agent.list_states[:-1]):
            state = DICT_STATES[tuple_state]

            # noinspection PyUnboundLocalVariable
            if (str_mode == 'first') and (array_visited_state[state] == 0):
                array_visited_state[state] += 1

            array_reward = np.array(agent.list_rewards[t:],
                                    dtype=float)
            # From tuples to state ID
            array_gamma = np.array([gamma] * len(array_reward),

                                   dtype=float)
            array_gamma = np.power(array_gamma,
                                   np.arange(len(array_reward)))
            cum_reward = np.dot(array_gamma,
                                array_reward)

            array_acc[state] += cum_reward
            array_n[state] += 1
            array_v[state] = array_acc[state] / array_n[state]

            if str_mode == 'first':
                if np.sum(array_visited_state) == len(array_visited_state):
                    break
    return array_v


if __name__ == '__main__':
    ARRAY_V = np.zeros(len(LIST_STATES))
    ARRAY_V[DICT_STATES[(3, 1)]] = -1
    ARRAY_V[DICT_STATES[(3, 2)]] = 1

    ARRAY_POLICY = np.zeros(len(LIST_NON_FINAL_STATES),
                            dtype=int)
    ARRAY_POLICY += 3

    mc_state_evaluation(ARRAY_POLICY, ARRAY_V, gamma=0.9, str_mode='first')

# Comparer les methodes ::: Sutton et Barto
