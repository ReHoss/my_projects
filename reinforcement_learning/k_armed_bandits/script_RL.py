import numpy as np
import random
import scipy.stats as stats
import pdb

# Soit la machine A
N_RUN = 1
N_STEP = 1000
N_BANDITS = 10
EPSILON = 0.1


def k_bandits_simulation(n_run=2000,
                         n_step=1000,
                         n_bandits=10,
                         epsilon=0,
                         value_optimist=0.0,
                         alpha=None,
                         ucb=None,
                         c=2):
    matrix_history = np.zeros([n_run, n_step], dtype=float)
    matrix_choice = np.zeros([n_run, n_step], dtype=float)
    array_best_choice = np.zeros(n_run)
    for n in range(n_run):
        # Generating true rewards values
        array_q_star = stats.norm.rvs(size=n_bandits)
        # Initialisation action value estimation
        array_q_hat = np.repeat(value_optimist, n_bandits)
        array_q_hat = array_q_hat.astype(float)
        # Initialise history arrays
        array_reward_hist = np.zeros(n_step)
        array_index_chosen = np.zeros(n_bandits)
        # Remembering the best rewarding action
        array_best_choice[n] = np.argmax(array_q_star)
        # Initial state
        index_chosen = random.randint(0, n_bandits - 1)
        matrix_choice[n, 0] = index_chosen
        array_index_chosen[index_chosen] += 1
        q_star_i = array_q_star[index_chosen]
        reward_i = stats.norm.rvs(q_star_i)
        # Estimator first update
        size = array_index_chosen[index_chosen]
        q_hat = array_q_hat[index_chosen]
        if alpha:
            q_hat = q_hat + alpha * (reward_i - q_hat)
        else:
            q_hat = q_hat + (1 / size) * (reward_i - q_hat)
        array_q_hat[index_chosen] = q_hat
        # array_q_hat[index_chosen] = reward_i
        array_reward_hist[0] = reward_i

        # Run start
        for i in range(1, n_step):
            # Compute choice : greedy or random
            if (epsilon == 0) or (epsilon < stats.uniform.rvs()):
                if ucb:
                    # in UCB we must act at least once
                    if 0 in array_index_chosen:
                        array_argwhere = np.argwhere(array_index_chosen == 0)
                        index_chosen = array_argwhere[0][0]
                    else:
                        array_upper_bound = array_q_hat.copy()
                        array_upper_bound += ucb * c * np.sqrt(np.log(i) / array_index_chosen)
                        index_chosen = np.argmax(array_upper_bound)

                else:
                    index_chosen = np.argmax(array_q_hat)
            else:
                index_chosen = random.randint(0, n_bandits - 1)

            array_index_chosen[index_chosen] += 1
            matrix_choice[n, i] = index_chosen
            # Compute ith reward
            q_star_i = array_q_star[index_chosen]
            reward_i = stats.norm.rvs(q_star_i)
            # Stock reward
            array_reward_hist[i] = reward_i

            # Estimator update
            size = array_index_chosen[index_chosen]
            q_hat = array_q_hat[index_chosen]
            if alpha:
                q_hat = q_hat + alpha * (reward_i - q_hat)
            else:
                q_hat = q_hat + (1 / size) * (reward_i - q_hat)
            array_q_hat[index_chosen] = q_hat

        matrix_history[n] = array_reward_hist

    return matrix_history, matrix_choice, array_best_choice


def compute_soft_max(h, n_bandits):  # h vecteur
    """ this function computed the function softmax over a vector h of len n_bandits"""
    p = np.zeros(n_bandits)
    for i in range(n_bandits):
        p[i] = np.exp(h[i]) / np.sum(np.exp(h))
    return p


def choose_armed_following_distribution(p):
    """ this function is taking a vector of probability as an entry and
    return a realisation of a ramdom variable following p
    """
    p_cum_sum = np.cumsum(p)
    u = np.random.uniform()
    i = 0
    while u >= p_cum_sum[i]:
        i += 1
    return i


def update_array_h(array_h, index, reward, baseline, p, alpha, n_bandits):
    """ this function is applying the update rule of the Stochastic Gradient Ascent algorithme"""
    for i in range(n_bandits):
        if i == index:
            array_h[i] += alpha * (reward - baseline) * (1 - p[i])
        else:
            array_h[i] -= alpha * (reward - baseline) * p[i]
    return array_h


def bandit_gradient(n_run=2000,
                    n_step=1000,
                    n_bandits=10,
                    alpha=0.1,
                    baseline_bool=True):
    matrix_choice = np.zeros([n_run, n_step], dtype=float)
    array_best_choice = np.zeros(n_run)
    for n in range(n_run):
        # Generating true rewards values
        array_q_star = stats.norm.rvs(loc=4, size=n_bandits)
        # Initialisation action value estimation
        array_h = np.zeros(n_bandits)
        # Initialise history arrays
        array_reward_hist = np.zeros(n_step)
        # Remembering the best rewarding action
        array_best_choice[n] = np.argmax(array_q_star)

        # Run start
        for i in range(n_step):
            # we compute the vector of probability against array_h
            proba = compute_soft_max(array_h, n_bandits)
            # we choose an index thanks to the distribution "proba"
            index_chosen = choose_armed_following_distribution(proba)
            matrix_choice[n, i] = index_chosen
            # compute reward
            q_star_i = array_q_star[index_chosen]
            reward_i = stats.norm.rvs(q_star_i)
            array_reward_hist[i] = reward_i
            if i != 0:  # don't update at initialisation
                if baseline_bool:
                    baseline = np.mean(array_reward_hist[:i])  # index i is exclude, so baseline is computed until step i-1 (last reward is not included in baseline)
                    # update of array_h
                    array_h = update_array_h(array_h, index_chosen, reward_i, baseline, proba, alpha, n_bandits)
                else:
                    array_h = update_array_h(array_h, index_chosen, reward_i, 0, proba, alpha, n_bandits)

    return matrix_choice, array_best_choice

# matrix_choice, array_best_choice = bandit_gradient(n_run=2000,n_step=1000,n_bandits=10,alpha=0.1,baseline_bool=True)
