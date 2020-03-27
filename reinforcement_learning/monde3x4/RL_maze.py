import numpy as np
from scipy.stats import uniform



# commenter

# noinspection PyShadowingNames
class Environment:
    matrix_world = np.array([[0, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, 0, 0]])

    _len_x_axis = matrix_world.shape[1]
    _len_y_axis = matrix_world.shape[0]

    def __init__(self, tuple_init_state=(0, 0)):
        self.tuple_actual_state = tuple_init_state

    def update_state_from_action(self, str_action):
        """Return new state from action, if its a wall we do not move !"""

        tuple_new_state = self._action_to_new_state_coordonates(str_action)
        bool_legal_state = not self._check_wall(tuple_new_state)
        if bool_legal_state:
            # Update state by new coordinates
            self.tuple_actual_state = tuple_new_state

    def reward_from_state(self, reward=-0.04):
        """Return the reward associated to the state t"""
        if self.tuple_actual_state == (3, 2):
            reward = 1
        elif self.tuple_actual_state == (3, 1):
            reward = -1
        else:
            reward = reward

        return reward

    def check_end(self):
        """Return bool"""
        if self.tuple_actual_state == (3, 2):
            return True
        elif self.tuple_actual_state == (3, 1):
            return True
        else:
            return False

    def _check_wall(self, tuple_state):
        """Returm True if tuple_state is not valid (pass through the wall"""
        if tuple_state[0] in [-1, self._len_x_axis]:
            return True
        elif tuple_state[1] in [-1, self._len_y_axis]:
            return True
        elif tuple_state == (1, 1):
            return True
        else:
            return False

    def print_board(self):
        x, y = self.tuple_actual_state
        i = (self._len_y_axis - 1) - y
        j = x
        matrix_copy = self.matrix_world.copy()
        matrix_copy[i, j] = 1
        print(matrix_copy)

    def _action_to_new_state_coordonates(self, str_action):
        """Given an action, valid or not, compute new coordonates"""
        if str_action not in ['LEFT', 'RIGHT', 'UP', 'DOWN']:
            raise ValueError(f'Wrong movement: {str_action}')

        x_actual = self.tuple_actual_state[0]
        y_actual = self.tuple_actual_state[1]
        if str_action in ['LEFT', 'RIGHT']:
            # If the action is left or right we compute x-axis new value
            if str_action == 'LEFT':
                # Going to left reduce x-axis coordinate by 1
                return (x_actual - 1, y_actual)
            else:
                # Going to right increase x-axis coordinate by 1
                return (x_actual + 1, y_actual)
        else:
            # If the action is up or down we look at y-axis new value
            if str_action == 'UP':
                # Going up increase y-axis coordinate by 1
                return (x_actual, y_actual + 1)
            else:
                # Going down decrease y-axis coordinate by 1
                return (x_actual, y_actual - 1)


# noinspection PyShadowingNames
class Agent:

    def __init__(self):
        self.list_actions = []
        self.list_states = []
        self.list_rewards = []

    @staticmethod
    def act(str_action):
        """Compute action perturbation and records the move"""

        if str_action not in ['LEFT', 'RIGHT', 'UP', 'DOWN']:
            raise ValueError(f'Wrong action: {str_action}')

        random_value = uniform.rvs()
        # Perturbated case
        if random_value <= 0.2:
            if str_action in ['UP', 'DOWN']:
                if random_value > 0.1:
                    str_new_action = 'LEFT'
                else:
                    str_new_action = 'RIGHT'
            else:
                if random_value > 0.1:
                    str_new_action = 'UP'
                else:
                    str_new_action = 'DOWN'
        else:
            str_new_action = str_action
        # Update list
        return str_new_action

    def update_action_list(self, str_action):
        self.list_actions.append(str_action)

    def update_state_list(self, tuple_state):
        self.list_states.append(tuple_state)

    def update_rewards_list(self, reward):
        self.list_rewards.append(reward)


if __name__ == '__main__':

    env = Environment()
    agent = Agent()
    agent.update_state_list(env.tuple_actual_state)
    #
    # while not env.check_end():
    #     # str_action = input()
    #     str_action = agent.act(str_action)
    #     print(str_action)
    #     agent.update_action_list(str_action)
    #     env.update_state_from_action(str_action)
    #     agent.update_state_list(env.tuple_actual_state)
    #     env.print_board()
    #     reward = env.reward_from_state()
    #     agent.update_rewards_list(reward)
    #     print(reward)
    #
