# import numpy as np


list_states = []

for i in range(4):
    for j in range(3):
        list_states.append((i, j))


def transition_matrix_up(tuple_state, tuple_new_state):
    x, y = tuple_state
    x_new, y_new = tuple_new_state

    delta_x = abs(x - x_new)
    delta_y = abs(y - y_new)

    proba = 0
    if (delta_x + delta_y) > 1:
        return proba

    if tuple_state == tuple_new_state:
        if x == 0:
            proba += 0.1
            if y == 1:
                proba += 0.1
            elif y == 2:
                proba += 0.8
        elif x == 1:
            proba += 0.8
        elif x == 2:
            if y == 1:
                proba += 0.1
            if y == 2:
                proba += 0.8
        elif x == 3:
            proba += 0.1

    elif y + 1 == y_new:
        proba += 0.8

    # Right move
    elif (x + 1) == x_new:
        if x == 0:
            if (y == 0) or (y == 2):
                proba += 0.1
        elif x == 1:
            proba += 0.1
        elif x == 2:
            proba += 0.1

    # Left move
    elif (x - 1) == x_new:
        if x == 1:
            if (y == 0) or (y == 2):
                proba += 0.1
        elif x == 2:
            if (y == 0) or (y == 2):
                proba += 0.1
        elif x == 3:
            proba += 0.1

    return proba


def transition_matrix_down(tuple_state, tuple_new_state):
    x, y = tuple_state
    x_new, y_new = tuple_new_state

    delta_x = abs(x - x_new)
    delta_y = abs(y - y_new)

    proba = 0
    if (delta_x + delta_y) > 1:
        return proba

    if tuple_state == tuple_new_state:
        if x == 0:
            proba += 0.1
            if y == 1:
                proba += 0.1
            elif y == 0:
                proba += 0.8
        elif x == 1:
            proba += 0.8
        elif x == 2:
            if y == 1:
                proba += 0.1
            if y == 0:
                proba += 0.8
        elif x == 3:
            proba += 0.9

    elif y - 1 == y_new:
        proba += 0.8

    # Right move
    elif (x + 1) == x_new:
        if x == 0:
            if (y == 0) or (y == 2):
                proba += 0.1
        elif x == 1:
            proba += 0.1
        elif x == 2:
            proba += 0.1

    # Left move
    elif (x - 1) == x_new:
        if x == 1:
            if (y == 0) or (y == 2):
                proba += 0.1
        elif x == 2:
            if (y == 0) or (y == 2):
                proba += 0.1
        elif x == 3:
            proba += 0.1

    return proba



def transition_matrix_left(tuple_state, tuple_new_state):
    x, y = tuple_state
    x_new, y_new = tuple_new_state

    delta_x = abs(x - x_new)
    delta_y = abs(y - y_new)

    proba = 0
    if (delta_x + delta_y) > 1:
        return proba

    if tuple_state == tuple_new_state:
        if x == 0:
            proba += 0.8
            if y == 0:
                proba += 0.1
            elif y == 2:
                proba += 0.1
        elif x == 1:
            proba += 0.2
        elif x == 2:
            if y == 0:
                proba += 0.1
            elif y == 1:
                proba += 0.8
            elif y == 2:
                proba += 0.1
        elif x == 3:
            proba += 0.1

    # Left move
    elif (x - 1) == x_new:
        proba += 0.8

    # Up move perturbation
    elif (y + 1) == y_new:
        if x == 0:
            if (y == 0) or (y == 1):
                proba += 0.1
        elif x == 2:
            if (y == 0) or (y == 1):
                proba += 0.1
        elif x == 3:
            proba += 0.1


    # Down move perturbation
    elif (y - 1) == y_new:
        if x == 0:
            if (y == 1) or (y == 2):
                proba += 0.1
        elif x == 2:
            if (y == 1) or (y == 2):
                proba += 0.1

    return proba


def transition_matrix_right(tuple_state, tuple_new_state):
    x, y = tuple_state
    x_new, y_new = tuple_new_state

    delta_x = abs(x - x_new)
    delta_y = abs(y - y_new)

    proba = 0
    if (delta_x + delta_y) > 1:
        return proba

    if tuple_state == tuple_new_state:
        if x == 0:
            if (y == 0) or (y == 2):
                proba += 0.1
            if y == 1:
                proba += 0.8
        elif x == 1:
            proba += 0.2
        elif x == 2:
            if (y == 0) or (y == 2):
                proba += 0.1
        elif x == 3:
            proba += 0.9

    # Right move
    elif (x + 1) == x_new:
        proba += 0.8

    # Upward move perturbation
    elif (y + 1) == y_new:
        if x == 0:
            if (y == 0) or (y == 1):
                proba += 0.1
        elif x == 2:
            if (y == 0) or (y == 1):
                proba += 0.1
        elif x == 3:
            proba += 0.1

    # Downward move perturbation
    elif (y - 1) == y_new:
        if x == 0:
            if (y == 1) or (y == 2):
                proba += 0.1
        elif x == 2:
            if (y == 1) or (y == 2):
                proba += 0.1

    return proba

# reverif tous les x et y axes SURTOUT SUR MAZE





