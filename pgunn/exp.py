import numpy as np

def split_2048(vector):
    """
    method splits 2048 gameboard into several vectors
    """
    if np.shape(vector) == (16,):
        vector = [vector]

    tensor = []
    for x in range(8):
        tensor.append(np.zeros((len(vector),4)))

    for i in range(4):
        for e in range(4):
            for y in range(len(vector)):
                tensor[i][y][e] = vector[y][i*4+e]

    for i in range(4):
        for e in range(4):
            for y in range(len(vector)):
                tensor[e+4][y][i] = vector[y][i*4+e]

    return tensor

def check_direction(matrix):
    """
    method checks if fields in vector can be merged
    returns (a, d), (w, s)
    """
    moves = [0,0]

    for i, item in enumerate(matrix):
        if item == 0:
            continue
        if i != 0:
            if matrix[i-1] == 0 and item != 0:
                moves[0] = 1
        for e, elem in enumerate(matrix[(1+i):]):
            if item == elem and item != 0:
                moves[0] = 1
                moves[1] = 1
                return moves[0], moves[1]
            elif elem == 0:
                moves[1] = 1
                continue
            else:
                break

    return moves[0], moves[1]

def possible_moves(matrix):
    """
    method checks if fields in vector can be merged
    returns (w,d,s,a)
    """
    moves = [0,0,0,0]
    w = 0
    a = 0
    s = 0
    d = 0
    
    tensor = split_2048(matrix)

    for i, item in enumerate(tensor):
        if i < 4:
            if moves[3] == 1 and moves[1] == 1:
                continue
            else:
                a, d = check_direction(item[0])
                if a == 1:
                    moves[3] = a
                if d == 1:
                    moves[1] = d
        else:
            if moves[0] == 1 and moves[2] == 1:
                continue
            else:
                w, s = check_direction(item[0])
                if w == 1:
                    moves[0] = w
                if s == 1:
                    moves[2] = s
    return moves

def test():
    #two
    assert check_direction([0,0,0,0]) == (0, 0)
    assert check_direction([2,0,0,0]) == (0, 1)
    assert check_direction([0,2,0,0]) == (1, 1)
    assert check_direction([2,2,0,0]) == (1, 1)
    assert check_direction([2,2,2,2]) == (1, 1)
    assert check_direction([0,0,0,2]) == (1, 0)
    assert check_direction([0,0,2,2]) == (1, 1)
    assert check_direction([0,2,2,2]) == (1, 1)
    assert check_direction([2,0,0,2]) == (1, 1)
    assert check_direction([0,2,2,0]) == (1, 1)
    assert check_direction([2,0,2,0]) == (1, 1)
    assert check_direction([0,2,0,2]) == (1, 1)
    assert check_direction([2,0,2,2]) == (1, 1)
    assert check_direction([2,0,2,2]) == (1, 1)
    #three
    assert check_direction([2,4,0,0]) == (0, 1)
    assert check_direction([2,0,4,0]) == (1, 1)
    assert check_direction([2,0,0,4]) == (1, 1)
    assert check_direction([2,2,0,4]) == (1, 1)
    assert check_direction([2,0,2,4]) == (1, 1)
    assert check_direction([2,0,4,2]) == (1, 1)
    assert check_direction([2,4,0,2]) == (1, 1)
    assert check_direction([2,4,2,0]) == (0, 1)
    assert check_direction([4,2,2,0]) == (1, 1)
    assert check_direction([0,2,2,4]) == (1, 1)
    assert check_direction([0,4,2,2]) == (1, 1)
    assert check_direction([4,2,0,2]) == (1, 1)
    assert check_direction([4,2,2,4]) == (1, 1)
    assert check_direction([4,4,2,2]) == (1, 1)
    assert check_direction([4,2,4,2]) == (0, 0)
    assert check_direction([2,4,2,4]) == (0, 0)
    assert check_direction([4,2,2,2]) == (1, 1)
    #four
    assert check_direction([2,4,6,0]) == (0, 1)
    assert check_direction([0,2,4,6]) == (1, 0)
    assert check_direction([2,0,4,6]) == (1, 1)
    assert check_direction([2,4,0,6]) == (1, 1)
    assert check_direction([2,2,4,6]) == (1, 1)
    assert check_direction([2,4,4,6]) == (1, 1)
    assert check_direction([2,4,6,6]) == (1, 1)
    assert check_direction([2,6,4,6]) == (0, 0)
    assert check_direction([4,6,4,2]) == (0, 0)
    assert check_direction([4,2,6,4]) == (0, 0)
    assert check_direction([4,2,8,16]) == (0, 0)


matrix = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
test()
print(possible_moves(matrix))

import gym
import gym_2048

env = gym.make("2048-v0")
state = env.reset()
while True:
    action = np.random.randint(0, 4, size=1)[0]

    env.render()
    print(possible_moves(list(state)))

    state, reward, done, info = env.step(action)
