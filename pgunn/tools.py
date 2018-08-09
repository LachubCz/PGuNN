"""
file contains support methods for processing of states
"""
import os
import sys
import time
import numpy as np
import scipy
import scipy.misc
from tqdm import trange

def split_2048(vector):
    """
    method splits 2048 gameboard into several vectors
    """
    if np.shape(vector) == (16,):
        vector = [vector]

    tensor = []
    for _ in range(8):
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

def create_buffer(task, state):
    """
    method creates buffer
    """
    if task.args.frames > 1:
        shape = list(np.shape(state))
        shape.insert(0, task.args.frames)
        buff = np.full(tuple(shape), state)
        return buff
    else:
        return state

def shift_buffer(task, buff, new_state):
    """
    method shifts buffer and adds new state
    """
    if task.args.frames > 1:
        new = np.empty_like(buff)

        new[-1:] = new_state
        new[:-1] = buff[1:]

        return new
    else:
        return new_state

def process_img(img):
    """
    method proceses image - make it gray and normalize
    """
    img = scipy.misc.imresize(img, (84, 84), interp='bilinear')

    red, green, blue = img[:,:,0], img[:,:,1], img[:,:,2]
    img = 0.299 * red + 0.587 * green + 0.114 * blue

    img = img.astype(np.uint8) / 255.0 - 0.5

    return img

def normalize(task, state):
    """
    method normalizes states
    """
    if task.type == "text":
        return state / 16384.0 - 0.5
    elif task.type == "ram":
        return state / 255.0 - 0.5
    elif task.type == "image":
        return process_img(state)
    else:
        return state

def get_name(graph_name):
    """
    method creates name for pdf
    """
    if os.path.isfile(graph_name+".pdf"):
        i=1
        while True:
            if not os.path.isfile(graph_name+"("+str(i)+").pdf"):
                graph_name = graph_name+"("+str(i)+").pdf"
                break
            i+=1
    else:
        graph_name = graph_name + ".pdf"

    return graph_name

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
        for _, elem in enumerate(matrix[(1+i):]):
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

def rand_score_estimate(task, games):
    """
    score estimation for random agent
    """
    total_reward = 0

    bar = trange(games, leave=True)
    for game in bar:
        task.env.reset()
        done = False

        while not done:
            action = np.random.randint(0, task.env_action_size, size=1)[0]
            _, reward, done, _ = task.env.step(action)
            total_reward = total_reward + reward

        if game != 0:
            bar.set_description("Average score: " + str(round(total_reward / (game+1), 2)))

    print("[Random agent average score: {}]" .format(round(total_reward / games, 2)))

    return total_reward / games

def agent_score_estimate(task, games, render=False, show_bar=False):
    """
    score estimation for trained agent
    """
    total_reward = 0
    bar = trange(games, leave=True, disable=(not show_bar))
    actions = None
    for game in bar:
        state = task.env.reset()
        last_state = state
        wrong_move = False
        done = False

        state = normalize(task, state)
        state = create_buffer(task, state)

        while not done:
            if render and task.type == "text":
                time.sleep(0.5)
                task.env.render()
            elif render:
                time.sleep(0.03)
                task.env.render()

            if task.name == "2048-v0" and wrong_move:
                action = np.argmax(actions)
                wrong_move = False
            else:
                if task.name == "Breakout-v0" or task.name == "Breakout-ram-v0" or task.name == "BeamRider-v0":
                    action = task.agent.get_action(state, epsilon=True)
                else:
                    actions = task.agent.model_net.predict(np.array([state]))
                    action = np.argmax(actions)

            next_state, reward, done, info = task.env.step(action)

            if task.name == "2048-v0":
                if sum(last_state) == sum(next_state):
                    wrong_move = True
                    actions[0][action] = -10000

            last_state = next_state
            next_state = normalize(task, next_state)
            next_state = shift_buffer(task, state, next_state)

            state = next_state
            total_reward = total_reward + reward
        if game != 0:
            bar.set_description("Average score: " + str(round(total_reward / (game+1), 2)))

    if show_bar:
        print("[Agent's average score: {}]" .format(round(total_reward / games, 2)))

    return total_reward / games

def load_memories(task, rnd, normalize_score=True):
    """
    method creates memories without training
    """
    new_observations = 0
    task.agent.clear_memory()

    while True:
        state = task.env.reset()
        last_state = state

        state = normalize(task, state)
        state = create_buffer(task, state)

        for _ in range(task.max_steps):
            if rnd:
                action = np.random.randint(0, task.env_action_size, size=1)[0]
            else:
                action = task.agent.get_action(task, state, last_state, epsilon=True)
            next_state, reward, done, _ = task.env.step(action)

            if normalize_score and task.type != "basic":
                if reward > 0.0:
                    reward = 1.0
                else:
                    reward = 0.0
                if task.type == "text":
                    if sum(last_state) == sum(next_state):
                        reward = -0.1

                if done:
                    reward = -100.0

            new_state = next_state
            last_state = next_state
            next_state = normalize(task, next_state)
            next_state = shift_buffer(task, state, next_state)

            if task.name == "2048-v0":
                task.agent.remember(last_state, action, reward, new_state, done, rand_agent=False)
            else:
                task.agent.remember(state, action, reward, next_state, done, rand_agent=False)
            new_observations = new_observations + 1

            state = next_state

            if task.args.memory == "basic":
                if len(task.agent.memory) == task.agent.memory_size:
                    print("[Agent added {} new memories. Current memory_size: {}]"
                          .format(new_observations, len(task.agent.memory) if task.agent.memory_type == "basic" else task.agent.memory.length))
                    return new_observations
            else:
                if task.agent.memory.length == task.agent.memory_size:
                    print("[Agent added {} new memories. Current memory_size: {}]"
                          .format(new_observations, len(task.agent.memory) if task.agent.memory_type == "basic" else task.agent.memory.length))
                    return new_observations

            if done:
                break

def err_print(*args, **kwargs):
    """
    method for printing to stderr
    """
    print(*args, file=sys.stderr, **kwargs)
