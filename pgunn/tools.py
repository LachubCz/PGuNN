"""
file contains support methods for processing of states
"""
import os
import numpy as np
import scipy
import scipy.misc

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
