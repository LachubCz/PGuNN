"""
file contains support methods for processing of states
"""
import numpy as np
import scipy
import scipy.misc

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
