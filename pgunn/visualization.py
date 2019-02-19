"""
file contains methods for visualization of learning progress
"""
import os
import os.path
import re
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', UserWarning)
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.ndimage.filters import gaussian_filter

from tools import get_name, err_print

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


def get_args():
    """
    method for parsing of arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-filename", action="store", dest="filename", required=True,
                        help="name of file containing output of training for processing and visualization")
    parser.add_argument("-graph_name", action="store", dest="graph_name", required=True,
                        help="name of output file containing processed graph")
    parser.add_argument("-idx_val", action="store", dest="idx_val", type=int, required=True,
                        help="index of column with relevant data")
    parser.add_argument("-coordinate_x", action="store", dest="coordinate_x", type=int, default=None,
                        help="maximum x coordinate")
    parser.add_argument("-coordinate_y", action="store", dest="coordinate_y", type=int, default=None,
                        help="maximum y coordinate")
    parser.add_argument("-lines", action="store", dest="lines", nargs='+', type=int, default=None,
                        help="y coordinates of arbitrary reference lines")
    parser.add_argument("-scatter", action="store_true", dest="scatter", default=False,
                        help="graph will show results from every round")

    args = parser.parse_args()

    if not os.path.isfile(args.filename):
        err_print("[File doesn't exist.]")
        sys.exit(-1)

    return args


def combined_graph(scores, episode_numbers, name, coordinates=None, linears=None, scatter=False):
    """
    method prints point graph and
    interpolation graph with gaussian filter of learning progress
    """
    if linears is not None:
        for key, value in linears.items():
            plt.plot([0, episode_numbers[-1]], [key, value], 'k-', linewidth=0.8)

    if scatter:
        plt.plot(episode_numbers, scores, 'ro', color='goldenrod', markersize=1)

    score_gf = gaussian_filter(scores, sigma=0.01791*episode_numbers[-1])

    plt.plot(episode_numbers, score_gf, color='teal', linewidth=1)

    plt.ylabel("Score")
    plt.xlabel("Episode")

    plt.xlim([0,coordinates[0]])
    if min(scores) < 0:
        plt.ylim([min(scores),coordinates[1]])
    else:
        plt.ylim([0,coordinates[1]])

    name = get_name(name)

    plt.savefig("./{}" .format(name), bbox_inches='tight')
    plt.clf()
    print("[Graph of learning progress visualization was saved to \"./{}\".]" .format(name))


def heat_map(array, graph_name, axes):
    """
    method creates heatmap
    """
    fig, ax = plt.subplots()
    im = ax.imshow(array, cmap=cm.YlOrRd)
    if not axes:
        im.axes.get_yaxis().set_visible(False)
        im.axes.get_xaxis().set_visible(False)
    fig.tight_layout()

    graph_name = get_name(graph_name)
    plt.savefig("./{}" .format(graph_name), bbox_inches='tight')
    print("[Heatmap was made.]")


def split_data(line):
    """
    method splits varibles on line
    """
    data = list()
    arr = np.array([string for string in line.split(", ")], dtype=str)

    for _, item in enumerate(arr):
        word_parse = re.compile(r'''  ((?<=:.)-*[0-9]+\.*[0-9]*)''', re.X)
        parts = word_parse.findall(item)

        if parts != []:
            data.append(float(parts[0]))

    if len(data) > 1:
        return data
    else:
        return []


def read_file(filename):
    """
    method opens file, processes it line by line and returns data with episode numbers
    """
    data = list()
    episode_numbers = list()
    counter = 0

    file = open(os.getcwd()+ "/" + filename, "r")

    while True:
        record = file.readline()
        if record == "":
            return data, episode_numbers

        processed = split_data(record)
        if processed:
            data.append(processed)
            episode_numbers.append(counter)
            counter=counter + 1


def get_visualization(filename, graph_name, idx_val, coordinate_x=None, coordinate_y=None, lines=None, scatter=False):
    """
    method reads and proceses file with learning log
    """
    if lines != None:
        linears_dict = dict()
        for _, item in enumerate(lines):
            linears_dict[item] = item
    else:
        linears_dict = None

    values = list()
    data, counter = read_file(filename)

    for _, item in enumerate(data):
        values.append(item[idx_val])

    if coordinate_x == None:
        coordinate_x = counter[-1]
    if coordinate_y == None:
        coordinate_y = max(values)+10

    combined_graph(values, counter, graph_name, [coordinate_x, coordinate_y], linears_dict, scatter)
    print("[SUCCESSFUL RUN]")


if __name__ == "__main__":
    args = get_args()
    get_visualization(args.filename, args.graph_name, args.idx_val, args.coordinate_x, args.coordinate_y, args.lines, args.scatter)
