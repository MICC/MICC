__author__ = 'Matt'
import re
import numpy as np


def relabel(ladder):
    """
    Relabels a ladder to a "standard" form where the first vertex's top edge is
    labeled 1, and the ith consecutive edge is labeled i. Useful for testing
    and removing duplicates.

    :param ladder: Any curve pair in ladder representation
    :return:
    """
    n = len(ladder[0])
    labels = range(1, n + 1)
    if type(ladder[0][0]) == tuple:
        ladder = [[x for x, y in ladder[0]], [x for x, y in ladder[1]]]
    ladder = np.array(ladder)
    newLadder = np.zeros_like(ladder)
    currentPos = (0, 0)
    while labels:
        val = ladder[currentPos]
        locations = np.where(ladder[:, :] == val)
        locations = np.array((locations[0], locations[1])).T
        row1 = np.where(currentPos[1] != locations[:, 1])
        newPos = tuple(locations[row1[0]][0])
        edge = labels.pop(0)
        newLadder[currentPos] = edge
        newLadder[newPos] = edge
        currentPos = tuple([(newPos[0] + 1) % 2, newPos[1]])

    return [list(newLadder[0, :]), list(newLadder[1, :])]


def ladder_to_cycle(ladder_top, ladder_bottom):
    """
    Converts a curve pair in a ladder representation to one in cycle notation

    :param ladder_top: the top identifications of a ladder
    :param ladder_bottom: the bottom identifications of a ladder
    :return: the cycle representation of the given ladder
    """

    # construct a dictionary of locations from the ladder's given identifications
    locations = {arc: {'top': [], 'bottom': []} for arc in
                 set(ladder_top + ladder_bottom)}
    for loc, varc in enumerate(ladder_top):
        locations[varc]['top'].append(loc)

    for loc, varc in enumerate(ladder_bottom):
        locations[varc]['bottom'].append(loc)

    n = len(ladder_top)
    cycle = ''
    # arbitrarily orient the curve positively
    orientation = '+'
    current = 'top'
    start = 1
    prev_loc = (ladder_top + ladder_bottom).index(start) % n
    for i in range(n):
        #get the top and bottom of the current vertex
        top = locations[start]['top']
        bottom = locations[start]['bottom']

        # if there is an endpoint on both the top and bottom,
        # then orientation is preserved and the location is simply
        # whatever hasn't already been used. We switch side of the
        # ladder accordingly
        if top and bottom:
            # switch side of ladder
            current = 'top' if current == 'bottom' else 'bottom'
            # preserve orientation
            orientation = '-' if orientation == '-' else '+'
            # get the locations avaiable there
            locs = locations[start][current]
            cycle += str(locs[0] + 1) + orientation
            if current == 'bottom':
                start = ladder_top[locs[0]]
            if current == 'top':
                start = ladder_bottom[locs[0]]
            prev_loc = locs[0]
            current = 'top' if current == 'bottom' else 'bottom'

        # if there isn't an endpoint on top, rather just on the bottom,
        # there is an orientation switch and it requires a bit more work
        # to find the location
        elif not top:
            current = 'bottom'
            orientation = '-' if orientation == '+' else '+'
            locs = locations[start][current]
            t = set(locs)
            t.remove(prev_loc)
            current_loc = t.pop()
            cycle += str(current_loc + 1) + orientation
            start = ladder_top[current_loc]
            prev_loc = current_loc
            current = 'top'

        # endpoints on the top only is the reverse case to the above.
        elif not bottom:
            current = 'top'
            orientation = '-' if orientation == '+' else '+'
            locs = locations[start][current]
            t = set(locs)
            t.remove(prev_loc)
            current_loc = t.pop()
            cycle += str(current_loc + 1) + orientation
            start = ladder_bottom[current_loc]
            prev_loc = current_loc

            current = 'bottom'
    return cycle


def cycle_to_ladder(cycle_rep):
    """
    Converts a curve pair in cycle notation to the equivalent ladder
    representation.
    :param cycle_rep: The cycle notation of a curve pair.
    :return: The curve pair's ladder representation
    """
    arcs = [int(i) for i in re.split('[-+]', cycle_rep)[:-1]]
    n = len(arcs)
    signs = re.split('[0-9]+', cycle_rep)[1:]
    top = [0 for i in range(len(arcs))]
    bottom = [0 for i in range(len(arcs))]
    ladder = [top, bottom]

    ladder_index = 0
    for i in range(1, len(arcs) + 1):
        current_sign = signs.pop(0)
        current_v = arcs.pop(0)
        if current_sign == '+':
            ladder[0][current_v - 1] = i
            if i == 1:
                ladder[1][current_v - 1] = n  # ((i - 2) % n)
            else:
                ladder[1][current_v - 1] = ((i - 1) % n)

        if current_sign == '-':
            ladder[1][current_v - 1] = i
            if i == 1:
                ladder[0][current_v - 1] = n  # ((i - 2) % n)
            else:
                ladder[0][current_v - 1] = ((i - 1) % n)

    return ladder

