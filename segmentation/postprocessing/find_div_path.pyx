import heapq
from collections import namedtuple, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import numpy as np

from segmentation.postprocessing.debug_draw import DebugDraw
from segmentation.preprocessing.source_image import SourceImage
from segmentation.util import logger


QueueElem = namedtuple("QueueElem", "f d point parent")
"""
cdef class QueueElem:
    cpdef int f
    cpdef int d
    cpdef object point
    cpdef object parent

    def __lt__(self, QueueElem other):
        return self.f < other.f
"""
# f is dist + heur, d is distance, n is node, p is parent

def extend_baselines(line_a: List, line_b: List) -> Tuple[List,List]:
    start_dif = line_b[0][0] - line_a[0][0]
    end_swap = False
    if start_dif < 0:
        start_dif = -start_dif
        end_swap = not end_swap
        line_a, line_b = line_b, line_a

    if start_dif > 0: # line_a is longer, extend line_b
        y = line_b[0][1]
        x_start = line_b[0][0] - start_dif
        line_b = [(x_start + i ,y) for i in range(start_dif)] + line_b



    end_dif = line_b[-1][0] - line_a[-1][0]
    if end_dif < 0:
        end_dif = -end_dif
        end_swap = not end_swap
        line_a, line_b = line_b, line_a

    if end_dif > 0: # line_b is longer, extend line a
        end_y = line_a[-1][1]
        end_start_x = line_a[-1][0]
        line_a = line_a + [(end_start_x + i + 1, end_y) for i in range(end_dif)]

    if end_swap:
        line_a, line_b = line_b, line_a

    if not (line_a[0][0] == line_b[0][0] and line_a[-1][0] == line_b[-1][0]):
        line_a_r = list(reversed(line_a))
        line_b_r = list(reversed(line_b))
        raise RuntimeError("oops")
    return line_a, line_b

def make_path(target_node:QueueElem, start_node: QueueElem) -> List:
    path_reversed = []
    node = target_node
    while node != start_node:
        #print(node.f, node.d, node.point)
        path_reversed.append(node.point)
        # if we are on home position, no need to move vertically
        if node.point[0] == start_node.point[0] + 1:
            break
        node = node.parent
    return list(reversed(path_reversed))


class DividingPathStartingBias(Enum):
    MID = 0
    TOP = 1
    BOTTOM = 2
    OFFSET = 3

@dataclass
class DividingPathStartConditions:
    starting_bias: DividingPathStartingBias = DividingPathStartingBias.MID
    starting_offset: int = None
    starting_cheaper_y: bool = True


def find_dividing_path_old(inv_binary_img: np.ndarray, cut_above, cut_below, starting_bias = DividingPathStartingBias.MID, start_conditions: DividingPathStartConditions = None) -> List:
    # assert, that both cut_baseline is a list of lists and cut_topline is also a list of lists
    tl, bl = extend_baselines(cut_above, cut_below)

    # deep copy the lines to avoid breaking stuff with the corridoor adjustment
    tl = [(p[0],p[1]) for p in tl]
    bl = [(p[0], p[1]) for p in bl]


    # see if there is a corridor, if not, push top baseline up by 1 px
    min_channel = 3
    for x, points in enumerate(zip(tl,bl)):
        p1, p2 = points
        if p1[1] > p2[1]:
            tl[x], bl[x] = bl[x],tl[x]
            p1, p2 = tl[x], bl[x]

        if abs(p1[1] - p2[1]) <= min_channel:
            if tl[x][1] > min_channel:
                tl[x] = (tl[x][0], tl[x][1] - min_channel)
            elif bl[x][1] + min_channel < inv_binary_img.shape[0]:
                bl[x] = (bl[x][0], bl[x][1] + min_channel)
            else:
                assert False, "Unlucky"

    """
    def find_children(cur_node, x_start):
        x = cur_node[0]
        xi = x - x_start
        y1, y2 = tl[xi][1], bl[xi][1] # TODO: shouldn't this be +1 ?
        if y1 > y2: y1, y2 = y2, y1
        #for y in range(max(tl[xi][1], cur_node[1] - 1), min(cur_node[1]+2, bl[xi][1] + 1)):
        for y in range(y1,y2+1):
            yield (x + 1,y)
    """
    tly_arr = np.array([x[1] for x in tl], dtype=np.int32)
    bly_arr = np.array([x[1] for x in bl], dtype=np.int32)

    cdef int[:] tly = tly_arr
    cdef int[:] bly = bly_arr

    def find_children_rect(cur_node, int x_start):
        out = [] # using a list here is probably faster than a generator
        cdef int x = cur_node[0]
        cdef int y = cur_node[1]
        cdef int xi = x - x_start
        cdef int y1 = tly[xi]
        cdef int y2 = bly[xi]  # TODO: shouldn't this be +1 ?
        # if y1 > y2: y1, y2 = y2, y1
        # if tl[xi+1][1] <= y <= bl[xi+1][1]: yield (x+1,y)
        if tly[xi + 1] <= y <= bly[xi + 1]: out.append((x + 1, y))
        if y > y1: out.append ((x, y-1))
        if y < y2: out.append( (x, y+1))
        return out

    # use Dijkstra's algorithm to find the shortest dividing path
    # dummy start point

    end_x = int(bl[-1][0])
    assert end_x == int(tl[-1][0]) and end_x == int(bl[-1][0])

    # adjust the constant factor for different cutting behaviours
    def dist_fn(p1,p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + int(inv_binary_img[p2[1],p2[0]]) * 1000
        # return 1+ abs( p2[1] - p1[1]) + int(inv_binary_img[p2[1],p2[0]]) * 1000
        # bottom one is ~ 4% faster

    def cheaper_y_dist_fn(p1,p2):
        return abs(p1[0] - p2[0]) + 0.5 * abs(p1[1] - p2[1]) + int(inv_binary_img[p2[1], p2[0]]) * 1000


    def H_fn(p):
        return end_x - p[0]
    #H_fn = lambda p: end_x - p[0]
    #nodeset = dict()
    if starting_bias == DividingPathStartingBias.MID:
        start_point = (bl[0][0] - 1, int(abs(bl[0][1] + tl[0][1]) / 2))
    elif starting_bias == DividingPathStartingBias.TOP:
        start_point = (bl[0][0] - 1, tl[0][1] + 1)
    elif starting_bias == DividingPathStartingBias.BOTTOM:
        start_point = (bl[0][0] - 1, bl[0][1] - 1)
    else:
        raise NotImplementedError()
    start_x = bl[0][0] # start_point[0] + 1

    y1, y2 = tl[0][1], bl[0][1]+ 1
    if y1 > y2: y1, y2 = y2,y1
    source_points = [(start_x, y) for y in range(y1, y2)]

    start_elem = QueueElem(0, 0, start_point, None)
    Q: List[QueueElem] = []

    if start_conditions and start_conditions.starting_cheaper_y:
        for p in source_points:
            heapq.heappush(Q,QueueElem(d=cheaper_y_dist_fn(start_point, p), f=cheaper_y_dist_fn(start_point,p) + H_fn(p),point=p,parent=start_elem))
    else: # default behaviour
        for p in source_points:
            heapq.heappush(Q,QueueElem(d=dist_fn(start_point, p), f=dist_fn(start_point,p) + H_fn(p),point=p,parent=start_elem))

    #distance = defaultdict(lambda: 2147483647)
    visited = set()
    shortest_found_dist = defaultdict(lambda: 2147483647)

    cdef float [:, :] inv_bin_view = inv_binary_img

    cdef int p10
    cdef int p11
    cdef int p20
    cdef int p21
    cdef int d

    for elem in Q:
        shortest_found_dist[elem.point] = elem.d
    while len(Q) > 0:
        node = heapq.heappop(Q)
        if node.point in visited: continue # if we already visited this node
        visited.add(node.point)
        if node.point[0] == end_x:
            path = make_path(node, start_elem)
            if True:
                dd = DebugDraw(SourceImage.from_numpy(np.array(255*(1 - inv_binary_img),dtype=np.uint8)))
                dd.draw_baselines([tl, bl, path])
                img = dd.image()
                #global seq_number
                #img.save(f"/tmp/seq{seq_number}.png")
                #seq_number += 1
                #dd.show()
            return path
        for child in find_children_rect(node.point, start_x):
            if child in visited: continue
            # calculate distance and heur
            p1, p2 = node.point, child
            # not having to call a function is ~10 % faster
            p10 = p1[0]
            p11 = p1[1]
            p20 = p2[0]
            p21 = p2[1]

            d = <int> node.d + abs(p10 - p20) + abs(p11 - p21) + <int>(inv_bin_view[p21, p20]) * 1000
            #d = dist_fn(node.point,child) + node.d

            if shortest_found_dist[child] <= d:
                continue  # we already found it

            # is this path to this node shorter?
            shortest_found_dist[child] = d

            #h = H_fn(child)
            h = end_x - child[0]

            heapq.heappush(Q, QueueElem(f=h+d, d=d, point=child, parent=node))
    logger.error("Cannot run A*")
    logger.error("Cut above: {}".format(cut_above))
    logger.error("Cut below: {}".format(cut_below))

    # raise RuntimeError("Unreachable")
    # Just use the middle line, to avoid crashing
    logger.error("Using fill path to avoid crashing")
    fill_path = []
    for pa, pb in zip(cut_above, cut_below):
        fill_path.append((pa[0],round((pa[1] + pb[1]) / 2)))
    return fill_path

def fallback_fill_path(cut_above, cut_below):
    logger.error("Using fill path to avoid crashing")
    fill_path = []
    for pa, pb in zip(cut_above, cut_below):
        fill_path.append((pa[0],round((pa[1] + pb[1]) / 2)))
    return fill_path




cdef double ysum(float[:,:] img, int start_x, int ystart, int ytarget):
    y1 = min(ystart, ytarget)
    y2 = max(ystart, ytarget)

    if y1 > y2:
        y1,y2 = y2,y1
    cdef int i
    cdef double s1 = 0.0
    for i in range(y1,y2+1):
        s1 += img[i, start_x]
    s1 += img[ytarget, start_x + 1]

    s1 *= 100.0 # weighting of black pixels

    # add the cost of jumping y steps
    return s1 + (y2 - y1)


def shortest_path_dag(inv_binary_img: np.ndarray, tl: List, bl: List, start_x, start_y: int, start_cheaper_y: bool) -> List:
    tly_arr = np.array([u[1] for u in tl], dtype=np.int32)
    bly_arr = np.array([u[1] for u in bl], dtype=np.int32)
    cdef float[:,:] img_arr = inv_binary_img
    cdef int[:] tly = tly_arr
    cdef int[:] bly = bly_arr

    # build the DAG
    # we need len(tly + 2) x-coords for the DAG and max_dist(tly, bly) y-coords
    cdef int dag_width = tly_arr.shape[0] + 1
    cdef int dag_height = 0
    cdef int i
    for i in range(tly_arr.shape[0]):
        if abs(tly[i] - bly[i]) > dag_height:
            dag_height = abs(tly[i] - bly[i])

    # create the array
    dag_arr = np.full(shape=(dag_height, dag_width), fill_value=np.finfo(np.double).max, dtype=np.double)

    cdef double[:,:] dag = dag_arr
    cdef int line_startx = tl[0][0]
    # we do not care about cost in x, because in every step we MUST move 1 step to the right, so x-cost will always be 2*w

    node_from_arr = np.zeros(shape=(dag_height,dag_width), dtype=np.int32)
    cdef int[:,:] node_from = node_from_arr

    # run the shortest path algorithm
    # do the first step manually, because we have 2 different types of implementation
    cdef int y = 0
    if start_cheaper_y:
        for y in range(dag_height):
            dag[y,0] = <double> (0.5 * abs(start_y - y + tly[0]))
    else:
        for y in range(dag_height):
            dag[y,0] = <double> abs(start_y - y + tly[0])
    cdef int x

    cdef int y_to
    cdef int y_from
    cdef double dist
    for x in range(1, dag_width):
        for y_from in range(0, dag_height):
            for y_to in range(0, dag_height):
                dist = ysum(img_arr, x + line_startx - 1,tly[x-1] + y_from, tly[x-1] + y_to) + dag[y_from, x-1]
                if dist < dag[y_to,x]:
                    dag[y_to,x] = dist
                    node_from[y_to,x] = y_from

    y_coords = []
    y_coords.append(min(node_from[-1]))
    last_from = y_coords[0]
    for xi in range(dag_width-1,0,-1):
        this_from = node_from[last_from,xi]
        y_coords.append(this_from + tly[xi -1])
        last_from = this_from

    # return a list of points
    points = []
    for xi, yi in zip(range(start_x, start_x + dag_width -1), reversed(y_coords)):
        points.append((xi,yi))
    print(points)
    dd = DebugDraw(SourceImage.from_numpy(np.array(255 * (1 - inv_binary_img), dtype=np.uint8)))
    dd.draw_baselines([tl, bl, points])
#    dd.show()
    return points






def find_dividing_path_dag(inv_binary_img: np.ndarray, cut_above, cut_below, starting_bias = DividingPathStartingBias.MID, start_conditions: DividingPathStartConditions = DividingPathStartConditions()) -> List:
    #assert inv_binary_img.dtype == np.float32, "Inv Binary img has wrong dtype. Required: float32"
    inv_binary_img = np.array(inv_binary_img, dtype=np.float32)
    # assert, that both cut_baseline is a list of lists and cut_topline is also a list of lists
    tl, bl = extend_baselines(cut_above, cut_below)

    # deep copy the lines to avoid breaking stuff with the corridoor adjustment
    tl = [(p[0],p[1]) for p in tl]
    bl = [(p[0], p[1]) for p in bl]


    # see if there is a corridor, if not, push top baseline up by 1 px
    min_channel = 3
    for x, points in enumerate(zip(tl,bl)):
        p1, p2 = points
        if p1[1] > p2[1]:
            tl[x], bl[x] = bl[x],tl[x]
            p1, p2 = tl[x], bl[x]

        if abs(p1[1] - p2[1]) <= min_channel:
            if tl[x][1] > min_channel:
                tl[x] = (tl[x][0], tl[x][1] - min_channel)
            elif bl[x][1] + min_channel < inv_binary_img.shape[0]:
                bl[x] = (bl[x][0], bl[x][1] + min_channel)
            else:
                assert False, "Unlucky"

    tly_arr = np.array([x[1] for x in tl], dtype=np.int32)
    bly_arr = np.array([x[1] for x in bl], dtype=np.int32)

    cdef int[:] tly = tly_arr
    cdef int[:] bly = bly_arr

    def find_children_rect(cur_node, int x_start):
        cdef list out = [] # using a list here is probably faster than a generator
        cdef int x = cur_node[0]
        cdef int y = cur_node[1]
        cdef int xi = x - x_start
        cdef int y1 = tly[xi]
        cdef int y2 = bly[xi]  # TODO: shouldn't this be +1 ?
        # if y1 > y2: y1, y2 = y2, y1
        # if tl[xi+1][1] <= y <= bl[xi+1][1]: yield (x+1,y)
        if tly[xi + 1] <= y <= bly[xi + 1]: out.append((x + 1, y))
        if y > y1: out.append ((x, y-1))
        if y < y2: out.append( (x, y+1))
        return out

    # use Dijkstra's algorithm to find the shortest dividing path
    # dummy start point

    end_x = int(bl[-1][0])
    assert end_x == int(tl[-1][0]) and end_x == int(bl[-1][0])

    # adjust the constant factor for different cutting behaviours
    def dist_fn(p1,p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + int(inv_binary_img[p2[1],p2[0]]) * 1000
        # return 1+ abs( p2[1] - p1[1]) + int(inv_binary_img[p2[1],p2[0]]) * 1000
        # bottom one is ~ 4% faster

    def cheaper_y_dist_fn(p1,p2):
        return abs(p1[0] - p2[0]) + 0.5 * abs(p1[1] - p2[1]) + int(inv_binary_img[p2[1], p2[0]]) * 1000


    #H_fn = lambda p: end_x - p[0]
    #nodeset = dict()
    if starting_bias == DividingPathStartingBias.MID:
        start_point = (bl[0][0] - 1, int(abs(bl[0][1] + tl[0][1]) / 2))
    elif starting_bias == DividingPathStartingBias.TOP:
        start_point = (bl[0][0] - 1, tl[0][1] + 1)
    elif starting_bias == DividingPathStartingBias.BOTTOM:
        start_point = (bl[0][0] - 1, bl[0][1] - 1)
    else:
        raise NotImplementedError()
    start_x = bl[0][0] # start_point[0] + 1

    return shortest_path_dag(inv_binary_img,tl,bl,start_x,start_point[1],start_conditions.starting_cheaper_y)


