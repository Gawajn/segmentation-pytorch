import multiprocessing

import numpy as np
import scipy

from segmentation.util import angle_to, PerformanceCounter, logger


class BaseLineCCs(object):
    def __init__(self, cc, type):
        self.cc = cc
        index_min = np.where(cc[1] == min(cc[1]))  # [0]
        index_max = np.where(cc[1] == max(cc[1]))  # [0]

        if type == 'baseline':
            self.cc_left = (np.mean(cc[0][index_min][0]), cc[1][index_min[0]][0])
            self.cc_right = (np.mean(cc[0][index_max][0]), cc[1][index_max[0]][0])

        else:
            self.cc_left = (np.mean(cc[0]), cc[1][index_min[0]][0])
            self.cc_right = (np.mean(cc[0]), cc[1][index_max[0]][0])

        self.type = type

    def __lt__(self, other):
        return self.cc < other


def calculate_distance(index, ccs, maximum_angle, baseline_border_image):
    index1 = []
    index2 = []
    value = []
    x = ccs[index]
    ind1 = index
    for ind2 in range(index, len(ccs)):  # 0
        y = ccs[ind2]
        if x is y:
            index1.append(ind1)
            index2.append(ind2)
            value.append(0)
        else:
            distance = 0
            same_type = 1 if x.type == y.type else 1000

            def left(x, y):
                return x.cc_left[1] > y.cc_right[1]

            def right(x, y):
                return x.cc_right[1] < y.cc_left[1]

            if left(x, y):
                angle = angle_to(np.array(y.cc_right), np.array(x.cc_left))
                distance = x.cc_left[1] - y.cc_right[1]
                test_angle = maximum_angle if distance > 30 else maximum_angle * 3 if distance > 5 else maximum_angle * 5
                if test_angle < angle < (360 - test_angle):
                    distance = 99999
                else:
                    point_c = y.cc_right
                    point_n = x.cc_left

                    x_points = np.arange(start=point_c[1], stop=point_n[1] + 1)
                    y_points = np.interp(x_points, [point_c[1], point_n[1]],
                                         [point_c[0], point_n[0]]).astype(
                        int)
                    indexes = (y_points, x_points)

                    blackness = np.sum(baseline_border_image[indexes])
                    # print('left' + str(blackness))
                    distance = distance * (blackness * 5000 + 1)

            elif right(x, y):
                angle = angle_to(np.array(x.cc_right), np.array(y.cc_left))
                distance = y.cc_left[1] - x.cc_right[1]
                test_angle = maximum_angle if distance > 30 else maximum_angle * 3 if distance > 5 else maximum_angle * 5

                if test_angle < angle < (360 - test_angle):
                    distance = 99999
                else:
                    distance = y.cc_left[1] - x.cc_right[1]

                    point_c = x.cc_right
                    point_n = y.cc_left

                    x_points = np.arange(start=point_c[1], stop=point_n[1] + 1)
                    y_points = np.interp(x_points, [point_c[1], point_n[1]],
                                         [point_c[0], point_n[0]]).astype(
                        int)
                    indexes = (y_points, x_points)

                    blackness = np.sum(baseline_border_image[indexes])
                    distance = distance * (blackness * 5000 + 1)
            else:
                distance = 99999
            index1.append(ind1)
            index2.append(ind2)
            value.append(distance * same_type)
            index1.append(ind2)
            index2.append(ind1)
            value.append(distance * same_type)
            # distance_matrix[ind1, ind2] = distance * same_type
    return (index1, index2), value


def extract_baselines_from_probability_map(image_map: np.array, base_line_index=1, base_line_border_index=2,
                                           original=None, processes=8, min_cc_area=10, max_cc_distance=100):
    image = np.argmax(image_map, axis=-1)
    with PerformanceCounter(function_name="baseline Extraction"):
        baselines = extract_baselines(image_map=image, base_line_index=base_line_index,
                                      base_line_border_index=base_line_border_index, original=original,
                                      processes=processes, min_cc_area=min_cc_area, connection_width=max_cc_distance)
    return baselines


def extracted_ccs_optimized(array: np.array):
    raveled_array = array.ravel()

    index = np.arange(len(raveled_array))
    sort_idx = np.argsort(raveled_array)
    cnt = np.bincount(raveled_array)
    res = np.split(index[sort_idx], np.cumsum(cnt[:-1]))[1:]
    ccs = [np.unravel_index(res[ind], array.shape) for ind, i in enumerate(res)]
    return ccs


def extract_baselines(image_map: np.array, base_line_index=1, base_line_border_index=2, original=None, processes=1,
                      connection_width=100, min_cc_area=10):
    from scipy.ndimage.measurements import label
    with PerformanceCounter(function_name="CCExtraction Line"):

        base_ind = np.where(image_map == base_line_index)
    with PerformanceCounter(function_name="CCExtraction Border"):

        base_border_ind = np.where(image_map == base_line_border_index)

    baseline = np.zeros(image_map.shape)
    baseline_border = np.zeros(image_map.shape)
    baseline[base_ind] = 1
    baseline_border[base_border_ind] = 1
    with PerformanceCounter(function_name="CC extraction"):

        baseline_ccs, n_baseline_ccs = label(baseline, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        baseline_ccs = extracted_ccs_optimized(baseline_ccs)
    baseline_ccs = [BaseLineCCs(x, 'baseline') for x in baseline_ccs if len(x[0]) > min_cc_area]

    all_ccs = baseline_ccs  # + baseline_border_ccs
    logger.info("Extracted {} CCs from probability map \n".format(len(all_ccs)))

    def calculate_distance_matrix(ccs, maximum_angle=5, processes=8):
        distance_matrix = np.zeros((len(ccs), len(ccs)))

        from functools import partial
        distance_func = partial(calculate_distance, ccs=ccs, maximum_angle=maximum_angle,
                                baseline_border_image=baseline_border)
        indexes_ccs = list(range(len(ccs)))
        if processes is not None and processes > 1:
            with multiprocessing.Pool(processes=processes, maxtasksperchild=100) as p:
                out = list(p.map(distance_func, indexes_ccs))
        else:
            out = list(map(distance_func, indexes_ccs))
        for x in out:
            indexes, values = x
            distance_matrix[indexes] = values
        return distance_matrix

    with PerformanceCounter(function_name="calculate_distance_matrix"):
        matrix = calculate_distance_matrix(all_ccs, processes=processes)

    from sklearn.cluster import DBSCAN
    if np.sum(matrix) == 0:
        print("Empty Image")
        return
    with PerformanceCounter(function_name="DBSCan"):
        t = DBSCAN(eps=connection_width, min_samples=1, metric="precomputed").fit(matrix)

    ccs = []
    for x in np.unique(t.labels_):
        ind = np.where(t.labels_ == x)
        line = []
        for d in ind[0]:
            if all_ccs[d].type == 'baseline':
                line.append(all_ccs[d])
        if len(line) > 0:
            ccs.append((np.concatenate([x.cc[0] for x in line]), np.concatenate([x.cc[1] for x in line])))

    ccs = [list(zip(x[0], x[1])) for x in ccs]

    from itertools import chain
    from typing import List, Tuple
    from collections import defaultdict

    def normalize_connected_components(cc_list: List[List[Tuple[int, int]]]):
        # Normalize the CCs (line segments), so that the height of each cc is normalized to one pixel
        def normalize(point_list):
            normalized_cc_list = []
            for cc in point_list:
                cc_dict = defaultdict(list)
                for y, x in cc:
                    cc_dict[x].append(y)
                normalized_cc = []
                for key in sorted(cc_dict.keys()):
                    value = cc_dict[key]
                    normalized_cc.append([int(np.floor(np.mean(value) + 0.5)), key])
                normalized_cc_list.append(normalized_cc)
            return normalized_cc_list

        return normalize(cc_list)

    ccs = normalize_connected_components(ccs)
    new_ccs = []
    for baseline in ccs:
        new_ccs.append([coord_tup[::-1] for coord_tup in baseline])

    return new_ccs
