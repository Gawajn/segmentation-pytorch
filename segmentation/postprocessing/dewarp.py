import logging
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
logger = logging.Logger(__name__)

class NoBaseLineAvailable(Exception):
    pass


def quad_as_rect(quad):
    if quad[0] != quad[2]: return False
    if quad[1] != quad[7]: return False
    if quad[4] != quad[6]: return False
    if quad[3] != quad[5]: return False
    return True

def quad_to_rect(quad):
    assert(len(quad) == 8)
    assert(quad_as_rect(quad))
    return (quad[0], quad[1], quad[4], quad[3])

def rect_to_quad(rect):
    assert(len(rect) == 4)
    return (rect[0], rect[1], rect[0], rect[3], rect[2], rect[3], rect[2], rect[1])

def shape_to_rect(shape):
    assert(len(shape) == 2)
    return (0, 0, shape[0], shape[1])

def griddify(rect, w_div, h_div):
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    x_step = w / float(w_div)
    y_step = h / float(h_div)
    y = rect[1]
    grid_vertex_matrix = []
    for _ in range(h_div + 1):
        grid_vertex_matrix.append([])
        x = rect[0]
        for _ in range(w_div + 1):
            grid_vertex_matrix[-1].append([int(x), int(y)])
            x += x_step
        y += y_step
    grid = np.array(grid_vertex_matrix)
    return grid

def distort_grid(org_grid, max_shift):
    new_grid = np.copy(org_grid)
    x_min = np.min(new_grid[:, :, 0])
    y_min = np.min(new_grid[:, :, 1])
    x_max = np.max(new_grid[:, :, 0])
    y_max = np.max(new_grid[:, :, 1])
    new_grid += np.random.randint(- max_shift, max_shift + 1, new_grid.shape)
    new_grid[:, :, 0] = np.maximum(x_min, new_grid[:, :, 0])
    new_grid[:, :, 1] = np.maximum(y_min, new_grid[:, :, 1])
    new_grid[:, :, 0] = np.minimum(x_max, new_grid[:, :, 0])
    new_grid[:, :, 1] = np.minimum(y_max, new_grid[:, :, 1])
    return new_grid

def grid_to_mesh(src_grid, dst_grid):
    assert(src_grid.shape == dst_grid.shape)
    mesh = []
    for i in range(src_grid.shape[0] - 1):
        for j in range(src_grid.shape[1] - 1):
            src_quad = [src_grid[i    , j    , 0], src_grid[i    , j    , 1],
                        src_grid[i + 1, j    , 0], src_grid[i + 1, j    , 1],
                        src_grid[i + 1, j + 1, 0], src_grid[i + 1, j + 1, 1],
                        src_grid[i    , j + 1, 0], src_grid[i    , j + 1, 1]]
            dst_quad = [dst_grid[i    , j    , 0], dst_grid[i    , j    , 1],
                        dst_grid[i + 1, j    , 0], dst_grid[i + 1, j    , 1],
                        dst_grid[i + 1, j + 1, 0], dst_grid[i + 1, j + 1, 1],
                        dst_grid[i    , j + 1, 0], dst_grid[i    , j + 1, 1]]
            dst_rect = quad_to_rect(dst_quad)
            mesh.append([dst_rect, src_quad])
    return mesh

def transform(point, baselines: List[Tuple[int]]):
    if len(baselines) == 0:
        raise NoBaseLineAvailable

    x, y = point[0], point[1]
    top_base_line_d = 10000000
    top_base_line: Optional[Tuple[int]] = None
    bot_base_line_d = 10000000
    bot_base_line: Optional[Tuple[int]] = None
    for baseline in baselines:
            baseline = np.array(baseline)
            o_y = np.interp(x, baseline[:, 0], baseline[:, 1])
            if o_y < y and y - o_y < top_base_line_d:
                top_base_line = baseline
                top_base_line_d = y - o_y

            if o_y > y and o_y - y < bot_base_line_d:
                bot_base_line = baseline
                bot_base_line_d = o_y - y

    if top_base_line_d > 1000000:
        top_base_line_d = bot_base_line_d
        top_base_line = bot_base_line
    elif bot_base_line_d > 1000000:
        bot_base_line_d = top_base_line_d
        bot_base_line = top_base_line

    if top_base_line is None or bot_base_line is None:
        raise NoBaseLineAvailable

    top_offset = np.mean(top_base_line[:, 1]) - np.interp(x, top_base_line[:, 0], top_base_line[:, 1])
    bot_offset = np.mean(bot_base_line[:, 1]) - np.interp(x, bot_base_line[:, 0], bot_base_line[:, 1])

    interp_y = np.interp(y, [np.interp(x, top_base_line[:, 0], top_base_line[:, 1]),
                             np.interp(x, bot_base_line[:, 0], bot_base_line[:, 1])], [top_offset, bot_offset])

    #return x, np.rint(y - interp_y)
    return x, y - interp_y


def transform_grid(dst_grid, baselines: List[Tuple[int]], shape):
    src_grid = dst_grid.copy()

    for idx in np.ndindex(src_grid.shape[:2]):
        p = src_grid[idx]
        if shape[0] - 1 > p[0] > 0 and shape[1] - 1 > p[1] > 0:
            out = transform(p, baselines)
            src_grid[idx][1] = out[1]

    return src_grid

class Dewarper:
    def __init__(self, shape, baselines: List[Tuple[int]]):
        logger.info("Creating dewarper based on {} staves with shape {}".format(len(baselines), shape))
        self.shape = shape
        self.dst_grid = griddify(shape_to_rect(self.shape), 10, 30)
        print(self.dst_grid)
        print("123")
        logger.debug("Transforming grid)")
        self.src_grid = transform_grid(self.dst_grid, baselines, self.shape)
        print(self.src_grid)

        logger.debug("Creating mesh")
        self.mesh = grid_to_mesh(self.src_grid, self.dst_grid)

    def dewarp(self, images, resamples: List[int] = None):
        if resamples is None:
            resamples = [0] * len(images)

        logger.debug("Transforming images based on mesh")
        out = [im.transform(im.size, Image.MESH, self.mesh, res) for im, res in zip(images, resamples)]
        logger.info("Finished")
        return out

    def inv_transform_point(self, p):
        p = np.asarray(p)
        for i, row in enumerate(self.dst_grid):
            for j, cell in enumerate(row):
                if (cell > p).all():
                    cell_origin = self.dst_grid[i - 1, j - 1]
                    rel = (p - cell_origin) / (cell - cell_origin)

                    target_cell_origin = self.src_grid[i - 1, j - 1]
                    target_cell_extend = self.src_grid[i, j] - target_cell_origin

                    return target_cell_origin + rel * target_cell_extend

        return p

    def inv_transform_points(self, ps):
        return np.array([self.inv_transform_point(p) for p in ps])

    def transform_point(self, p):
        p = np.asarray(p)
        for i, row in enumerate(self.src_grid):
            for j, cell in enumerate(row):
                if (cell > p).all():
                    cell_origin = self.src_grid[i - 1, j - 1]
                    rel = (p - cell_origin) / (cell - cell_origin)

                    target_cell_origin = self.dst_grid[i - 1, j - 1]
                    target_cell_extend = self.dst_grid[i, j] - target_cell_origin

                    return target_cell_origin + rel * target_cell_extend

        return p

    def transform_points(self, ps):
        return np.array([self.transform_point(p) for p in ps])

