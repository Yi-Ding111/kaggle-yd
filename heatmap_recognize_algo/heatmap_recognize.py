import pandas as pd
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import random
import statistics


from config import cfg
from common import Dimensions
from typing import Literal


def find_consecutive_lengths(arr: np.array) -> list:
    """
    calculate the length of all consecutive subsequences with the same value in a sequence

    parameters:
        arr: the sub array in image pixel matrix

    return:
        returns a list containing the length values of all subsequences
    """
    if len(arr) == 0:
        return []

    consecutive_lengths = []  # used to store the length of all consecutive subsequences
    current_num = arr[0]  # current consecutive numbers
    current_count = 1  # the length of current number

    for i in range(1, len(arr)):
        if arr[i] == current_num:
            current_count += 1  # add 1 if the current number is the same
        else:
            consecutive_lengths.append(current_count)

            # reset
            current_num = arr[i]
            current_count = 1

    consecutive_lengths.append(current_count)

    return consecutive_lengths


def count_with_tolerance(arr: np.array, tolerance: int = 2, filters: int = 10):
    """
    Ignore some values in the list and calculate the value that appears most frequently among the remaining values

    paramaters:
        arr: the array we want to do the calculation
        tolerance: The maximum floating range that can be tolerated. define the range we allow the value to fluctuate.
            e.g.: If the calculated maximum value is 10, and if tolerance = 1, then we also count 9 and 11 as 10.
        filters: if it is less than this value, it will be ignored and not included in the statistics.
    return:
        (int)
    """

    # count the frequence of each number
    counts = Counter(arr)
    # print(counts)

    total_count = 0

    # del Messy statistics, which usually have large counts and affect judgment
    for key in range(0, filters + 1):
        if key in counts:
            del counts[key]

    if len(counts) == 0:
        return 0

    # Find the key corresponding to the maximum value in the remaining statistics
    target = max(counts, key=counts.get)

    # Calculate the total frequency of all numbers within the target and its fluctuation range
    for i in range(target - tolerance, target + tolerance + 1):
        total_count += counts[i]  # If i is not in counts, it returns 0

    return total_count


def add_or_update_dict(ids: dict, key: int, value: int) -> dict:
    """
    add or update key value pairs in the dict
    parameters:
        ids: the dict needs to be added or updated
    """
    if key not in ids.keys():
        ids[key] = [value]

    else:
        ids[key].append(value)

    return ids


def general_dim_infer(
    pixel_mat: np.array,
    random_seq_pick_num=100,
    obj: Literal["row", "height", "col", "width"] = Dimensions.COL,
    filters: int = 10,
    tolerance: int = 2,
):
    """
    inferring #rows and #cols
    parameters:
        tolerance: define the range we allow the value to fluctuate
        pixel_mat: the matrix of the pixel image.
        obj: define which pixel dims, choose from [row / height,col/width]
        filters: if it is less than this value, it will be ignored and not included in the statistics.
        random_seq_pick_num: the number of sequences we want to pick for calculating

    return:
        target: the inferred value for #cols or #rows
        ids_dict: the dict has the info for the row ids or col ids
    """
    # store all counts from all random sequences as list
    total_count_list = []

    # save the sequence ID corresponding to each different answer
    ids_dict = {}

    subseq = 0
    for subseq in range(0, random_seq_pick_num):
        if obj == "row" or obj == "height":
            # create a random number for pixel sequences
            # # Here, 0.25* and -0.25* are used to ignore the surrounding information.
            # We just want to ensure that the randomly selected sequence ID has a greater chance of falling on the image rather than the edge area.
            random_seq_id = random.randint(
                int(0 + 0.25 * pixel_mat.shape[1]),
                int(pixel_mat.shape[1] - 0.25 * pixel_mat.shape[1]),
            )
            sub_arr = pixel_mat.T[random_seq_id]

        else:
            random_seq_id = random.randint(
                int(0 + 0.25 * pixel_mat.shape[0]),
                int(pixel_mat.shape[0] - 0.25 * pixel_mat.shape[0]),
            )
            sub_arr = pixel_mat[random_seq_id]

        # count the most possible rows
        result = count_with_tolerance(
            arr=find_consecutive_lengths(sub_arr), tolerance=tolerance, filters=filters
        )
        total_count_list.append(result)

        # Save the corresponding calculated col or row value and the corresponding sequence id
        ids_dict = add_or_update_dict(ids=ids_dict, key=result, value=random_seq_id)

        subseq += 1

    # find the most possible rows or cols
    counts = Counter(total_count_list)
    target = max(counts, key=counts.get)

    return target, ids_dict


def average_multiple_lists(*lists) -> list:
    """
    calculate the the average value of tuple in lists
    """
    # check if all lists have the same length
    length = len(lists[0])
    if any(len(lst) != length for lst in lists):
        raise ValueError("All lists must have the same length")

    averages = []
    for pairs in zip(*lists):
        avg_pair = tuple(sum(values) / len(values) for values in zip(*pairs))
        averages.append(avg_pair)

    return averages


def round_tuples(vertexs: list) -> list:
    """
    Use list comprehension to iterate over a list and round each element in the tuple
    """
    return [(round(x), round(y)) for x, y in vertexs]


def find_sequence_start_end_pos(
    pixel_mat: np.array,
    ids: dict,
    target: int,
    obj: Literal["row", "height", "col", "width"] = "col",
    tolerance: int = 2,
    filters: int = 10,
) -> list:
    """
    calculate the start and end position of each row or col pixel sequence, return the average value
    if for row, return value represents y-axis value
    if for y, return value represents x-axis value
    parameters:
        pixel_mat: the pixel matrix needs to be calculated
        ids: the dict of sequence ids we want for calculation
        target: the calculated row number of col number in first step
        obj: [row,col,height,width]

    return:
        returns the start or end index pair of each cell's row or column
    """
    # get all sequence ids we want to do calculation here
    target_ids = ids[target]

    cell_length_ls = []
    start_end_index_ls = []

    for id in target_ids:
        if obj == "row" or obj == "height":
            sub_arr = pixel_mat.T[id]

        else:
            sub_arr = pixel_mat[id]

        result = find_consecutive_lengths(sub_arr)

        counts = Counter(result)

        # del Messy statistics, these data usually have large counts and will affect judgment
        for key in range(0, filters + 1):
            if key in counts:
                del counts[key]

        # print(counts)

        if len(counts) == 0:
            return 0, 0, 0

        # Find the key corresponding to the maximum value in the remaining statistics
        cell_length = max(counts, key=counts.get)
        cell_length_ls.append(cell_length)

        # Here we calculate the start and end positions of each array
        first_cell_length = None
        start_position_ls = []
        end_position_ls = []

        for i in range(len(result)):
            if result[i] in [
                x for x in range(cell_length - tolerance, cell_length + tolerance + 1)
            ]:
                first_index = sum(result[:i])
                last_index = sum(result[: i + 1])
                start_position_ls.append(first_index)
                end_position_ls.append(last_index)

                if first_cell_length is None:
                    first_cell_length = result[i]

        start_end_index_tup = list(zip(start_position_ls, end_position_ls))
        if len(start_end_index_tup) == target:
            start_end_index_ls.append(start_end_index_tup)

    result = average_multiple_lists(*start_end_index_ls)

    rounded_result = round_tuples(result)

    return rounded_result


def find_all_vertex(row_vertex: list, col_vertex: list) -> list:
    """
    calculate all cells' vertex coordinates

    parameters:
        row_vertex:the start and end index of all cells in the row direction
        col_vertex:the start and end index of all cells in the col direction
    """
    # Store the vertex positions of all cells
    grid_cells = []

    # Iterate over each row and column separator position
    for y1, y2 in row_vertex:
        for x1, x2 in col_vertex:
            # The vertex position of each cell
            cell_vertices = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
            grid_cells.append(cell_vertices)

    return grid_cells


def refresh_image_pixels_mat(
    row_vertx: list, col_vertex: list, pixel_mat: np.array
) -> np.array:
    """
    Update the data information of each cell according to the provided coordinate information

    parameters:
        pixel_mat: the matrix of the pixel image.
    """
    for y1, y2 in row_vertx:
        for x1, x2 in col_vertex:
            cell_pixels = pixel_mat[y1:y2, x1:x2]

            # Find the most frequent value in the cell
            most_frequent_value = np.argmax(np.bincount(cell_pixels.flatten()))

            # Update all values ​​of the cell to the most frequent value
            cell_pixels[:] = most_frequent_value

            # Update the corresponding area in the original image
            pixel_mat[y1:y2, x1:x2] = cell_pixels

    return pixel_mat


def find_dark_col(
    row_vertex: list, col_vertex: list, dark_value: int, pixel_mat: np.array
) -> list:
    """
    Find the columns where all cells in the grayscale image are dark.
    The specific definition of the value of the dark cell needs to be customized.
    It is worth noting that in the original heatmap, the grayscale values corresponding to overheated or overcooled colors are close to black,
    so the determined columns may be overheated columns or overcooled columns.

    The judgment of these two needs to be made in the subsequent RGB image.

    parameters:
        dark_value: defines the value should be thought as overheat
    """
    # used to store column indexes that meet the conditions
    columns_with_low_avg = []

    # iterate over the cell range of each column
    for col_idx, (x1, x2) in enumerate(col_vertex):
        all_cells_below_threshold = True

        for y1, y2 in row_vertex:
            cell_pixels = pixel_mat[y1:y2, x1:x2]
            cell_mean = cell_pixels.mean()

            # If the mean of a cell is not less than dark value, skip this column
            if cell_mean >= dark_value:
                all_cells_below_threshold = False
                break

        if all_cells_below_threshold:
            columns_with_low_avg.append(col_idx + 1)

    return columns_with_low_avg


def filter_cold_from_warm(
    row_vertex: list,
    col_vertex: list,
    col_ids: list,
    temperature_differ: float,
    image_rgb: np.array,
) -> list:
    """
    distinguish between overcooled and overheated pixel values

    because in gray scale map, what ever overcold or overheat are all represented with low pixel values

    parameters:
        temperature_differ: defines the difference between the red and blue layers in the RGBpixel value
    """

    overheat_col_id = []

    for id in col_ids:

        cell_red_layer_pixel_values = []
        cell_blue_layer_pixel_values = []

        (x1, x2) = col_vertex[id - 1]

        for y1, y2 in row_vertex:
            cell_pixels_red_layer = image_rgb[y1:y2, x1:x2, 0]  # red
            cell_pixels_blue_layer = image_rgb[y1:y2, x1:x2, 2]  # blue layer

            cell_red_layer_pixel_values.append(cell_pixels_red_layer.mean())
            cell_blue_layer_pixel_values.append(cell_pixels_blue_layer.mean())

        # define a formula based on the difference between the red and blue channels,
        # and determine the temperature state based on this difference.
        if (
            np.mean(cell_red_layer_pixel_values) - np.mean(cell_blue_layer_pixel_values)
            > temperature_differ
        ):
            overheat_col_id.append(id)

        else:
            continue

    return overheat_col_id
