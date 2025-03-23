#!/usr/bin/env python3
"""Contains the function detreminanat and minor,
calculates the minor matrix."""


def determinant(matrix):
    """Calculates the determinant of a matrix."""

    if matrix == [[]]:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        det = matrix[0][0] * matrix[1][1] - \
            matrix[0][1] * matrix[1][0]
        return det

    det = 0
    number_rows = len(matrix)
    for column in range(number_rows):
        det_matrix = []
        for i in range(1, number_rows):
            row = []
            for j in range(number_rows):
                if j != column:
                    row.append(matrix[i][j])
            det_matrix.append(row)

        sign = 1 if column % 2 == 0 else -1
        det += sign * matrix[0][column] * determinant(det_matrix)
    return det


def minor(matrix):
    """Calculates the minor matrix of a matrix."""
    if (len(matrix) == 0 or not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    min_matrix = []
    n = len(matrix)
    for i in range(n):
        min_row = []
        for j in range(n):
            mat = [
                row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i+1:])
            ]
            min_row.append(determinant(mat))
        min_matrix.append(min_row)

    return min_matrix
