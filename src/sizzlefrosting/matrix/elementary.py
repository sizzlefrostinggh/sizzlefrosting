"""
Elementary module containing reduce row echelon form function and simple matrix elementary operation functions.
"""
import numpy as np

def rowswap(A, source, target):
  """
  Swap two rows of a matrix.

  Args:
      A (array): Matrix to modify.
      source (int): Index of the first row.
      target (int): Index of the second row.

  Returns:
      array: Matrix with the two specified rows swapped.

  Example:
      >>> A = np.array([[1, 2], [3, 4]])
      >>> rowswap(A, 0, 1)
      array([[3, 4],
             [1, 2]])
  """
  A[[source, target]] = A[[target, source]]
  return(A)

def rowscale(A, source, scalar):
  """
  Multiply a row of a matrix by a scalar.

  Args:
      A (array): Matrix to modify.
      source (int): Index of the row to scale.
      scalar (float): Value by which to multiply the row.

  Returns:
      array: Matrix with the specified row scaled.

  Example:
      >>> A = np.array([[1, 2], [3, 4]])
      >>> rowscale(A, 0, 2)
      array([[2, 4],
             [3, 4]])
  """
  A[[source]] = A[[source]] * scalar
  return(A)

def rowreplacement(A, row_i, row_j, j, k):
  """
  Perform a row replacement operation.

  The operation is given by:

      R_j = j * R_i + k * R_j

  Args:
      A (array): Matrix to modify.
      row_i (int): Index of the source row.
      row_j (int): Index of the row being replaced.
      j (float): Scalar applied to the source row.
      k (float): Scalar applied to the target row.

  Returns:
      array: Matrix after the row replacement operation.

  Example:
      >>> A = np.array([[1, 2], [3, 4]])
      >>> rowreplacement(A, 0, 1, -3, 1)
      array([[ 1,  2],
              [ 0, -2]])
  """
  rowscale(A, row_j, k)
  A[row_j] = (j * A[row_i]) + A[row_j]
  return(A)

def initial_swap(A):
  """
  Reorder the rows of a matrix based on the location of their first nonzero entries.

  Args:
      A (array): Matrix to modify.

  Returns:
      array: Matrix with rows reordered according to their pivot points..

  Example:
      >>> A = np.array([[0, 1], [1, 0]])
      >>> initial_swap(A)
      array([[1, 0],
              [0, 1]])
  """
  swapped_rows = 0
  for col in range(len(A[0])):
    for row in range(swapped_rows, len(A)):
      if row != swapped_rows:
        if A[row, col] != 0:
          rowswap(A, col, row)
        swapped_rows += 1
        break
  return(A)

def find_scalar(A, column):
  """
  Find the first nonzero value in a specified column.

  Args:
      A (array): Matrix to search.
      column (int): Index of the column to search.

  Returns:
      float: First nonzero value found in the column 0 if no nonzero numbers are found.

  Example:
      >>> A = np.array([[0, 2], [0, 4]])
      >>> find_scalar(A, 1)
      2
  """

  for i in range(len(A[column:, column])):
    if A[i, column] != 0:
      return(A[row, column])
  return(0)

def rref(A):
  """
  Reduce a matrix to reduced row echelon form (RREF).

  Args:
      A (array): Matrix to reduce.

  Returns:
      array: Matrix in reduced row echelon form.

  Example:
      >>> A = np.array([[1, 2], [3, 4]])
      >>> rref(A)
      array([[1., 0.],
             [0., 1.]])
  """
  A = np.array(A, dtype = float)
  initial_swap(A)
  for row in range(len(A)):
    for col in range(len(A[0])):
      if A[row, col] != 0:
        scalar = A[row, col]
        rowscale(A, row, 1 / scalar)
        for below_row in range(row + 1, len(A)):
          if A[below_row, col] != 0:
            rowreplacement(A, row, below_row, -A[below_row, col], 1)
        break
  for row in range(len(A) -1, -1, -1):
    for col in range(len(A[0])):
      if A[row, col] == 1:
        for above_row in range(row):
          if A[above_row, col] != 0:
            rowreplacement(A, row, above_row, -A[above_row, col], 1)
        break
  return(A + 0.0)
