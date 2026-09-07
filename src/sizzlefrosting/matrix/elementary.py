def rowswap(A, source, target):
  A[[source, target]] = A[[target, source]]
  return(A)

def rowscale(A, source, scalar):
  A[[source]] = A[[source]] * scalar
  return(A)

def rowreplacement(A, row_i, row_j, j, k):
  rowscale(A, row_j, k)
  A[row_j] = (j * A[row_i]) + A[row_j]
  return(A)

def initial_swap(A):
  swapped_rows = 0
  for col in range(len(A[:, 0])):
    for row in range(swapped_rows, len(A[:, 0])):
      if row != swapped_rows:
        if A[row, col] != 0:
          rowswap(A, col, row)
        swapped_rows += 1
  return(A)

def find_scalar(A, column):
  for i in range(len(A[column:, column])):
    if A[i, column] != 0:
      return(A[row, column])
  return(0)

def rref(A):
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
