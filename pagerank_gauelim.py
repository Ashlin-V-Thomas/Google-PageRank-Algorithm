import numpy as np

def row_echelon_form(A): #Gaussian Elimination with partial pivoting
    A = A.astype(float)
    rows, cols = A.shape

    for i in range(min(rows, cols)):
        # Find the pivot row using partial pivoting
        max_row = np.argmax(np.abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]

        # Make the pivot element 1
        if A[i, i] == 0:
            continue
        A[i] = A[i] / A[i, i]

        # Eliminate the column entries below the pivot
        for j in range(i + 1, rows):
            A[j] = A[j] - A[j, i] * A[i]

    return A

def pagerank_gauelim(H):
    N = H.shape[0]
    A = H - np.eye(N)
    A = row_echelon_form(A)
    bs = np.zeros(N)
    xs = np.zeros(N)
    xs[-1] = 1
    for i in reversed(range(N-1)):
        xs[i] = (bs[i] - A[i,i+1:]@xs[i+1:])/A[i,i] 
        # Backward substitution
    return xs/np.sum(xs)
