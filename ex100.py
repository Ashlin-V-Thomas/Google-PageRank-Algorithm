import numpy as np

def pagerank_iterative(H, max_iter=1000, tol=1e-9):
    N = H.shape[0]
    # Initialize PageRank vector
    x = np.ones(N) / N
    ranks_history = [x.copy()]

    for k in range(max_iter):
        x_new = H @ x  # Update using the hyperlink matrix
        ranks_history.append(x_new.copy())

        # Check for convergence
        if np.linalg.norm(x_new - x, 1) < tol:
            break
        x = x_new

    return x, ranks_history

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

H = np.array([[0, 0, 1/3, 0, 0, 0],
    [1/2, 0, 1/3, 0, 0, 0],
    [1/2, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1/2, 1],
    [0, 0, 1/3, 1/2, 0, 0],
    [0, 0, 0, 1/2, 1/2, 0]])

final_ranks_iterative, ranks_history = pagerank_iterative(H)
final_ranks_gauelim = pagerank_gauelim(H)

print("Final ranks obtained using the iterative scheme : ")
print(np.round(final_ranks_iterative, 4))
print("Final ranks obtained using the dominant eigenvector method : ")
print(np.round(final_ranks_gauelim, 4))