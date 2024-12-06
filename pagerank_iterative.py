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

