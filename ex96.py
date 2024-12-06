import numpy as np
import matplotlib.pyplot as plt

def pagerank_iterative(H, max_iter=1000, tol=1e-9):
    N = H.shape[0]
    x = np.ones(N) / N
    ranks_history = [x.copy()]
    for k in range(max_iter):
        x_new = H @ x 
        ranks_history.append(x_new.copy())
        if np.linalg.norm(x_new - x, 1) < tol:
            break
        x = x_new
    return x, ranks_history

H = np.array([[0, 0, 1/3, 0, 0, 0],
    [1/2, 0, 1/3, 0, 0, 0],
    [1/2, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1/2, 1],
    [0, 0, 1/3, 1/2, 0, 0],
    [0, 0, 0, 1/2, 1/2, 0]])
final_ranks, ranks_history = pagerank_iterative(H)

for i in range(len(ranks_history[0])):
    plt.plot(range(len(ranks_history)), [rank[i] for rank in ranks_history], 
             label=f'Page {i+1}')
plt.title('Evolution of PageRank over Iterations')
plt.xlabel('Iteration')
plt.ylabel('PageRank Value')
plt.legend()
plt.grid()
plt.show()