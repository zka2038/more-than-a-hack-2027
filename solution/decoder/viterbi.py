"""Turn per-frame evidence into one H/A/D sequence with few, well-placed changes.

The decoder is a 3-state hidden Markov model solved exactly with Viterbi. A switch between states
costs a fixed penalty, so the decoder changes possession only when the evidence for the new state,
summed over frames, outweighs that penalty. That is what the ranking metric wants: spellF1 scores
whole possessions, and every invented flicker splits one.
"""

import numpy as np

H, A, D = 0, 1, 2


def viterbi(E, switch):
    """Most likely state path.

    E: (T, 3) per-frame log-scores for H, A, D (higher = more likely).
    switch: (3, 3) cost of moving from state i to state j (0 on the diagonal).
    """
    T = len(E)
    back = np.zeros((T, 3), dtype=np.int8)
    S = [float(x) for x in E[0]]
    c = [[float(switch[i][j]) for j in range(3)] for i in range(3)]
    El = E.tolist()
    for t in range(1, T):
        e = El[t]
        new = [0.0, 0.0, 0.0]
        bt = back[t]
        row = [0, 0, 0]
        for j in range(3):
            best, arg = S[0] - c[0][j], 0
            x = S[1] - c[1][j]
            if x > best:
                best, arg = x, 1
            x = S[2] - c[2][j]
            if x > best:
                best, arg = x, 2
            new[j] = best + e[j]
            row[j] = arg
        bt[:] = row
        S = new
    path = np.empty(T, dtype=np.int8)
    path[-1] = int(np.argmax(S))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
