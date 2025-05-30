import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression

def estimate_lyapunov(ts, emb_dim=5, tau=1, max_t=50):
    n = len(ts) - (emb_dim - 1) * tau
    emb_data = np.array([ts[i:i + n] for i in range(0, emb_dim * tau, tau)]).T
    dists = np.linalg.norm(emb_data[:, None, :] - emb_data[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    nearest_idx = np.argmin(dists, axis=1)
    
    divergence = []
    for t in range(1, max_t):
        dist_sum = 0
        count = 0
        for i in range(n - t):
            j = nearest_idx[i]
            if i + t < n and j + t < n:
                dist = norm(emb_data[i + t] - emb_data[j + t])
                if dist > 0:
                    dist_sum += np.log(dist)
                    count += 1
        if count > 0:
            divergence.append(dist_sum / count)
        else:
            divergence.append(0)
    return np.array(divergence)

def estimate_slope(divergence, steps=20):
    X = np.arange(steps).reshape(-1, 1)
    y = divergence[:steps]
    model = LinearRegression().fit(X, y)
    return model.coef_[0]
