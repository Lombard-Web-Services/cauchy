import numpy as np
from scipy.optimize import minimize

# Données synthétiques avec outliers
np.random.seed(0)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 2 * X.ravel() + 1 + np.random.normal(0, 0.5, 100)
# Ajout d'outliers
y[::10] += np.random.normal(0, 10, 10)

def irls_cauchy(X, y, max_iter=15, sigma=1.0):
    n = X.shape[1]
    theta = np.random.randn(n + 1)  # [b, w]

    for k in range(max_iter):
        # Prédiction et résidus
        y_pred = X @ theta[1:] + theta[0]
        r = y - y_pred

        # Poids lorentziens
        w = 1.0 / (sigma**2 + r**2 + 1e-8)

        # Perte quadratique pondérée
        def weighted_mse(params):
            y_p = X @ params[1:] + params[0]
            return np.sum(w * (y - y_p)**2)

        # Mise à jour avec L-BFGS
        res = minimize(weighted_mse, theta, method='L-BFGS-B')
        theta = res.x

        # Option : mise à jour adaptative de sigma
        sigma = np.median(np.abs(r)) + 1e-6

    return theta

theta_est = irls_cauchy(X, y)
print("θ estimé (b, w) :", theta_est)
