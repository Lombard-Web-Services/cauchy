# 🌀 Solveur Lorentzien avec Cauchy Loss — IRLS Robust Optimizer

**Auteur : Thibaut LOMBARD**  
**Licence : MIT**

Ce dépôt contient un solveur robuste basé sur une approximation **quasi-lorentzienne** de la perte de Cauchy, implémenté via un schéma **IRLS — Iteratively Reweighted Least Squares**.

L’objectif est de fournir une alternative robuste aux méthodes classiques (MSE, L-BFGS, Adam), capable de **résister aux outliers** et d’éviter les minima dégénérés liés à l’hypothèse gaussienne des erreurs.



## 🚀 Pourquoi un solveur « Lorentzien » ?

La plupart des optimisations utilisent la perte quadratique :
```
L_MSE = ∑ (y - fθ(x))²
```
Elle explose en présence d’outliers.

La perte **de Cauchy**, issue d’un modèle lorentzien, limite l’influence des grandes erreurs :
```
L_Cauchy = ∑ log(1 + r² / σ²)
```
où `r = y − fθ(x)` et `σ` contrôle l’échelle des résidus.



## 🔁 Approche quasi-lorentzienne via IRLS + L-BFGS

La perte de Cauchy n’est pas quadratique → difficile à minimiser directement.  
On l’approxime itérativement par une perte quadratique **pondérée** :
```
log(1 + r² / σ²) ≈ w(r) · r² + constante
```
avec le **poids Lorentzien** :
```
w(r) = 1 / (σ² + r²)
```
→ Les points éloignés (outliers) obtiennent un poids faible  
→ Les points fiables guident réellement la descente


## 🧮 Boucle IRLS

Pour des données `{(xi, yi)}` et un modèle `ŷ = fθ(x)` :

1. Initialiser les paramètres `θ`
2. Choisir ou estimer `σ`
3. Répéter :
```
ri = yi - fθ(xi) # résidus
wi = 1 / (σ² + ri²) # poids Lorentziens
Minimiser ∑ wi · (yi - fθ(xi))² avec L-BFGS
σ ← median(|ri|) # optionnel : adaptation automatique
```
→ Chaque itération résout un problème localement quadratique  
→ L-BFGS assure une mise à jour stable et précise  
→ Les outliers voient leur poids tendre vers zéro



## 🐍 Exemple minimal (fourni dans le dépôt)

Le solveur ci-dessous effectue une régression linéaire robuste :

```python
import numpy as np
from scipy.optimize import minimize

# Données synthétiques avec outliers
np.random.seed(0)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 2 * X.ravel() + 1 + np.random.normal(0, 0.5, 100)
y[::10] += np.random.normal(0, 10, 10)  # outliers

def irls_cauchy(X, y, max_iter=15, sigma=1.0):
    n = X.shape[1]
    theta = np.random.randn(n + 1)  # [b, w]

    for k in range(max_iter):
        y_pred = X @ theta[1:] + theta[0]
        r = y - y_pred
        w = 1.0 / (sigma**2 + r**2 + 1e-8)

        def weighted_mse(params):
            y_p = X @ params[1:] + params[0]
            return np.sum(w * (y - y_p)**2)

        # Mise à jour par L-BFGS
        theta = minimize(weighted_mse, theta, method='L-BFGS-B').x
        sigma = np.median(np.abs(r)) + 1e-6

    return theta

theta_est = irls_cauchy(X, y)
print("θ estimé (b, w) :", theta_est)```

## 🧠 Avantages

✔ Résultats stables
✔ Paramètres corrects malgré les outliers
✔ Surclasse nettement une régression MSE classique
| Critère                      |  Lorentzien |
| ---------------------------- | :---------: |
| Sensible aux outliers        |      ❌      |
| Mise à jour stable           | ✔️ (L-BFGS) |
| Interprétable (poids)        |      ✔️     |
| Compatible modèles complexes |      ✔️     |
| Adaptation automatique du σ  |      ✔️     |

---

## 📦 Installation

```bash
git clone https://github.com/cauchy
python solver.py
```
