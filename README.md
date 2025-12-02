# 🌀 Solveur Lorentzien avec Cauchy Loss — IRLS + L-BFGS

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

## 🧠 Utilisation avec un réseau de neurones (PyTorch)

Avec PyTorch, on ne peut pas résoudre les moindres carrés pondérés analytiquement.
On applique donc une boucle interne d’optimisation l’idée IRLS reste identique.

```python
import torch

def cauchy_irls_step(model, X, y, sigma=1.0):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        r = y - y_pred
        w = 1.0 / (sigma**2 + r**2)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):  # mini-optim interne
        optimizer.zero_grad()
        y_pred = model(X)
        loss = (w * (y - y_pred)**2).mean()
        loss.backward()
        optimizer.step()
```

## Avantages

* ✔ Résultats stables
* ✔ Paramètres corrects malgré les outliers
* ✔ Surclasse nettement une régression MSE classique


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
