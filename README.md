# 🌀 Solveur Lorentzien avec Cauchy Loss — IRLS Robust Optimizer

**Auteur : Thibaut LOMBARD**  
**Licence : MIT**

Ce dépôt contient un solveur robuste basé sur une approximation **quasi-lorentzienne** de la perte de Cauchy, implémenté via un schéma **IRLS — Iteratively Reweighted Least Squares**.

L’objectif est de fournir une alternative robuste aux méthodes classiques (MSE, L-BFGS, Adam), capable de **résister aux outliers** et d’éviter les minima dégénérés liés à l’hypothèse gaussienne des erreurs.

---

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

---

## 🔁 Approche quasi-lorentzienne via IRLS

La perte de Cauchy n’est pas quadratique → difficile à optimiser directement.

On l’approxime localement par une perte quadratique **pondérée** :
```
log(1 + r² / σ²) ≈ w(r) · r² + constante
```
avec le **poids Lorentzien** :
```
w(r) = 1 / (σ² + r²)
```
→ Les points éloignés (outliers) obtiennent un poids faible  
→ Les points fiables guident réellement la descente

---

## 🧮 Algorithme IRLS

Pour des données `{(xi, yi)}` et un modèle `ŷ = fθ(x)` :

1. Initialiser les paramètres `θ`
2. Fixer ou estimer `σ`
3. Répéter jusqu’à convergence :
```
ri = yi - fθ(xi) # résidus
wi = 1 / (σ² + ri²) # poids Lorentziens
Minimiser ∑ wi · (yi - fθ(xi))²
Option : mettre à jour σ via la médiane des résidus
```
✔ Chaque étape est un problème convexe local  
✔ Converge vers un minimum robuste de la perte Cauchy  
✔ Identique à ce qui est utilisé dans Ceres Solver, GTSAM, OpenCV

---

## 🐍 Contenu du dépôt
solver.py # implémentation IRLS + perte de Cauchy

Le script d’exemple montre que la méthode :

- résiste aux outliers
- récupère des paramètres fiables
- surpasse une régression MSE simple

---

## 🧠 Avantages

| Critère | Solveur Lorentzien |
|--------|-------------------|
| Sensible aux outliers | ❌ |
| Convergence stable | ✔️ |
| Compatible deep learning | ✔️ |
| Interprétable (poids = confiance) | ✔️ |

---

## 📦 Installation

```bash
git clone https://github.com/cauchy
python solver.py
```
