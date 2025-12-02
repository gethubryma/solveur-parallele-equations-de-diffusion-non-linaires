# Simulation 1D d’un problème de diffusion non linéaire

Ce projet implémente plusieurs schémas numériques pour résoudre un problème 1D de diffusion non linéaire couplée à un terme de rayonnement et une source volumique, avec différentes stratégies de résolution :

* Schéma explicite en temps
* Schéma implicite linéarisé
* Résolution stationnaire par méthode de Newton (solveur tridiagonal maison)
* Résolution stationnaire par méthode de Newton couplée au solveur **HYPRE BoomerAMG** (version parallèle / évolutive)

Le code est écrit en **C++11**, utilise **MPI** et la bibliothèque **HYPRE** pour la résolution de systèmes linéaires de grande taille.

---

## 1. Modèle mathématique

Le modèle correspond à une équation de diffusion non linéaire 1D sur le domaine $x \in [0, 1]$, avec conduction, rayonnement et source localisée.

La forme évolutive (schémas explicite / implicite) est de type :

$$
\frac{\partial u}{\partial t} = \frac{\partial}{\partial x}\left( \kappa(u),\frac{\partial u}{\partial x} \right) -\sigma \left(u^4 - 1\right) + Q(x),
$$

avec :

* $u = u(x,t)$ : température (ou variable scalaire diffusante),
* $\kappa(u) = k_0 u^q$ : conductivité non linéaire,
* $\sigma > 0$ : coefficient de rayonnement,
* $Q(x)$ : terme source volumique, ici localisé près de $x = 0$,
* $k_0 > 0$, $q > 0$ : paramètres de non-linéarité.

Dans le code, la source est définie par :

$$
Q(x) =
\begin{cases}
\beta, & x \le 0.2,\
0, & x > 0.2.
\end{cases}
$$

### 1.1. Conditions aux limites

Les conditions aux limites implémentées sont :

* À gauche $x = 0$ : condition de type Neumann, approx. par une réflexion du point intérieur
  $u_{-1} \approx u_0$ (symétrie → dérivée première nulle).
* À droite $x = 1$ : condition de Dirichlet u(1,t) = 1.

### 1.2. Problème stationnaire

La méthode de Newton cherche à résoudre le problème stationnaire associé :

$$

* \frac{\mathrm{d}}{\mathrm{d}x}\left( \kappa(u),\frac{\mathrm{d}u}{\mathrm{d}x} \right)

+ \sigma \left(u^4 - 1\right) - Q(x)= 0,
  $$

sous les mêmes conditions aux limites. Le résidu non linéaire et le Jacobien tridiagonal correspondant sont construits dans `methodes.cc`.

---

## 2. Discrétisation numérique

### 2.1. Maillage spatial

Le domaine $[0,1]$ est discrétisé de manière uniforme avec $N$ sous-intervalles :

* Nombre de points : $N + 1$
* Pas d’espace :
  $$
  \Delta x = \frac{1}{N}.
  $$

Dans le code :

```cpp
int N = 50;
double dx = 1.0 / N;
```

### 2.2. Schéma explicite en temps

Le schéma explicite est implémenté dans :

```cpp
void explicitStep(std::vector<double>& u, double dt, double dx,
                  double sigma, double k0, double q, double beta);
```

Principe général :

* Avancement temporel avec $u^{n+1} = u^n + \Delta t,F(u^n)$,
* Calcul de la conduction via un flux diffusif avec conductivité moyenne aux interfaces,
* Calcul du terme de rayonnement $\sigma (u^4 - 1)$,
* Ajout de la source $Q(x)$,
* Application des conditions aux limites (Neumann à gauche, Dirichlet à droite).

Le pas de temps $\Delta t$ est choisi à partir d’un critère de stabilité de type CFL :

```cpp
double gamma = 0.1;
double dt = gamma * 2.0 / (4.0 * sigma * std::pow(2.0, 3)
                          + 4.0 * k0 * std::pow(2.0, q) / (dx * dx));
int nsteps = 1000;
```

Le code évolue la solution initiale (ici $u(x,0) = 1$) pendant `nsteps` itérations.

### 2.3. Schéma implicite linéarisé

Le schéma implicite linéarisé est implémenté dans :

```cpp
void implicitStep(std::vector<double>& u, double dt, double dx,
                  double sigma, double k0, double q, double beta);
```

Idée :

* On part d’un schéma implicite $u^{n+1} = u^n + \Delta t,F(u^{n+1})$,
* La non-linéarité est linéarisée autour de l’itéré courant,
* On obtient à chaque pas un système linéaire tridiagonal en $u^{n+1}$,
* Ce système est résolu par l’algorithme de Thomas (`solveTridiag`).

Fonction interne :

```cpp
static void solveTridiag(std::vector<double>& a,
                         const std::vector<double>& b,
                         std::vector<double>& c,
                         std::vector<double>& d);
```

où `a`, `b`, `c` représentent les diagonales de la matrice tridiagonale et `d` le second membre.

### 2.4. Méthode de Newton (stationnaire, solveur tridiagonal)

Pour le problème stationnaire, la méthode de Newton classique est implémentée dans :

```cpp
int newtonSolve(std::vector<double>& u,
                double dx, double sigma, double k0, double q, double beta,
                double tol, int maxIter);
```

Étapes principales par itération :

1. Calcul du résidu non linéaire $F(u)$ :

   ```cpp
   std::vector<double> computeResidualNewton(const std::vector<double>& u,
                                             double dx, double sigma,
                                             double k0, double q, double beta);
   ```

2. Construction du Jacobien tridiagonal $J(u)$ :

   ```cpp
   void computeJacobianNewton(const std::vector<double>& u,
                              double dx, double sigma,
                              double k0, double q, double beta,
                              std::vector<double>& diag,
                              std::vector<double>& upper,
                              std::vector<double>& lower);
   ```

3. Résolution du système linéaire $J(u),\delta u = -F(u)$ par `solveTridiag`.

4. Mise à jour : $u \leftarrow u + \delta u$.

Critère d’arrêt : norme $|\delta u|_\infty < \text{tol}$.

---

## 3. Méthode de Newton avec HYPRE (version parallèle)

La version parallèle de la méthode de Newton utilise HYPRE pour résoudre le système linéaire issu du Jacobien :

```cpp
int newtonSolveHypre(std::vector<double>& u,
                     double dx, double sigma,
                     double k0, double q, double beta,
                     double tol, int maxIter);
```

Principales étapes :

1. Construction du résidu $F(u)$ et des diagonales du Jacobien (comme dans la version séquentielle).

2. Assemblage de la matrice tridiagonale dans un objet `HYPRE_IJMatrix`.

3. Assemblage du vecteur second membre `b_` (valeurs de $-F(u)$).

4. Création d’un vecteur solution `x_` initialisé à zéro.

5. Configuration et appel de **HYPRE BoomerAMG** comme solveur linéaire :

   ```cpp
   HYPRE_BoomerAMGCreate(&solver);

   HYPRE_BoomerAMGSetCoarsenType(solver, 8);
   HYPRE_BoomerAMGSetInterpType(solver, 6);
   HYPRE_BoomerAMGSetPrintLevel(solver, 1);
   HYPRE_BoomerAMGSetRelaxType(solver, 6);

   HYPRE_BoomerAMGSetTol(solver, 1e-14);
   HYPRE_BoomerAMGSetMaxIter(solver, 50);

   HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
   ```

6. Récupération de la correction `delta` et mise à jour de la solution $u$.

Cette approche permet de traiter des tailles de problèmes beaucoup plus grandes en s’appuyant sur un solveur multigrille robuste, potentiellement en parallèle via MPI.

---

## 4. Structure du projet

Arborescence typique :

* `Makefile`
* `main.cc`
* `methodes.cc`
* `methodes.hh`
* `hypre/` (dossier contenant la bibliothèque HYPRE compilée, par exemple `hypre/src/hypre/...`)

### 4.1. `main.cc`

* Définition des paramètres numériques : `N`, `dx`, `sigma`, `k0`, `q`, `beta`, `dt`, `nsteps`.
* Construction de trois solutions :

  * `u_explicit` : schéma explicite en temps,
  * `u_implicit` : schéma implicite linéarisé,
  * `u_newton` : solution stationnaire par Newton séquentiel.
* Affichage de messages de fin de calcul pour chaque méthode.
* Possibilité de tracer les profils via `printSolution` (commenté par défaut).

### 4.2. `methodes.hh`

Fichier d’en-tête déclarant toutes les fonctions numériques :

```cpp
double computeKappa(double u, double k0, double q);

void explicitStep(std::vector<double>& u,
                  double dt, double dx,
                  double sigma, double k0, double q,
                  double beta);

void implicitStep(std::vector<double>& u,
                  double dt, double dx,
                  double sigma, double k0, double q,
                  double beta);

int newtonSolve(std::vector<double>& u,
                double dx, double sigma,
                double k0, double q, double beta,
                double tol, int maxIter);

std::vector<double> computeResidualNewton(const std::vector<double>& u,
                                          double dx, double sigma,
                                          double k0, double q, double beta);

void computeJacobianNewton(const std::vector<double>& u,
                           double dx, double sigma,
                           double k0, double q, double beta,
                           std::vector<double>& diag,
                           std::vector<double>& upper,
                           std::vector<double>& lower);

int newtonSolveHypre(std::vector<double>& u,
                     double dx, double sigma,
                     double k0, double q, double beta,
                     double tol, int maxIter);
```

### 4.3. `methodes.cc`

Implémentation de l’ensemble des méthodes numériques :

* `computeKappa` : calcul de $\kappa(u) = k_0 u^q$,
* `explicitStep` : schéma explicite,
* `implicitStep` + `solveTridiag` : schéma implicite linéarisé et solveur tridiagonal,
* `computeResidualNewton`, `computeJacobianNewton` : résidu et Jacobien pour Newton,
* `newtonSolve` : Newton séquentiel avec tridiagonal,
* `newtonSolveHypre` : Newton avec solveur HYPRE BoomerAMG.

---

## 5. Dépendances et environnement

### 5.1. Compilateur et MPI

* Compilateur C++ compatible C++11 (par exemple `g++` via `mpicxx`),
* Bibliothèque **MPI** installée (`mpicxx` disponible).

Dans le `Makefile` :

```makefile
CXX = mpicxx
CXXFLAGS = -O3 -Wall -std=c++11
```

### 5.2. Bibliothèque HYPRE

Le projet suppose une installation locale de HYPRE accessible via :

```makefile
HYPRE_DIR = $(PWD)/hypre/src/hypre

CXXFLAGS += -I$(HYPRE_DIR)/include
LDFLAGS  += -L$(HYPRE_DIR)/lib -lHYPRE
```

Il faut donc :

1. Télécharger HYPRE,
2. Le configurer et le compiler,
3. Vérifier que `include` et `lib` se trouvent bien dans `$(PWD)/hypre/src/hypre/`.

---

## 6. Compilation

À la racine du projet, utiliser :

```bash
make
```

Le `Makefile` va :

* Compiler `methodes.cc` en `methodes.o`,
* Compiler `main.cc` en `main.o`,
* Lier le tout avec HYPRE pour produire l’exécutable :

```makefile
EXEC = diffusion
```

Pour nettoyer les fichiers objets et l’exécutable :

```bash
make clean
```

---

## 7. Exécution

Une fois compilé, exécuter simplement :

```bash
./diffusion
```

ou, si l’on souhaite explicitement passer par MPI (même en mono-processus) :

```bash
mpirun -np 1 ./diffusion
```

Le programme affiche :

* Un message pour la fin du calcul explicite,
* Un message pour la fin du calcul implicite linéarisé,
* Le nombre d’itérations de Newton et la norme de la correction, par exemple :

```text
Newton converge en XX itérations, ||delta||inf=...
```

Les appels à `printSolution` sont commentés dans `main.cc`. Pour visualiser le profil final $u(x)$ d’une méthode, il suffit de décommenter :

```cpp
// printSolution(u_explicit, dx);
// printSolution(u_implicit, dx);
// printSolution(u_newton, dx);
```

et de rediriger la sortie vers un fichier, par exemple :

```bash
./diffusion > solution.dat
```

Ce fichier peut ensuite être tracé avec gnuplot, Python, etc.

---

## 8. Paramètres numériques et ajustements

Les paramètres se trouvent dans `main.cc` :

```cpp
int N = 50;             // nombre de sous-intervalles (N+1 points)
double dx = 1.0 / N;    // pas d'espace

double sigma = 0.1;
double k0    = 0.01;
double q     = 1.0;
double beta  = 1.0;

double gamma = 0.1;     // facteur de sécurité CFL
double dt    = ...;     // calculé à partir de gamma, sigma, k0, q, dx
int nsteps   = 1000;    // nombre de pas de temps

double tol     = 1.0e-10; // tolérance Newton
int maxIter    = 50;      // nombre max d'itérations Newton
```

Pour modifier :

* La résolution spatiale, changer `N`,
* La durée de la simulation temporelle, ajuster `nsteps` ou `dt`,
* La non-linéarité, modifier `k0` et `q`,
* La source locale, changer `beta` ou le seuil `x <= 0.2`,
* La précision de Newton, modifier `tol` et `maxIter`.

---

## 9. Utilisation de la version HYPRE de Newton

Par défaut, `main.cc` utilise la version séquentielle de Newton :

```cpp
newtonSolve(u_newton, dx, sigma, k0, q, beta, tol, maxIter);
```

Pour tester la version HYPRE, il est possible d’appeler à la place :

```cpp
newtonSolveHypre(u_newton, dx, sigma, k0, q, beta, tol, maxIter);
```

Cela permet de bénéficier de la robustesse et de l’évolutivité du solveur multigrille BoomerAMG pour des tailles de problèmes plus importantes et/ou en contexte parallèle.

---

## 10. Résumé

Ce projet fournit un cadre complet pour :

* Expérimenter différents schémas temporels (explicite / implicite) pour une équation de diffusion non linéaire avec rayonnement,
* Étudier la convergence vers l’état stationnaire via une méthode de Newton,
* Comparer une résolution séquentielle tridiagonale et une résolution via un solveur multigrille parallèle (HYPRE).
