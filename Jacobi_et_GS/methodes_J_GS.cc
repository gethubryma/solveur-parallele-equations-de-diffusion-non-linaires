#include "methodes_J_GS.hh"
#include <cmath>
#include <algorithm>

using namespace std;

// Méthode de Jacobi
vector<double> jacobi(const vector<vector<double>> &A,
                      const vector<double> &b,
                      vector<double> x0,
                      double tol,
                      int max_iter)
{
    int n = b.size();
    for (int k = 0; k < max_iter; k++) {
        vector<double> x_new(n, 0.0);
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (j != i)
                    sum += A[i][j] * x0[j];
            }
            x_new[i] = (b[i] - sum) / A[i][i];
        }
   
        
        double erreur = 0.0;
        for (int i = 0; i < n; i++) {
            erreur = max(erreur, fabs(x_new[i] - x0[i]));
        }
        if (erreur < tol)
            return x_new;
        x0 = x_new;
    }
    return x0;
}



// Méthode de Gauss-Seidel

vector<double> gaussSeidel(const vector<vector<double>> &A,
                           const vector<double> &b,
                           vector<double> x0,
                           double tol,
                           int max_iter)
{
    int n = b.size();
    for (int k = 0; k < max_iter; k++) {
        vector<double> x_new = x0;
        for (int i = 0; i < n; i++) {
            double sum1 = 0.0;
            double sum2 = 0.0;

            for (int j = 0; j < i; j++) {
                sum1 += A[i][j] * x_new[j];
            }
            for (int j = i + 1; j < n; j++) {
                sum2 += A[i][j] * x0[j];
            }
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i];
        }
        double erreur = 0.0;
        for (int i = 0; i < n; i++) {
            erreur = max(erreur, fabs(x_new[i] - x0[i]));
        }
        if (erreur < tol)
            return x_new;
        x0 = x_new;
    }
    return x0;
}
