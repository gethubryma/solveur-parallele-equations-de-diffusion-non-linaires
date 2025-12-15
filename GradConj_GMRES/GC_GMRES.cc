#include "GC_GMRES.hh"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;

// Méthode du Gradient Conjugué 

std::vector<double> conjugateGradient(const vector<vector<double>> &A,
                                      const vector<double> &b,
                                      vector<double> x0,
                                      double tol,
                                      int max_iter)
{
    int n = b.size();
    vector<double> r(n), p(n), Ap(n);

    // r0 = b - A*x0
    for (int i = 0; i < n; i++) {
        double Ax = 0.0;
        for (int j = 0; j < n; j++)
            Ax += A[i][j] * x0[j];
        r[i] = b[i] - Ax;
    }
    p = r;

    double r_norm = 0.0;
    for (int i = 0; i < n; i++)
        r_norm += r[i] * r[i];

    for (int k = 0; k < max_iter; k++) {
        // Ap = A*p
        for (int i = 0; i < n; i++) {
            Ap[i] = 0.0;
            for (int j = 0; j < n; j++)
                Ap[i] += A[i][j] * p[j];
        }
        double pAp = 0.0;
        for (int i = 0; i < n; i++)
            pAp += p[i] * Ap[i];

        double alpha = r_norm / pAp;

        for (int i = 0; i < n; i++)
            x0[i] += alpha * p[i];

       
        for (int i = 0; i < n; i++)
            r[i] -= alpha * Ap[i];

        double r_norm_new = 0.0;
        for (int i = 0; i < n; i++)
            r_norm_new += r[i] * r[i];

        //arreter si la norme du résidu est assez petite
        if (sqrt(r_norm_new) < tol)
            break;

        double beta = r_norm_new / r_norm;

        for (int i = 0; i < n; i++)
            p[i] = r[i] + beta * p[i];

        r_norm = r_norm_new;
    }

    return x0;
}

// Méthode GMRES

std::vector<double> gmres(const vector<vector<double>> &A,
                          const vector<double> &b,
                          vector<double> x0,
                          double tol,
                          int max_iter)
{
    int n = b.size();

    vector<vector<double>> Q(max_iter + 1, vector<double>(n, 0.0));
    vector<vector<double>> H(max_iter + 1, vector<double>(max_iter, 0.0));

    vector<double> r0(n, 0.0);
    for (int i = 0; i < n; i++) {
        double Ax = 0.0;
        for (int j = 0; j < n; j++)
            Ax += A[i][j] * x0[j];
        r0[i] = b[i] - Ax;
    }

    double beta = 0.0;
    for (int i = 0; i < n; i++)
        beta += r0[i] * r0[i];
    beta = sqrt(beta);

    if (beta < tol)
        return x0;  


    for (int i = 0; i < n; i++)
        Q[0][i] = r0[i] / beta;

    int k;
    for (k = 0; k < max_iter; k++) {
        vector<double> v(n, 0.0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                v[i] += A[i][j] * Q[k][j];
            }
        }
        for (int j = 0; j <= k; j++) {
            double h = 0.0;
            for (int i = 0; i < n; i++)
                h += Q[j][i] * v[i];
            H[j][k] = h;
            for (int i = 0; i < n; i++)
                v[i] -= h * Q[j][i];
        }
        double h_next = 0.0;
        for (int i = 0; i < n; i++)
            h_next += v[i] * v[i];
        h_next = sqrt(h_next);
        H[k+1][k] = h_next;

        if (h_next < tol) {
            break;
        }

        for (int i = 0; i < n; i++)
            Q[k+1][i] = v[i] / h_next;
    }

    int m = k;  
    vector<double> y(m, 0.0);
    vector<double> g(m+1, 0.0);
    g[0] = beta;

    for (int i = m - 1; i >= 0; i--) {
        double s = g[i];
        for (int j = i + 1; j < m; j++) {
            s -= H[i][j] * y[j];
        }
        if (fabs(H[i][i]) > 1e-14)
            y[i] = s / H[i][i];
        else
            y[i] = 0.0;
    }

    vector<double> x = x0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            x[j] += Q[i][j] * y[i];
        }
    }

    return x;
}
