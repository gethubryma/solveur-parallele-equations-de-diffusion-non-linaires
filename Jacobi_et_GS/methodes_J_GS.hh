#ifndef METHODES_J_GS_HH
#define METHODES_J_GS_HH

#include <vector>

// Méthode de Jacobi
std::vector<double> jacobi(const std::vector<std::vector<double>> &A,
                           const std::vector<double> &b,
                           std::vector<double> x0,
                           double tol,
                           int max_iter);

// Méthode de Gauss-Seidel
std::vector<double> gaussSeidel(const std::vector<std::vector<double>> &A,
                                const std::vector<double> &b,
                                std::vector<double> x0,
                                double tol,
                                int max_iter);

#endif
