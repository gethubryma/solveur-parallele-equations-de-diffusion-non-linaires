#ifndef GC_GMRES_HH
#define GC_GMRES_HH

#include <vector>

// Méthode du Gradient Conjugué 
std::vector<double> conjugateGradient(const std::vector<std::vector<double>> &A,
                                      const std::vector<double> &b,
                                      std::vector<double> x0,
                                      double tol,
                                      int max_iter);

// Méthode GMRES
std::vector<double> gmres(const std::vector<std::vector<double>> &A,
                          const std::vector<double> &b,
                          std::vector<double> x0,
                          double tol,
                          int max_iter);

#endif
