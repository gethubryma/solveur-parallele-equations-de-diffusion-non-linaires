#include "GC_GMRES.hh"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<vector<double>> A = {
        {4.0, -1.0, 0.0},
        {-1.0, 4.0, -1.0},
        {0.0, -1.0, 4.0}
    };
    vector<double> b = {15.0, 10.0, 10.0};
    vector<double> x0 = {0.0, 0.0, 0.0};

    double tol = 1e-6;
    int max_iter = 100;

    cout << "Matrice A:" << endl;
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[i].size(); j++) {
            cout << A[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

    cout << "Vecteur b:" << endl;
    for (size_t i = 0; i < b.size(); i++) {
        cout << b[i] << " ";
    }
    cout << "\n\n";

    cout << "Solution avec la methode du Gradient Conjugué:" << endl;
    vector<double> sol_cg = conjugateGradient(A, b, x0, tol, max_iter);
    for (double val : sol_cg)
        cout << val << " ";
    cout << "\n\n";

    cout << "Solution avec la methode GMRES:" << endl;
    vector<double> sol_gmrs = gmres(A, b, x0, tol, max_iter);
    for (double val : sol_gmrs)
        cout << val << " ";
    cout << "\n";

    return 0;
}
