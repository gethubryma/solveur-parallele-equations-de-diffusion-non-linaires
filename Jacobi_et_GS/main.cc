#include "methodes_J_GS.hh"
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

    cout << "Solution avec la méthode de Jacobi:" << endl;
    vector<double> sol_jacobi = jacobi(A, b, x0, tol, max_iter);
    for (size_t i = 0; i < sol_jacobi.size(); i++) {
        cout << sol_jacobi[i] << " ";
    }
    cout << "\n\n";

    cout << "Solution avec la méthode de Gauss-Seidel:" << endl;
    vector<double> sol_gs = gaussSeidel(A, b, x0, tol, max_iter);
    for (size_t i = 0; i < sol_gs.size(); i++) {
        cout << sol_gs[i] << " ";
    }
    cout << "\n";

    return 0;
}
