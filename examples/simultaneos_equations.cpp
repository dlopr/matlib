#include <iostream>
#include "matlib.h"

//========================================================================
// This example demonstrates solving the following simultaneous equations
// 2 * x1 + 8 * x2 + 5 * x3 = 5,
// 1 * x1 + 1 * x2 + 1 * x3 = -2,
// 1 * x1 + 2 * x2 - 1 * x3 = 2.
//========================================================================

template <size_t N>
void test() {
    Matrix<N, N, double> A;
    Vector<N, double> x;
    Vector<N, double> b;

    // Write in Ax = b form
    A = 2, 8, 5,
        1, 1, 1,
        1, 2, -1;
    A += 1;
    A += -1;

    b = 5, -2, 2;

    // solve for x
    x = inverse(A) * b;

    std::cout << "\nA:\n" << A << std::endl;
    std::cout << "\nb: " << transpose(b) << std::endl;
    std::cout << "\ninv_A:\n" << inverse(A) << std::endl;
    std::cout << "\nA * inv_A:\n" << A * inverse(A) << std::endl;
    std::cout << "\ninv_A * A:\n" << inverse(A) * A << std::endl;
    std::cout << "\nsolution: " << transpose(x) << std::endl;
}

int main() {
    test<1>();
    test<2>();
    test<3>();
    return 0;
}
