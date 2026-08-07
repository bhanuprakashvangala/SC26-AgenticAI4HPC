#include <vector>
#include <omp.h>

void negateOddsAndHalveEvens(std::vector<int> &x) {
    #pragma omp parallel for default(none) shared(x)
    #pragma omp critical
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] % 2 != 0) {
            x[i] = -x[i];
        } else {
            x[i] /= 2;
        }
    }
}