#include <vector>
#include <omp.h>

bool reduceLogicalXOR(std::vector<bool> const& x) {
    int xor_acc = 0; // use int for OpenMP reduction portability
    const std::size_t n = x.size();

    #pragma omp parallel for reduction(^:xor_acc)
    #pragma omp critical
    for (std::size_t i = 0; i < n; ++i) {
        xor_acc ^= static_cast<int>(x[i]);
    }

    return xor_acc != 0;
}