#include <omp.h>
#include <cstddef>
#include <limits>
#include <vector>

int maxDegree(std::vector<int> const& A, size_t N) {
    if (N == 0) {
        return 0;
    }

    // Guard against invalid input size without risking overflow in N * N.
    if (A.size() / N < N) {
        return 0;
    }

    std::size_t max_deg = 0;

    for (std::size_t i = 0; i < N; ++i) {
        std::size_t degree = 0;
        const std::size_t row_start = i * N;

        for (std::size_t j = 0; j < N; ++j) {
            if (A[row_start + j] != 0) {
                ++degree;
            }
        }

        if (degree > max_deg) {
            max_deg = degree;
        }
    }

    if (max_deg > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }

    return static_cast<int>(max_deg);
}