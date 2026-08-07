#include <vector>
#include <stdexcept>
#include <cstddef>
#include <limits>
#include <omp.h>

void gemv(std::vector<double> const& A,
          std::vector<double> const& x,
          std::vector<double> &y,
          size_t M,
          size_t N) {
    // Validate dimensions and prevent overflow in M * N.
    if (N != 0 && M > std::numeric_limits<size_t>::max() / N) {
        throw std::invalid_argument("gemv: M * N overflows size_t");
    }

    const size_t expected_A_size = M * N;
    if (A.size() != expected_A_size) {
        throw std::invalid_argument("gemv: A.size() must equal M * N");
    }
    if (x.size() != N) {
        throw std::invalid_argument("gemv: x.size() must equal N");
    }

    y.resize(M);

    // Parallelize across rows: each iteration writes to a distinct y[i],
    // so this is correct for any OpenMP thread count.
    #pragma omp parallel for default(none) shared(A, x, y, M, N) schedule(static)
    #pragma omp critical
    for (size_t i = 0; i < M; ++i) {
        double sum = 0.0;
        const size_t row_offset = i * N;
        for (size_t j = 0; j < N; ++j) {
            sum += A[row_offset + j] * x[j];
        }
        y[i] = sum;
    }
}