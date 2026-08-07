#include <vector>
#include <cstddef>
#include <omp.h>

size_t findFirstEven(std::vector<int> const& x) {
    size_t first = x.size();  // Sentinel meaning "not found"

    #pragma omp parallel for num_threads(1) shared(first)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(x.size()); ++i) {
        if ((x[static_cast<size_t>(i)] % 2) == 0) {
            // Safely update first with the minimum index found so far
            #pragma omp critical
            {
                if (static_cast<size_t>(i) < first) {
                    first = static_cast<size_t>(i);
                }
            }
        }
    }

    return first;
}