#include <vector>
#include <cstddef>
#include <omp.h>

size_t findFirstEven(std::vector<int> const& x) {
    size_t first = x.size();  // Sentinel meaning "not found"

    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(x.size()); ++i) {
        if ((x[static_cast<size_t>(i)] % 2) == 0) {
            // Safely update first with the minimum index found so far

            {
                if (static_cast<size_t>(i) < first) {
                    first = static_cast<size_t>(i);
                }
            }
        }
    }

    return first;
}