#include <vector>
#include <array>
#include <cstddef>
#include <omp.h>

struct Point {
    double x, y;
};

/* Count the number of cartesian points in each quadrant. The vector points contains a list of `Point` objects.
   Store the counts in `bins`.
   Use OpenMP to count in parallel.
*/
void countQuadrants(std::vector<Point> const& points, std::array<size_t, 4> &bins) {
    bins = {0, 0, 0, 0};

    {
        std::array<size_t, 4> local = {0, 0, 0, 0};

        for (std::size_t i = 0; i < points.size(); ++i) {
            const double x = points[i].x;
            const double y = points[i].y;

            if (x > 0.0 && y > 0.0) {
                ++local[0]; // Quadrant I
            } else if (x < 0.0 && y > 0.0) {
                ++local[1]; // Quadrant II
            } else if (x < 0.0 && y < 0.0) {
                ++local[2]; // Quadrant III
            } else if (x > 0.0 && y < 0.0) {
                ++local[3]; // Quadrant IV
            }
            // Points on axes are ignored.
        }

        {
            bins[0] += local[0];
            bins[1] += local[1];
            bins[2] += local[2];
            bins[3] += local[3];
        }
    }
}