/* Fast validation-only driver for our differential verifier. Unlike ParEval's
 * omp-driver.cc (which validates once then runs a timing benchmark), this loops
 * over a THREAD-COUNT SWEEP x REPEATS calling validate() only, in a SINGLE
 * process. It relies on the same externally-defined functions from the ParEval
 * benchmark's cpu.cc (init/validate/reset/destroy). Output: one line per run
 *   RUN t=<threads> r=<rep> valid=PASS|FAIL
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <omp.h>

class Context;
extern "C++" {
    Context *init();
    bool validate(Context *ctx);
    void reset(Context *ctx);
    void destroy(Context *ctx);
}

int main(int argc, char **argv) {
    /* thread counts from argv (comma-separated) or a default sweep */
    int threads[16]; int nt = 0;
    if (argc > 1) {
        char *s = strdup(argv[1]); char *tok = strtok(s, ",");
        while (tok && nt < 16) { threads[nt++] = atoi(tok); tok = strtok(NULL, ","); }
    } else { int d[] = {1,2,4,8,16}; for (int i=0;i<5;i++) threads[nt++]=d[i]; }
    int reps = (argc > 2) ? atoi(argv[2]) : 2;

    for (int i = 0; i < nt; i++) {
        omp_set_num_threads(threads[i]);
        for (int r = 0; r < reps; r++) {
            Context *ctx = init();
            bool ok = validate(ctx);
            printf("RUN t=%d r=%d valid=%s\n", threads[i], r+1, ok ? "PASS" : "FAIL");
            fflush(stdout);
            destroy(ctx);
        }
    }
    return 0;
}
