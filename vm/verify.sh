#!/bin/bash
echo "=== toolchain presence ==="
for t in gcc g++ gfortran mpicc mpirun nvcc; do
  p=$(which $t 2>/dev/null)
  echo "$t: ${p:-MISSING}"
done
echo "gcc_ver=$(gcc -dumpversion 2>/dev/null)"
echo "nvcc_ver=$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' || echo none)"
echo "nproc=$(nproc)"

echo "=== OpenMP compile+run ==="
cat > /tmp/omp.c <<'EOF'
#include <stdio.h>
#include <omp.h>
int main(){ int n=0;
  #pragma omp parallel reduction(+:n)
  { n += 1; }
  printf("threads_seen=%d\n", n); return 0; }
EOF
gcc -fopenmp /tmp/omp.c -o /tmp/omp && echo "omp_compiled=OK"
for th in 1 2 4 8; do echo -n "OMP_NUM_THREADS=$th -> "; OMP_NUM_THREADS=$th /tmp/omp; done

echo "=== MPI compile+run ==="
cat > /tmp/hi.c <<'EOF'
#include <mpi.h>
#include <stdio.h>
int main(int argc,char**argv){ MPI_Init(&argc,&argv);
  int r,s; MPI_Comm_rank(MPI_COMM_WORLD,&r); MPI_Comm_size(MPI_COMM_WORLD,&s);
  if(r==0) printf("mpi_size=%d\n", s); MPI_Finalize(); return 0; }
EOF
mpicc /tmp/hi.c -o /tmp/hi && echo "mpi_compiled=OK"
mpirun --allow-run-as-root --oversubscribe -n 4 /tmp/hi 2>&1 | tail -2
echo "=== DONE ==="
