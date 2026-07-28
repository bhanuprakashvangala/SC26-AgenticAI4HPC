#!/bin/bash
WORK=/tmp/pdemo; mkdir -p $WORK; cd $WORK

# --- reference: serial, correct ---
cat > ref.c <<'EOF'
#include <stdio.h>
int main(){ long long N=20000000, c=0;
  for(long long i=0;i<N;i++){ int v=(int)((i*1103515245LL+12345)%100); if(v>50) c++; }
  printf("%lld\n", c); return 0; }
EOF
gcc -O2 ref.c -o ref && REF=$(./ref)
echo "REFERENCE=$REF"

# --- candidate A: correct (reduction) ---
cat > candA.c <<'EOF'
#include <stdio.h>
#include <omp.h>
int main(){ long long N=20000000, c=0;
  #pragma omp parallel for reduction(+:c)
  for(long long i=0;i<N;i++){ int v=(int)((i*1103515245LL+12345)%100); if(v>50) c++; }
  printf("%lld\n", c); return 0; }
EOF

# --- candidate B: racy (shared counter, no sync) ---
cat > candB.c <<'EOF'
#include <stdio.h>
#include <omp.h>
int main(){ long long N=20000000, c=0;
  #pragma omp parallel for
  for(long long i=0;i<N;i++){ int v=(int)((i*1103515245LL+12345)%100); if(v>50) c++; }
  printf("%lld\n", c); return 0; }
EOF

test_cand(){
  name=$1; src=$2
  gcc -O2 -fopenmp $src -o $name 2>/tmp/err || { echo "$name: COMPILE_FAIL"; return; }
  allok=1; det=""
  for th in 1 2 4 8; do
    for rep in 1 2 3; do
      out=$(OMP_NUM_THREADS=$th ./$name)
      ok=$([ "$out" = "$REF" ] && echo 1 || echo 0)
      [ "$ok" = "0" ] && allok=0
      det="$det t${th}r${rep}=${out}(${ok});"
    done
  done
  [ "$allok" = "1" ] && echo "$name: ROBUST_CORRECT" || echo "$name: NOT_ROBUST -> $det"
}
test_cand candA candA.c
test_cand candB candB.c
echo DONE
