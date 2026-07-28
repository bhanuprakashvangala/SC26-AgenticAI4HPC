#!/bin/bash
source /etc/profile.d/modules.sh 2>/dev/null || true
echo "=== available modules ==="
module avail 2>&1 | head -40
echo "=== try load mpi + cuda ==="
module load mpi/openmpi 2>/dev/null || module load mpi 2>/dev/null || true
module load cuda 2>/dev/null || true
echo "mpicc=$(which mpicc 2>/dev/null)"
echo "mpirun=$(which mpirun 2>/dev/null)"
echo "nvcc=$(which nvcc 2>/dev/null)"
# also probe common fixed paths
echo "=== fixed-path probe ==="
ls /usr/local/cuda/bin/nvcc 2>/dev/null && echo "cuda_at_/usr/local/cuda"
ls /opt/openmpi*/bin/mpicc 2>/dev/null | head -1
which hpcx 2>/dev/null
echo DONE
