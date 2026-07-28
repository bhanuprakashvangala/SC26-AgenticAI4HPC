#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq build-essential libopenmpi-dev openmpi-bin cmake git python3 python3-pip >/dev/null 2>&1
echo "GCC=$(gcc -dumpversion)"
echo "GPP=$(g++ -dumpversion)"
echo "MPICC=$(which mpicc)"
echo "MPIRUN=$(which mpirun)"
echo "NPROC=$(nproc)"
echo "DONE"
