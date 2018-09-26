#!/bin/bash

bin_procs="get_procs"
cd ../util
g++ -o ${bin_procs} get_maximum_procs.cpp -fopenmp
export MXNET_CPU_WORKER_NTHREADS=`./${bin_procs}`
echo "MXNET_CPU_WORKER_NTHREADS number is ${MXNET_CPU_WORKER_NTHREADS}"
rm -rf ${bin_procs}
cd ../

source activate gluon
python train_net_1.py 
