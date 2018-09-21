#!/bin/bash

bin_procs="get_procs"
cd ../util
g++ -o ${bin_procs} get_maximum_procs.cpp -fopenmp
export MXNET_CPU_WORKER_NTHREADS=`./${bin_procs}`
echo "MXNET_CPU_WORKER_NTHREADS number is ${MXNET_CPU_WORKER_NTHREADS}"
rm -rf ${bin_procs}
cd ../

source activate gluon
python train_net.py --train-list 'test_dataset/data.lst' --test-list 'test_dataset/data_test.lst' --root '/mnt/data-1/data/jiajie.tang/highway/test_dataset' --lr 0.01 --batch-size 64 --gpus 0,1,2,3
