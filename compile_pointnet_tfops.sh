#!/bin/bash

CUDA_INCLUDE='-I/usr/local/cuda/include/'
CUDA_LIB='-L/usr/local/cuda/lib64/'

############################################################
# Section 0: Bash Error Handling                           #
############################################################
set -eEu -o pipefail
trap 'catch' ERR  # Trap all errors (status != 0) and call catch()
catch() {
  local err="$?"
  local err_command="$BASH_COMMAND"
  set +xv  # disable trace printing

  echo -e "\n\e[1;31mCaught error in ${BASH_SOURCE[1]}:${BASH_LINENO[0]} ('${err_command}' exited with status ${err})\e[0m" >&2
  echo "Traceback (most recent call last, command might not be complete):" >&2
  for ((i = 0; i < ${#FUNCNAME[@]} - 1; i++)); do
    local funcname="${FUNCNAME[$i]}"
    [ "$i" -eq "0" ] && funcname=$err_command
    echo -e "  ($i) ${BASH_SOURCE[$i+1]}:${BASH_LINENO[$i]}\t'${funcname}'" >&2
  done
  exit "$err"
}

############################################################
# Section 1: Compile pointnet2 tf_ops                      #
############################################################
# Move to the repo folder, so later commands can use relative paths
SCRIPT_PATH=$(readlink -f "$0")
REPO_DIR=$(dirname "$SCRIPT_PATH")
cd "$REPO_DIR"

TF_CFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# tf_ops/sampling
cd contact_graspnet/pointnet2/tf_ops/sampling

nvcc -std=c++11 tf_sampling_g.cu -o tf_sampling_g.cu.o -c \
  ${CUDA_INCLUDE} ${TF_CFLAGS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O3

g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared \
  ${CUDA_INCLUDE} ${TF_CFLAGS} -fPIC -lcudart ${CUDA_LIB} ${TF_LFLAGS} -O3
rm -v tf_sampling_g.cu.o

echo 'testing sampling'
python3 tf_sampling.py
rm -v 1.pkl
 
# tf_ops/grouping
cd ../grouping

nvcc -std=c++11 tf_grouping_g.cu -o tf_grouping_g.cu.o -c \
  ${CUDA_INCLUDE} ${TF_CFLAGS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O3

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared \
  ${CUDA_INCLUDE} ${TF_CFLAGS} -fPIC -lcudart ${CUDA_LIB} ${TF_LFLAGS} -O3
rm -v tf_grouping_g.cu.o

echo 'testing grouping'
python3 tf_grouping_op_test.py
rm -rf __pycache__

# tf_ops/interpolation
cd ../interpolation

g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared \
  ${CUDA_INCLUDE} ${TF_CFLAGS} -fPIC -lcudart ${CUDA_LIB} ${TF_LFLAGS} -O3

echo 'testing interpolate'
python3 tf_interpolate_op_test.py
rm -rf __pycache__
