TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O3 \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared \
  -I /usr/local/cuda/include ${TF_CFLAGS[@]} -fPIC \
  -lcudart -L /usr/local/cuda/lib64 ${TF_LFLAGS[@]} -O3
