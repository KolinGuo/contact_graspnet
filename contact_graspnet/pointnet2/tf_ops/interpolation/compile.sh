TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared \
  -I /usr/local/cuda/include ${TF_CFLAGS[@]} -fPIC \
  -lcudart -L /usr/local/cuda/lib64 ${TF_LFLAGS[@]} -O3
