CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python3

ROTATED_IOU_SRCS = tensorflow_rotated_iou/cc/kernels/rotated_iou_kernels.cc $(wildcard tensorflow_rotated_iou/cc/kernels/*.h) $(wildcard tensorflow_rotated_iou/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# I had to remove fPIC here
CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
CFLAGS_NVCC = ${TF_CFLAGS} -Xcompiler -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

ROTATED_IOU_GPU_ONLY_TARGET_LIB = tensorflow_rotated_iou/python/ops/_rotated_iou_ops.cu.o
ROTATED_IOU_TARGET_LIB = tensorflow_rotated_iou/python/ops/_rotated_iou_ops.so

# rotated_iou op for GPU
rotated_iou_gpu_only: $(ROTATED_IOU_GPU_ONLY_TARGET_LIB)

$(ROTATED_IOU_GPU_ONLY_TARGET_LIB): tensorflow_rotated_iou/cc/kernels/rotated_iou_kernels.cu.cc
	$(NVCC) -std=c++11 -c -o $@ $^  $(CFLAGS_NVCC) $(TF_LFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

rotated_iou_op: $(ROTATED_IOU_TARGET_LIB)
$(ROTATED_IOU_TARGET_LIB): $(ROTATED_IOU_SRCS) $(ROTATED_IOU_GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda-10.0/targets/x86_64-linux/lib

rotated_iou_test: tensorflow_rotated_iou/python/ops/rotated_iou_ops_gpu_test.py tensorflow_rotated_iou/python/ops/rotated_iou_ops.py $(ROTATED_IOU_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_rotated_iou/python/ops/rotated_iou_ops_gpu_test.py
	$(PYTHON_BIN_PATH) tensorflow_rotated_iou/python/ops/rotated_iou_ops_cpu_test.py


benchmark: tensorflow_rotated_iou/python/ops/benchmark.py $(ROTATED_IOU_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_rotated_iou/python/ops/benchmark.py

pip_pkg: rotated_iou_op
	./build_pip_pkg.sh make artifacts

clean:
	rm -f $(ROTATED_IOU_GPU_ONLY_TARGET_LIB) $(ROTATED_IOU_TARGET_LIB)

