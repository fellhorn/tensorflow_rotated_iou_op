CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

ZERO_OUT_SRCS = $(wildcard tensorflow_zero_out/cc/kernels/*.cc) $(wildcard tensorflow_zero_out/cc/ops/*.cc)
TIME_TWO_SRCS = tensorflow_time_two/cc/kernels/time_two_kernels.cc $(wildcard tensorflow_time_two/cc/kernels/*.h) $(wildcard tensorflow_time_two/cc/ops/*.cc)

ROTATED_IOU_SRCS = tensorflow_rotated_iou/cc/kernels/rotated_iou_kernels.cc $(wildcard tensorflow_rotated_iou/cc/kernels/*.h) $(wildcard tensorflow_rotated_iou/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# I had to remove fPIC here
CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
CFLAGS_NVCC = ${TF_CFLAGS} -Xcompiler -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

ZERO_OUT_TARGET_LIB = tensorflow_zero_out/python/ops/_zero_out_ops.so
TIME_TWO_GPU_ONLY_TARGET_LIB = tensorflow_time_two/python/ops/_time_two_ops.cu.o
TIME_TWO_TARGET_LIB = tensorflow_time_two/python/ops/_time_two_ops.so

ROTATED_IOU_GPU_ONLY_TARGET_LIB = tensorflow_rotated_iou/python/ops/_rotated_iou_ops.cu.o
ROTATED_IOU_TARGET_LIB = tensorflow_rotated_iou/python/ops/_rotated_iou_ops.so

# zero_out op for CPU
zero_out_op: $(ZERO_OUT_TARGET_LIB)

$(ZERO_OUT_TARGET_LIB): $(ZERO_OUT_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

zero_out_test: tensorflow_zero_out/python/ops/zero_out_ops_test.py tensorflow_zero_out/python/ops/zero_out_ops.py $(ZERO_OUT_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_zero_out/python/ops/zero_out_ops_test.py

zero_out_pip_pkg: $(ZERO_OUT_TARGET_LIB)
	./build_pip_pkg.sh make artifacts


# time_two op for GPU
time_two_gpu_only: $(TIME_TWO_GPU_ONLY_TARGET_LIB)

$(TIME_TWO_GPU_ONLY_TARGET_LIB): tensorflow_time_two/cc/kernels/time_two_kernels.cu.cc
	$(NVCC) -std=c++11 -c -o $@ $^  $(CFLAGS) $(TF_LFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

time_two_op: $(TIME_TWO_TARGET_LIB)
$(TIME_TWO_TARGET_LIB): $(TIME_TWO_SRCS) $(TIME_TWO_GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda-10.0/targets/x86_64-linux/lib -lcudart

time_two_test: tensorflow_time_two/python/ops/time_two_ops_test.py tensorflow_time_two/python/ops/time_two_ops.py $(TIME_TWO_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_time_two/python/ops/time_two_ops_test.py


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

pip_pkg: rotated_iou_op
	./build_pip_pkg.sh make artifacts

clean:
	rm -f $(ZERO_OUT_TARGET_LIB) $(TIME_TWO_GPU_ONLY_TARGET_LIB) $(TIME_TWO_TARGET_LIB) $(ROTATED_IOU_GPU_ONLY_TARGET_LIB) $(ROTATED_IOU_TARGET_LIB)

