# TensorFlow Rotated IOU GPU & CPU op

Calculate the piecewise rotated/skew IOU of two lists of boxes.

Based on [the custom op guide for TensorFlow](https://github.com/tensorflow/custom-op/)

# Quick build guide:
```bash
docker pull tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16 # so far there is no image for tensorflow-2.2.0

docker run -it -v ${PWD}:/working_dir -w /working_dir tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16
```

Within the docker container:
```bash
# Answer all questions with Y
./configure.sh
bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts
```

You will find a python whl file in the artifacts folder.

## Tests

To run the tests:

```bash
export PATH=/usr/local/cuda-10.1/bin/:/usr/local/cuda-10.1/lib64/:$PATH
make rotated_iou_test
```

## Benchmark 

There are some test to measure the performance under real world use cases:

```bash
export PATH=/usr/local/cuda-10.1/bin/:/usr/local/cuda-10.1/lib64/:$PATH
make benchmark
```

## Help

Some FAQ answers can be found with the original repo:

[the custom op guide for TensorFlow](https://github.com/tensorflow/custom-op/)