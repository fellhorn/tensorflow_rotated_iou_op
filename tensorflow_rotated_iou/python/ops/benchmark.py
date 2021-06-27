# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for rotated_iou ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow import tile
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
# from tensorflow.test import Benchmark

try:
  from tensorflow_rotated_iou.python.ops import rotated_iou_ops
except ImportError:
  import rotated_iou_ops

tf.debugging.set_log_device_placement(True)

tf.compat.v1.disable_eager_execution()

def rad(value: float) -> float:
  return value / 180.0 * np.pi

with tf.compat.v1.Session() as sess:
    with tf.device('/GPU:0'):
        a = tile([[0., 0., 2., 3., rad(1.)]], [100, 1])
        b = tile([[0., 0., 2., 3., 0.]], [3000000, 1])
        op = rotated_iou_ops.rotated_iou_grid(a, b)
        result = tf.test.Benchmark().run_op_benchmark(sess, op)
        print(result)
