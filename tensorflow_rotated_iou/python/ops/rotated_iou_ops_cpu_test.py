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

from tensorflow import tile
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
try:
  from tensorflow_rotated_iou.python.ops import rotated_iou_ops
except ImportError:
  import rotated_iou_ops

def rad(value: float) -> float:
  return value / 180.0 * np.pi

class RotatedIOUTestCPU(test.TestCase):

  def testRotatedIOUGridFullIntersection(self):
    with self.cached_session():
      with ops.device("/cpu:0"):
        self.assertAllClose(
            rotated_iou_ops.rotated_iou_grid([[0., 0., 2., 3., 0.]], [[0., 0., 2., 3., 0.]]), np.array([[1.]]))

  def testRotatedIOUGridNoIntersection(self):
    with self.cached_session():
      with ops.device("/cpu:0"):
        self.assertAllClose(
          rotated_iou_ops.rotated_iou_grid([[0., 0., 2., 3., 0.]], [[100., 100., 2., 3., 0.]]), np.array([[0.]]))

  def testRotatedIOUGridOneThirdIntersection(self):
    with self.cached_session():
      with ops.device("/cpu:0"):
        self.assertAllClose(
          rotated_iou_ops.rotated_iou_grid([[0., 0., 2., 3., 0.]], [[1., 0., 2., 3., 0.]]), np.array([[0.333333]]))

  def testRotatedIOUGridMultiIntersection(self):
    with self.cached_session():
      with ops.device("/cpu:0"):
        self.assertAllClose(
          rotated_iou_ops.rotated_iou_grid([[0., 0., 2., 3., 0.], [0., 0., 2., 3., 0.], [0., 0., 2., 3., 0.]],
                                      [[0., 0., 2., 3., 0.], [100., 100., 2., 3., 0.], [1., 0., 2., 3., 0.], ]),
          np.array([[1.0, 0.0, 0.333333],
                    [1.0, 0.0, 0.333333],
                    [1.0, 0.0, 0.333333]])
        )

  def testRotatedIOUGridRowColumns(self):
    with self.cached_session():
      with ops.device("/cpu:0"):
        self.assertAllClose(
          rotated_iou_ops.rotated_iou_grid([[0., 0., 2., 3., 0.], [100., 100., 2., 3., 0.], [0., 0., 2., 3., 0.]],
                                      [[100., 100., 2., 3., 0.], [0., 0., 2., 3., 0.], [0., 0., 2., 3., 0.], ]),
          np.array([[0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0]])
        )

  def testRotatedIOUGridWithRotation(self):
    with self.cached_session():
      with ops.device("/cpu:0"):
        self.assertAllClose(
          rotated_iou_ops.rotated_iou_grid([[0., 0., 2., 6., 0.]], [[0., 0., 2., 6., rad(90.)]]), np.array([[0.2]]))

  def testRotatedIOUGridRotated30(self):
    with self.cached_session():
      with ops.device("/cpu:0"):
        self.assertAllClose(
          rotated_iou_ops.rotated_iou_grid([[0., 0., 2., 3., 0.]], [[0., 0., 2., 3., rad(30.)]]), np.array([[0.69422]]))

  def testRotatedIOUGridRotated1(self):
    with self.cached_session():
      with ops.device("/cpu:0"):
        self.assertAllClose(
          rotated_iou_ops.rotated_iou_grid([[0., 0., 2., 3., 0.]], [[0., 0., 2., 3., rad(1.)]]), np.array([[0.981565]]))

  def testRotatedIOUZeroSizeBox(self):
    with self.cached_session():
      with ops.device("/cpu:0"):
        self.assertAllClose(
          rotated_iou_ops.rotated_iou_grid([[0., 0., 2., 3., 0.]], [[0., 0., 0., 0., 0.]]), np.array([[0.0]]))

  def testRotatedIOU(self):
    # Test the picewise IOU function
    with self.cached_session():
      with ops.device("/cpu:0"):
        self.assertAllClose(
          rotated_iou_ops.rotated_iou(
            [[0., 0., 2., 3., 0.], [100., 100., 2., 3., 0.], [0., 0., 2., 6., 0.], [0., 0., 2., 3., 0.]],
            [[100., 100., 2., 3., 0.], [0., 0., 2., 3., 0.], [0., 0., 2., 6., rad(90.)], [0., 0., 2., 3., 0.]]),
          np.array([0.0, 0.0, 0.2, 1.0])
        )

if __name__ == '__main__':
  test.main()
