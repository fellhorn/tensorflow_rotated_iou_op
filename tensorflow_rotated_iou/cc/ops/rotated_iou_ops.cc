/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("RotatedIOUGrid")
    .Attr("T: {float}")
    .Input("vertices1: T")
    .Input("vertices2: T")
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle result_shape;

      shape_inference::ShapeHandle boxes1_input_shape = c->input(0);
      shape_inference::ShapeHandle boxes2_input_shape = c->input(1);

      result_shape = c->MakeShape({c->Dim(boxes1_input_shape, 0), c->Dim(boxes2_input_shape, 0)});

      c->set_output(0,result_shape);
      return Status::OK();
    });

REGISTER_OP("RotatedIOU")
    .Attr("T: {float}")
    .Input("vertices1: T")
    .Input("vertices2: T")
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle result_shape;

      shape_inference::ShapeHandle boxes1_input_shape = c->input(0);
      shape_inference::ShapeHandle boxes2_input_shape = c->input(1);

      // DCHECK_EQ(c->Dim(boxes1_input_shape, 0), c->Dim(boxes1_input_shape, 0));

      result_shape = c->MakeShape({c->Dim(boxes1_input_shape, 0)});

      c->set_output(0,result_shape);
      return Status::OK();
    });
