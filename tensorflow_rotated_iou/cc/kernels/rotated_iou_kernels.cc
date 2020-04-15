/* 
Based on https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation/blob/master/libs/box_utils/rbbox_overlaps_kernel.cu

Published under MIT License

Original Copyright (c) 2018 DetectionTeamUCAS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "rotated_iou.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

inline float trangle_area(float * a, float * b, float * c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))/2.0;
}


inline float area(float * int_pts, int num_of_inter) {
  float area = 0.0;
  for(int i = 0;i < num_of_inter - 2;i++) {
    area += fabs(trangle_area(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}


inline void reorder_pts(float * int_pts, int num_of_inter) {
  if(num_of_inter > 0) {

    float center[2];

    center[0] = 0.0;
    center[1] = 0.0;

    for(int i = 0;i < num_of_inter;i++) {
      center[0] += int_pts[2 * i];
      center[1] += int_pts[2 * i + 1];
    }
    center[0] /= num_of_inter;
    center[1] /= num_of_inter;

    float vs[16];
    float v[2];
    float d;
    for(int i = 0;i < num_of_inter;i++) {
      v[0] = int_pts[2 * i]-center[0];
      v[1] = int_pts[2 * i + 1]-center[1];
      d = sqrt(v[0] * v[0] + v[1] * v[1]);
      v[0] = v[0] / d;
      v[1] = v[1] / d;
      if(v[1] < 0) {
        v[0]= - 2 - v[0];
      }
      vs[i] = v[0];
    }

    float temp,tx,ty;
    int j;
    for(int i=1;i<num_of_inter;++i){
      if(vs[i-1]>vs[i]){
        temp = vs[i];
        tx = int_pts[2*i];
        ty = int_pts[2*i+1];
        j=i;
        while(j>0&&vs[j-1]>temp){
          vs[j] = vs[j-1];
          int_pts[j*2] = int_pts[j*2-2];
          int_pts[j*2+1] = int_pts[j*2-1];
          j--;
        }
        vs[j] = temp;
        int_pts[j*2] = tx;
        int_pts[j*2+1] = ty;
      }
    }
  }
}


inline bool inter2line(float * pts1, float *pts2, int i, int j, float * temp_pts) {
  float a[2];
  float b[2];
  float c[2];
  float d[2];

  float area_abc, area_abd, area_cda, area_cdb;

  a[0] = pts1[2 * i];
  a[1] = pts1[2 * i + 1];

  b[0] = pts1[2 * ((i + 1) % 4)];
  b[1] = pts1[2 * ((i + 1) % 4) + 1];

  c[0] = pts2[2 * j];
  c[1] = pts2[2 * j + 1];

  d[0] = pts2[2 * ((j + 1) % 4)];
  d[1] = pts2[2 * ((j + 1) % 4) + 1];

  area_abc = trangle_area(a, b, c);
  area_abd = trangle_area(a, b, d);

  if(area_abc * area_abd >= -1e-5) {
    return false;
  }

  area_cda = trangle_area(c, d, a);
  area_cdb = area_cda + area_abc - area_abd;

  if (area_cda * area_cdb >= -1e-5) {
    return false;
  }
  float t = area_cda / (area_abd - area_abc);

  float dx = t * (b[0] - a[0]);
  float dy = t * (b[1] - a[1]);
  temp_pts[0] = a[0] + dx;
  temp_pts[1] = a[1] + dy;

  return true;
}


inline bool inrect(float pt_x, float pt_y, float * pts) {

  double ab[2];
  double ad[2];
  double ap[2];

  double abab;
  double abap;
  double adad;
  double adap;

  ab[0] = pts[2] - pts[0];
  ab[1] = pts[3] - pts[1];

  ad[0] = pts[6] - pts[0];
  ad[1] = pts[7] - pts[1];

  ap[0] = pt_x - pts[0];
  ap[1] = pt_y - pts[1];

  abab = ab[0] * ab[0] + ab[1] * ab[1];
  abap = ab[0] * ap[0] + ab[1] * ap[1];
  adad = ad[0] * ad[0] + ad[1] * ad[1];
  adap = ad[0] * ap[0] + ad[1] * ap[1];

  bool result = (abab >= abap) and (abap >= 0) and (adad >= adap) and (adap >= 0);
  return result;
}


inline int inter_pts(float * pts1, float * pts2, float * int_pts) {
  int num_of_inter = 0;

  for(int i = 0;i < 4;i++) {
    if(inrect(pts1[2 * i], pts1[2 * i + 1], pts2)) {
      int_pts[num_of_inter * 2] = pts1[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
      num_of_inter++;
    }
     if(inrect(pts2[2 * i], pts2[2 * i + 1], pts1)) {
      int_pts[num_of_inter * 2] = pts2[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
      num_of_inter++;
    }
  }

  float temp_pts[2];

  for(int i = 0;i < 4;i++) {
    for(int j = 0;j < 4;j++) {
      bool has_pts = inter2line(pts1, pts2, i, j, temp_pts);
      if(has_pts) {
        int_pts[num_of_inter * 2] = temp_pts[0];
        int_pts[num_of_inter * 2 + 1] = temp_pts[1];
        num_of_inter++;
      }
    }
  }

  return num_of_inter;
}


inline void convert_region(float * pts , float const * const region) {

  float angle = region[4];
  float a_cos = cos(angle/180.0*3.1415926535);
  float a_sin = sin(angle/180.0*3.1415926535);

  float ctr_x = region[0];
  float ctr_y = region[1];

  float w = region[2];
  float h = region[3];

  float pts_x[4];
  float pts_y[4];

  pts_x[0] = - w / 2;
  pts_x[1] = w / 2;
  pts_x[2] = w / 2;
  pts_x[3] = - w / 2;

  pts_y[0] = - h / 2;
  pts_y[1] = - h / 2;
  pts_y[2] = h / 2;
  pts_y[3] = h / 2;

  for(int i = 0;i < 4;i++) {
    pts[7 - 2 * i - 1] = a_cos * pts_x[i] - a_sin * pts_y[i] + ctr_x;
    pts[7 - 2 * i] = a_sin * pts_x[i] + a_cos * pts_y[i] + ctr_y;

  }

}


inline float inter(float const * const region1, float const * const region2) {

  float pts1[8];
  float pts2[8];
  float int_pts[16];
  int num_of_inter;

  convert_region(pts1, region1);
  convert_region(pts2, region2);

  num_of_inter = inter_pts(pts1, pts2, int_pts);

  reorder_pts(int_pts, num_of_inter);

  return area(int_pts, num_of_inter);

}


inline float devRotateIoU(float const * const region1, float const * const region2) {

  if((fabs(region1[0] - region2[0]) < 1e-5) && (fabs(region1[1] - region2[1]) < 1e-5) && (fabs(region1[2] - region2[2]) < 1e-5) && (fabs(region1[3] - region2[3]) < 1e-5) && (fabs(region1[4] - region2[4]) < 1e-5)) {
    return 1.0;
  }

  float area1 = region1[2] * region1[3];
  float area2 = region2[2] * region2[3];
  float area_inter = inter(region1, region2);

  float result = area_inter / (area1 + area2 - area_inter);

  if(result < 0) {
    result = 0.0;
  }
  return result;
}

// CPU specialization of actual computation.
template <typename T>
struct RotatedIOUFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d,
    const int boxes1_size, const int boxes2_size,
    const T* boxes1, const T* boxes2,
    T* out
  ) {
    for (int i = 0; i < boxes1_size; ++i) {
        for (int j = 0; j < boxes2_size; ++j) {
            out[i * boxes2_size + j] = devRotateIoU(boxes1 + i * 5, boxes2 + j * 5);
        }
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class RotatedIOUOp : public OpKernel {
 public:
  explicit RotatedIOUOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& boxes1_input_tensor = context->input(0);
    const Tensor& boxes2_input_tensor = context->input(1);

    // Create an output tensor
    Tensor* output_tensor = NULL;

    TensorShape output_shape;
    output_shape.AddDim(boxes1_input_tensor.dim_size(0));
    output_shape.AddDim(boxes2_input_tensor.dim_size(0));

    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, boxes1_input_tensor.dim_size(0) <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    OP_REQUIRES(context, boxes2_input_tensor.dim_size(0) <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    RotatedIOUFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(boxes1_input_tensor.dim_size(0)),
        static_cast<int>(boxes2_input_tensor.dim_size(0)),
        boxes1_input_tensor.flat<T>().data(),
        boxes2_input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("RotatedIOU").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RotatedIOUOp<CPUDevice, T>);
REGISTER_CPU(float);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template struct RotatedIOUFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("RotatedIOU").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      RotatedIOUOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow
