/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "rotated_iou.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cmath>

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define THREADS_PER_BLOCK 256

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

__device__ inline float trangle_area(float * a, float * b, float * c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))/2.0;
}


__device__ inline float area(float * int_pts, int num_of_inter) {
  float area = 0.0;
  for(int i = 0;i < num_of_inter - 2;i++) {
    area += fabs(trangle_area(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}


__device__ inline void reorder_pts(float * int_pts, int num_of_inter) {
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


__device__ inline bool inter2line(float * pts1, float *pts2, int i, int j, float * temp_pts) {
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


__device__ inline bool inrect(float pt_x, float pt_y, float * pts) {

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
  // bool result = (abab - abap >=  -1) and (abap >= -1) and (adad - adap >= -1) and (adap >= -1);
  bool result = (abab >= abap) and (abap >= 0) and (adad >= adap) and (adap >= 0);
  return result;
}


__device__ inline int inter_pts(float * pts1, float * pts2, float * int_pts) {
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


__device__ inline void convert_region(float * pts , float const * const region) {

  float angle = region[4];
  float a_cos = cos(angle);
  float a_sin = sin(angle);

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


__device__ inline float inter(float const * const region1, float const * const region2) {

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


__device__ inline float devRotateIoU(float const * const region1, float const * const region2) {
  // no intersection for zero size boxes
  if(((fabs(region1[2]) < 1e-5) && (fabs(region1[3] < 1e-5))) ||
      ((fabs(region2[2]) < 1e-5) && (fabs(region2[3] < 1e-5)))) {
    return 0.0;
  }

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


// Define the CUDA kernel.
template <typename T>
__global__ void RotatedIOUGridCudaKernel(
  const int boxes1_size, const int boxes2_size,
  const T* boxes1, const T* boxes2,
  T* out) {

  // copy the data to shared data for faster access
  __shared__ float block_boxes1[THREADS_PER_BLOCK * 5];
  __shared__ float block_boxes2[THREADS_PER_BLOCK * 5];

  const int row_start = blockIdx.x * THREADS_PER_BLOCK;
  const int col_start = blockIdx.y * THREADS_PER_BLOCK;

  const int num_rows_block = min(boxes1_size - row_start, THREADS_PER_BLOCK);
  const int num_cols_block = min(boxes2_size - col_start, THREADS_PER_BLOCK);

  const int index_col = col_start + threadIdx.x;

  // move all data which is handled in this block to shared memory
  if (threadIdx.x < num_rows_block) {
    const int index_row = row_start + threadIdx.x;
    block_boxes1[threadIdx.x * 5 + 0] = boxes1[index_row * 5 + 0];
    block_boxes1[threadIdx.x * 5 + 1] = boxes1[index_row * 5 + 1];
    block_boxes1[threadIdx.x * 5 + 2] = boxes1[index_row * 5 + 2];
    block_boxes1[threadIdx.x * 5 + 3] = boxes1[index_row * 5 + 3];
    block_boxes1[threadIdx.x * 5 + 4] = boxes1[index_row * 5 + 4];
  }

  if (index_col < boxes2_size) {
    block_boxes2[threadIdx.x * 5 + 0] = boxes2[index_col * 5 + 0];
    block_boxes2[threadIdx.x * 5 + 1] = boxes2[index_col * 5 + 1];
    block_boxes2[threadIdx.x * 5 + 2] = boxes2[index_col * 5 + 2];
    block_boxes2[threadIdx.x * 5 + 3] = boxes2[index_col * 5 + 3];
    block_boxes2[threadIdx.x * 5 + 4] = boxes2[index_col * 5 + 4];
  }

  __syncthreads();

  // iterate over rows instead of cols as boxes1 < boxes2 in our case
  if (index_col < boxes2_size) {
    for(int i = 0; i < num_rows_block; i++) {
        int offset = (row_start + i) * boxes2_size + index_col;
        out[offset] = devRotateIoU(block_boxes1 + i * 5, block_boxes2 + threadIdx.x * 5);
    }
  }
}

template <typename T>
__global__ void RotatedIOUCudaKernel(
  const int boxes1_size, const int boxes2_size,
  const T* boxes1, const T* boxes2,
  T* out) {

  // copy the data to shared data for faster access
  __shared__ float block_boxes1[THREADS_PER_BLOCK * 5];
  __shared__ float block_boxes2[THREADS_PER_BLOCK * 5];
  const int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  if (i < boxes1_size) {
    block_boxes1[threadIdx.x * 5 + 0] = boxes1[i * 5 + 0];
    block_boxes1[threadIdx.x * 5 + 1] = boxes1[i * 5 + 1];
    block_boxes1[threadIdx.x * 5 + 2] = boxes1[i * 5 + 2];
    block_boxes1[threadIdx.x * 5 + 3] = boxes1[i * 5 + 3];
    block_boxes1[threadIdx.x * 5 + 4] = boxes1[i * 5 + 4];

    block_boxes2[threadIdx.x * 5 + 0] = boxes2[i * 5 + 0];
    block_boxes2[threadIdx.x * 5 + 1] = boxes2[i * 5 + 1];
    block_boxes2[threadIdx.x * 5 + 2] = boxes2[i * 5 + 2];
    block_boxes2[threadIdx.x * 5 + 3] = boxes2[i * 5 + 3];
    block_boxes2[threadIdx.x * 5 + 4] = boxes2[i * 5 + 4];
  }

  if (i < boxes1_size) {
    out[i] = devRotateIoU(block_boxes1 + threadIdx.x * 5, block_boxes2 + threadIdx.x * 5);
  }
}


// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct RotatedIOUGridFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
    const int boxes1_size, const int boxes2_size,
    const T* boxes1, const T* boxes2,
    T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    dim3 block_count(DIVUP(boxes1_size, THREADS_PER_BLOCK), DIVUP(boxes2_size, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    RotatedIOUGridCudaKernel<T>
        <<<block_count, threads, 0, d.stream()>>>(boxes1_size, boxes2_size, boxes1, boxes2, out);
  }
};

template <typename T>
struct RotatedIOUFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
    const int boxes1_size, const int boxes2_size,
    const T* boxes1, const T* boxes2,
    T* out) {
    dim3 block_count(DIVUP(boxes1_size, THREADS_PER_BLOCK));

    RotatedIOUCudaKernel<T>
        <<<block_count, THREADS_PER_BLOCK, 0, d.stream()>>>(boxes1_size, boxes2_size, boxes1, boxes2, out);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct RotatedIOUGridFunctor<GPUDevice, float>;
template struct RotatedIOUFunctor<GPUDevice, float>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
