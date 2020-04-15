// kernel_example.h
#ifndef KERNEL_ROTATED_IOU_H_
#define KERNEL_ROTATED_IOU_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct RotatedIOUFunctor {
  void operator()(const Device& d,
    const int boxes1_size, const int boxes2_size,
    const T* boxes1, const T* boxes2,
    T* out
  );
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_ROTATED_IOU_H_
