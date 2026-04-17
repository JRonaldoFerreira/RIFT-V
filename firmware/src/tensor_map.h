#pragma once
#include <stdint.h>

typedef struct {
  const char* name;
  uint32_t tensor_index;
  uint32_t offset;
  uint32_t length;
} tensor_span_t;

static const tensor_span_t g_tensor_map[] = {
  {"sequential_1_1/global_average_pooling2d_1_1/Mean/reduction_indices", 1u, 26856u, 8u},
  {"sequential_1_1/dense_3_1/BiasAdd", 2u, 26804u, 40u},
  {"sequential_1_1/dense_3_1/MatMul", 3u, 26472u, 320u},
  {"sequential_1_1/dense_2_1/Relu;sequential_1_1/dense_2_1/BiasAdd", 4u, 26332u, 128u},
  {"sequential_1_1/dense_2_1/MatMul", 5u, 24272u, 2048u},
  {"sequential_1_1/conv2d_5_1/Relu;sequential_1_1/conv2d_5_1/BiasAdd;sequential_1_1/conv2d_5_1/convolution;sequential_1_1/conv2d_5_1/Squeeze", 6u, 24004u, 256u},
  {"sequential_1_1/conv2d_5_1/convolution", 7u, 5560u, 18432u},
  {"sequential_1_1/conv2d_4_1/Relu;sequential_1_1/conv2d_4_1/BiasAdd;sequential_1_1/conv2d_4_1/convolution;sequential_1_1/conv2d_4_1/Squeeze", 8u, 5420u, 128u},
  {"sequential_1_1/conv2d_4_1/convolution", 9u, 800u, 4608u},
  {"sequential_1_1/conv2d_3_1/Relu;sequential_1_1/conv2d_3_1/BiasAdd;sequential_1_1/conv2d_3_1/convolution;sequential_1_1/conv2d_3_1/Squeeze", 10u, 724u, 64u},
  {"sequential_1_1/conv2d_3_1/convolution", 11u, 568u, 144u},
};
static const uint32_t g_tensor_map_count = 11u;
