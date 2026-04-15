#pragma once
#include <stdint.h>

typedef struct {
  const char* name;
  uint32_t tensor_index;
  uint32_t offset;
  uint32_t length;
} tensor_span_t;

static const tensor_span_t g_tensor_map[] = {
  {"le_net5_1_1/lenet5_1/flatten_1/strided_slice/stack", 1u, 63144u, 4u},
  {"arith.constant", 2u, 63128u, 4u},
  {"le_net5_1_1/lenet5_1/flatten_1/Reshape/shape/1", 3u, 63112u, 4u},
  {"le_net5_1_1/lenet5_1/fc2_1/BiasAdd", 4u, 63060u, 40u},
  {"le_net5_1_1/lenet5_1/fc2_1/MatMul", 5u, 62208u, 840u},
  {"le_net5_1_1/lenet5_1/fc1_1/Relu;le_net5_1_1/lenet5_1/fc1_1/BiasAdd", 6u, 61860u, 336u},
  {"le_net5_1_1/lenet5_1/fc1_1/MatMul", 7u, 51768u, 10080u},
  {"le_net5_1_1/lenet5_1/conv3_1/Relu;le_net5_1_1/lenet5_1/conv3_1/BiasAdd;le_net5_1_1/lenet5_1/conv3_1/convolution;le_net5_1_1/lenet5_1/conv3_1/Squeeze", 8u, 51276u, 480u},
  {"le_net5_1_1/lenet5_1/conv3_1/convolution", 9u, 3264u, 48000u},
  {"le_net5_1_1/lenet5_1/conv2_1/Relu;le_net5_1_1/lenet5_1/conv2_1/BiasAdd;le_net5_1_1/lenet5_1/conv2_1/convolution;le_net5_1_1/lenet5_1/conv2_1/Squeeze", 10u, 3188u, 64u},
  {"le_net5_1_1/lenet5_1/conv2_1/convolution", 11u, 776u, 2400u},
  {"le_net5_1_1/lenet5_1/conv1_1/Relu;le_net5_1_1/lenet5_1/conv1_1/BiasAdd;le_net5_1_1/lenet5_1/conv1_1/convolution;le_net5_1_1/lenet5_1/conv1_1/Squeeze", 12u, 740u, 24u},
  {"le_net5_1_1/lenet5_1/conv1_1/convolution", 13u, 576u, 150u},
};
static const uint32_t g_tensor_map_count = 13u;
