#pragma once
#include <stdint.h>

typedef struct {
  const char* name;
  uint32_t tensor_index;
  uint32_t offset;
  uint32_t length;
} tensor_span_t;

static const tensor_span_t g_tensor_map[] = {
  {"LeNet5_MicroFI_like_1/Flatten_1/strided_slice/stack", 1u, 63184u, 4u},
  {"arith.constant", 2u, 63168u, 4u},
  {"LeNet5_MicroFI_like_1/Flatten_1/Reshape/shape/1", 3u, 63152u, 4u},
  {"LeNet5_MicroFI_like_1/FC2_1/BiasAdd", 4u, 63100u, 40u},
  {"LeNet5_MicroFI_like_1/FC2_1/MatMul", 5u, 62248u, 840u},
  {"LeNet5_MicroFI_like_1/FC1_1/BiasAdd", 6u, 61900u, 336u},
  {"LeNet5_MicroFI_like_1/FC1_1/MatMul", 7u, 51808u, 10080u},
  {"LeNet5_MicroFI_like_1/Conv3_1/BiasAdd;LeNet5_MicroFI_like_1/Conv3_1/convolution;LeNet5_MicroFI_like_1/Conv3_1/Squeeze", 8u, 51316u, 480u},
  {"LeNet5_MicroFI_like_1/Conv3_1/convolution", 9u, 3304u, 48000u},
  {"LeNet5_MicroFI_like_1/Conv2_1/BiasAdd;LeNet5_MicroFI_like_1/Conv2_1/convolution;LeNet5_MicroFI_like_1/Conv2_1/Squeeze", 10u, 3228u, 64u},
  {"LeNet5_MicroFI_like_1/Conv2_1/convolution", 11u, 816u, 2400u},
  {"LeNet5_MicroFI_like_1/Conv1_1/BiasAdd;LeNet5_MicroFI_like_1/Conv1_1/convolution;LeNet5_MicroFI_like_1/Conv1_1/Squeeze", 12u, 780u, 24u},
  {"LeNet5_MicroFI_like_1/Conv1_1/convolution", 13u, 616u, 150u},
};
static const uint32_t g_tensor_map_count = 13u;
