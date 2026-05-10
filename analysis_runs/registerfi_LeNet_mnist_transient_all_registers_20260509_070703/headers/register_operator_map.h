#pragma once
#include <stdint.h>

typedef struct {
  uint32_t op_index;
  const char* op_name;
} register_op_name_t;

static const register_op_name_t g_register_op_names[] = {
  {0u, "CONV_2D"},
  {1u, "CONV_2D"},
  {3u, "CONV_2D"},
  {4u, "CONV_2D"},
  {6u, "CONV_2D"},
  {7u, "CONV_2D"},
  {13u, "FULLY_CONNECTED"},
  {14u, "FULLY_CONNECTED"},
  {15u, "FULLY_CONNECTED"},
  {16u, "FULLY_CONNECTED"},
};
static const uint32_t g_register_op_names_count = 10u;
