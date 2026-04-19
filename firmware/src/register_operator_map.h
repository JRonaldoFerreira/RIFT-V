#pragma once
#include <stdint.h>

typedef struct {
  uint32_t op_index;
  const char* op_name;
} register_op_name_t;

static const register_op_name_t g_register_op_names[] = {
  {0u, "CONV_2D"},
  {2u, "CONV_2D"},
  {4u, "CONV_2D"},
  {6u, "FULLY_CONNECTED"},
  {7u, "FULLY_CONNECTED"},
};
static const uint32_t g_register_op_names_count = 5u;
