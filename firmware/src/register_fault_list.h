#pragma once
#include <stdint.h>

typedef struct {
  int32_t fault_id;
  uint32_t op_index;
  uint8_t time_slot_index;
  uint8_t time_slot_count;
  uint8_t register_id;
  uint8_t bit_index;
} register_fault_t;

static const register_fault_t g_fault_list[] = {
  {0, 0u, 0u, 1u, 13u, 22u},
  {1, 2u, 0u, 1u, 13u, 22u},
  {2, 4u, 0u, 1u, 13u, 22u},
  {3, 6u, 0u, 1u, 13u, 22u},
  {4, 7u, 0u, 1u, 13u, 22u},
};
static const uint32_t g_fault_list_count = 5u;
