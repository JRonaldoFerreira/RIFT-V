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
  {0, 0u, 0u, 1u, 19u, 30u},
  {1, 0u, 0u, 1u, 20u, 30u},
  {2, 3u, 0u, 1u, 19u, 30u},
  {3, 3u, 0u, 1u, 20u, 30u},
  {4, 6u, 0u, 1u, 19u, 30u},
  {5, 6u, 0u, 1u, 20u, 30u},
  {6, 12u, 0u, 1u, 19u, 30u},
  {7, 12u, 0u, 1u, 20u, 30u},
  {8, 14u, 0u, 1u, 19u, 30u},
  {9, 14u, 0u, 1u, 20u, 30u},
};
static const uint32_t g_fault_list_count = 10u;
