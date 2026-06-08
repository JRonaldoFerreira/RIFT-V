#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FI_OP_INDEX_WILDCARD 0xFFFFFFFFu

#if defined(FI_CAMPAIGN_MEMORY)
#include "fault_list.h"
#else
typedef struct {
  int32_t fault_id;
  uint32_t op_index;
  uint32_t tensor_index;
  uint32_t tensor_offset;
  uint32_t tensor_length;
  uint32_t byte_in_tensor;
  uint32_t global_byte_offset;
  uint8_t bit_index;
} memory_fault_t;
#endif

/* Memory-FI overhead accumulators — see register_fi.h for the matching
 * register-campaign struct.  Memory injections are synchronous (bit
 * flip happens directly inside `memory_fi_before_op`), so the
 * accounting is simpler than the register path: there is no ISR
 * dispatch latency and no separate timer cost. */
typedef struct {
  uint64_t before_op_cycles;
  uint32_t before_op_count;
  uint64_t after_op_cycles;
  uint32_t after_op_count;
  uint32_t injections;
} memory_fi_overhead_t;

typedef struct {
  uint8_t* model_base;
  const memory_fault_t* fault_list;
  uint32_t fault_count;
  volatile int32_t active_fault_index;
  bool injected_this_inference;
  uint8_t saved_original_byte;
  bool injection_active;

  bool overhead_enabled;
  memory_fi_overhead_t overhead;

#ifdef MULTI_FI_MAX_COUNT
  const memory_fault_t* group_faults[MULTI_FI_MAX_COUNT];
  uint8_t group_fault_count;
  uint8_t saved_group_bytes[MULTI_FI_MAX_COUNT];
  bool group_injection_active[MULTI_FI_MAX_COUNT];
#endif
} memory_fi_context_t;

void memory_fi_init(memory_fi_context_t* ctx,
                    uint8_t* model_base,
                    const memory_fault_t* fault_list,
                    uint32_t fault_count);

void memory_fi_begin_inference(memory_fi_context_t* ctx, int32_t fault_index);
void memory_fi_end_inference(memory_fi_context_t* ctx);
const memory_fault_t* memory_fi_current_fault(const memory_fi_context_t* ctx);

void memory_fi_before_op(memory_fi_context_t* ctx, uint32_t op_index);
void memory_fi_after_op(memory_fi_context_t* ctx, uint32_t op_index);

void memory_fi_set_overhead_enabled(memory_fi_context_t* ctx, bool enabled);
void memory_fi_reset_overhead(memory_fi_context_t* ctx);
void memory_fi_get_overhead(const memory_fi_context_t* ctx,
                            memory_fi_overhead_t* out);

static inline uint8_t* memory_fi_resolve_addr(
    const memory_fi_context_t* ctx,
    const memory_fault_t* fault) {
  return &ctx->model_base[fault->global_byte_offset];
}

#ifdef MULTI_FI_MAX_COUNT
void memory_fi_set_group(memory_fi_context_t* ctx,
                         const memory_fault_t* pool,
                         const uint16_t* pool_indices,
                         uint8_t count);
void memory_fi_clear_group(memory_fi_context_t* ctx);
#endif

#ifdef __cplusplus
}
#endif
