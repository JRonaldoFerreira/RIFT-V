#include "memory_fi.h"

#include "fi_cycles.h"

#if defined(FI_CAMPAIGN_MEMORY)
static memory_fi_context_t* g_memory_ctx = nullptr;

extern "C" {
volatile uint32_t g_microfi_active_op_mask = 0u;
volatile uint8_t g_microfi_profile_all_ops = 0u;
volatile uint8_t g_microfi_count_skipped_hooks = 0u;
volatile uint32_t g_microfi_skipped_before_op_count = 0u;
volatile uint32_t g_microfi_skipped_after_op_count = 0u;
}
#endif

namespace {

constexpr uint32_t kMicrofiAllOpsMask = 0xFFFFFFFFu;
constexpr uint32_t kMicrofiNoOpsMask = 0u;

static inline uint32_t memory_fi_mask_for_op(uint32_t op_index) {
  if (op_index == FI_OP_INDEX_WILDCARD || op_index >= 32u) {
    return kMicrofiAllOpsMask;
  }
  return 1u << op_index;
}

static uint32_t memory_fi_build_active_op_mask(const memory_fi_context_t* ctx) {
  if (ctx == nullptr) {
    return kMicrofiNoOpsMask;
  }

  uint32_t op_mask = kMicrofiNoOpsMask;
  if (ctx->active_fault != nullptr) {
    op_mask = memory_fi_mask_for_op(ctx->active_target_op);
  }

#ifdef MULTI_FI_MAX_COUNT
  if (ctx->group_fault_count > 0u) {
    op_mask = kMicrofiNoOpsMask;
    for (uint8_t i = 0; i < ctx->group_fault_count; ++i) {
      const memory_fault_t* f = ctx->group_faults[i];
      if (f == nullptr) {
        continue;
      }
      const uint32_t fault_mask = memory_fi_mask_for_op(f->op_index);
      if (fault_mask == kMicrofiAllOpsMask) {
        return kMicrofiAllOpsMask;
      }
      op_mask |= fault_mask;
    }
  }
#endif

  return op_mask;
}

static void memory_fi_publish_active_op_mask(const memory_fi_context_t* ctx) {
#if defined(FI_CAMPAIGN_MEMORY)
  g_microfi_active_op_mask = g_microfi_profile_all_ops
      ? kMicrofiAllOpsMask
      : memory_fi_build_active_op_mask(ctx);
#else
  (void)ctx;
#endif
}

static void memory_fi_reset_callsite_counters(void) {
#if defined(FI_CAMPAIGN_MEMORY)
  g_microfi_skipped_before_op_count = 0u;
  g_microfi_skipped_after_op_count = 0u;
#endif
}

}  // namespace

void memory_fi_init(memory_fi_context_t* ctx,
                    uint8_t* model_base,
                    const memory_fault_t* fault_list,
                    uint32_t fault_count) {
  if (ctx == nullptr) {
    return;
  }
  ctx->model_base = model_base;
  ctx->fault_list = fault_list;
  ctx->fault_count = fault_count;
  ctx->active_fault_index = -1;
  ctx->active_fault = nullptr;
  ctx->active_target_op = FI_OP_INDEX_WILDCARD;
  ctx->active_addr = nullptr;
  ctx->active_mask = 0u;
  ctx->active_is_wildcard = false;
  ctx->injected_this_inference = false;
  ctx->saved_original_byte = 0u;
  ctx->injection_active = false;
  ctx->overhead_enabled = false;
  ctx->overhead = {};

#ifdef MULTI_FI_MAX_COUNT
  ctx->group_fault_count = 0;
  for (uint8_t i = 0; i < MULTI_FI_MAX_COUNT; ++i) {
    ctx->group_faults[i] = nullptr;
    ctx->saved_group_bytes[i] = 0;
    ctx->group_injection_active[i] = false;
  }
#endif

#if defined(FI_CAMPAIGN_MEMORY)
  g_memory_ctx = ctx;
  g_microfi_profile_all_ops = 0u;
  g_microfi_count_skipped_hooks = 0u;
#endif
  memory_fi_reset_callsite_counters();
  memory_fi_publish_active_op_mask(ctx);
}

void memory_fi_begin_inference(memory_fi_context_t* ctx, int32_t fault_index) {
  if (ctx == nullptr) {
    return;
  }
  ctx->active_fault_index = fault_index;
  ctx->active_fault = nullptr;
  ctx->active_target_op = FI_OP_INDEX_WILDCARD;
  ctx->active_addr = nullptr;
  ctx->active_mask = 0u;
  ctx->active_is_wildcard = false;
  if (fault_index >= 0 &&
      static_cast<uint32_t>(fault_index) < ctx->fault_count) {
    const memory_fault_t* f = &ctx->fault_list[fault_index];
    ctx->active_fault = f;
    ctx->active_target_op = f->op_index;
    ctx->active_addr = memory_fi_resolve_addr(ctx, f);
    ctx->active_mask = static_cast<uint8_t>(1u << f->bit_index);
    ctx->active_is_wildcard = (f->op_index == FI_OP_INDEX_WILDCARD);
  }

  ctx->injected_this_inference = false;
  ctx->saved_original_byte = 0u;
  ctx->injection_active = false;
  memory_fi_publish_active_op_mask(ctx);
}

void memory_fi_end_inference(memory_fi_context_t* ctx) {
  if (ctx == nullptr) {
    return;
  }

  if (ctx->injection_active && ctx->active_addr != nullptr) {
    *ctx->active_addr = ctx->saved_original_byte;
  }

#ifdef MULTI_FI_MAX_COUNT
  for (int8_t i = static_cast<int8_t>(ctx->group_fault_count) - 1; i >= 0; --i) {
    if (ctx->group_injection_active[i]) {
      uint8_t* addr = memory_fi_resolve_addr(ctx, ctx->group_faults[i]);
      *addr = ctx->saved_group_bytes[i];
      ctx->group_injection_active[i] = false;
    }
  }
#endif

  ctx->active_fault_index = -1;
  ctx->active_fault = nullptr;
  ctx->active_target_op = FI_OP_INDEX_WILDCARD;
  ctx->active_addr = nullptr;
  ctx->active_mask = 0u;
  ctx->active_is_wildcard = false;
  ctx->injected_this_inference = false;
  ctx->injection_active = false;
  memory_fi_publish_active_op_mask(ctx);
}

const memory_fault_t* memory_fi_current_fault(const memory_fi_context_t* ctx) {
  if (ctx == nullptr) {
    return nullptr;
  }
  return ctx->active_fault;
}

static inline void _force_stuck_bit(uint8_t* addr, uint8_t original_byte,
                                    uint8_t bit_index) {
  uint8_t mask = static_cast<uint8_t>(1u << bit_index);
  uint8_t original_bit = original_byte & mask;
  if (original_bit) {
    *addr &= ~mask;
  } else {
    *addr |= mask;
  }
}

/* Multi-fault stays outlined; the single-fault hot path uses cached
 * active_addr/active_mask metadata directly. */
#ifdef MULTI_FI_MAX_COUNT
namespace {
__attribute__((noinline)) static uint32_t memory_fi_before_op_group_slow(
    memory_fi_context_t* ctx, uint32_t op_index) {
  uint32_t injected_now = 0u;

#ifdef MULTI_FI_MAX_COUNT
  if (ctx->group_fault_count > 0) {
    for (uint8_t i = 0; i < ctx->group_fault_count; ++i) {
      const memory_fault_t* f = ctx->group_faults[i];
      if (f->op_index == op_index) {
        uint8_t* addr = memory_fi_resolve_addr(ctx, f);
        ctx->saved_group_bytes[i] = *addr;
      }
    }
    for (uint8_t i = 0; i < ctx->group_fault_count; ++i) {
      const memory_fault_t* f = ctx->group_faults[i];
      if (f->op_index == op_index) {
        uint8_t* addr = memory_fi_resolve_addr(ctx, f);
        *addr ^= static_cast<uint8_t>(1u << f->bit_index);
        ctx->group_injection_active[i] = true;
        ctx->injected_this_inference = true;
        injected_now += 1u;
      }
    }
    for (uint8_t i = 0; i < ctx->group_fault_count; ++i) {
      const memory_fault_t* f = ctx->group_faults[i];
      if (f->op_index != FI_OP_INDEX_WILDCARD) continue;
      uint8_t* addr = memory_fi_resolve_addr(ctx, f);
      if (!ctx->group_injection_active[i]) {
        ctx->saved_group_bytes[i] = *addr;
        ctx->group_injection_active[i] = true;
      }
      _force_stuck_bit(addr, ctx->saved_group_bytes[i], f->bit_index);
      ctx->injected_this_inference = true;
      injected_now += 1u;
    }
    return injected_now;
  }
#endif

  return injected_now;
}
}  // namespace
#endif

void memory_fi_before_op(memory_fi_context_t* ctx, uint32_t op_index) {
  if (ctx == nullptr) {
    return;
  }

  const uint32_t entry_cycle = ctx->overhead_enabled ? fi_read_cycles() : 0u;

  /* Single-fault hot path: use the pre-materialized address and mask. */
  uint32_t injected_now = 0u;
#ifdef MULTI_FI_MAX_COUNT
  if (ctx->group_fault_count > 0u) {
    injected_now = memory_fi_before_op_group_slow(ctx, op_index);
  } else
#endif
  {
    const memory_fault_t* fault = ctx->active_fault;
    if (fault != nullptr && ctx->active_addr != nullptr) {
      if (ctx->active_is_wildcard) {
        if (!ctx->injection_active) {
          ctx->saved_original_byte = *ctx->active_addr;
          ctx->injection_active = true;
        }
        _force_stuck_bit(ctx->active_addr,
                         ctx->saved_original_byte,
                         fault->bit_index);
        ctx->injected_this_inference = true;
        injected_now = 1u;
      } else if (ctx->active_target_op == op_index) {
        ctx->saved_original_byte = *ctx->active_addr;
        *ctx->active_addr ^= ctx->active_mask;
        ctx->injection_active = true;
        ctx->injected_this_inference = true;
        injected_now = 1u;
      }
    }
  }

  if (ctx->overhead_enabled) {
    const uint32_t exit_cycle = fi_read_cycles();
    ctx->overhead.before_op_cycles += (exit_cycle - entry_cycle);
    ctx->overhead.before_op_count  += 1u;
    ctx->overhead.active_op_hook_count += 1u;
    ctx->overhead.injections       += injected_now;
  }
}

void memory_fi_after_op(memory_fi_context_t* ctx, uint32_t op_index) {
  if (ctx == nullptr) {
    return;
  }

  const uint32_t entry_cycle = ctx->overhead_enabled ? fi_read_cycles() : 0u;

  do {
  /* Single-fault restore uses pre-materialized metadata.  Wildcard
   * faults keep their previous lifetime and are restored at inference end. */
#ifdef MULTI_FI_MAX_COUNT
    if (ctx->group_fault_count > 0) {
      for (int8_t i = static_cast<int8_t>(ctx->group_fault_count) - 1; i >= 0; --i) {
        if (ctx->group_injection_active[i] && ctx->group_faults[i]->op_index == op_index) {
          uint8_t* addr = memory_fi_resolve_addr(ctx, ctx->group_faults[i]);
          *addr = ctx->saved_group_bytes[i];
          ctx->group_injection_active[i] = false;
        }
      }
      break;
    }
#endif

    if (!ctx->injection_active) {
      break;
    }

    if (ctx->active_fault == nullptr || ctx->active_addr == nullptr ||
        ctx->active_is_wildcard) {
      break;
    }
    if (ctx->active_target_op != op_index) {
      break;
    }

    *ctx->active_addr = ctx->saved_original_byte;
    ctx->injection_active = false;
  } while (0);

  if (ctx->overhead_enabled) {
    const uint32_t exit_cycle = fi_read_cycles();
    ctx->overhead.after_op_cycles += static_cast<uint32_t>(exit_cycle - entry_cycle);
    ctx->overhead.after_op_count  += 1u;
  }
}

void memory_fi_set_profiling(memory_fi_context_t* ctx, bool enabled) {
  if (ctx == nullptr) {
    return;
  }
#if defined(FI_CAMPAIGN_MEMORY)
  g_microfi_profile_all_ops = enabled ? 1u : 0u;
#else
  (void)enabled;
#endif
  memory_fi_publish_active_op_mask(ctx);
}

void memory_fi_set_overhead_enabled(memory_fi_context_t* ctx, bool enabled) {
  if (ctx != nullptr) {
    ctx->overhead_enabled = enabled;
#if defined(FI_CAMPAIGN_MEMORY)
    g_microfi_count_skipped_hooks = enabled ? 1u : 0u;
#endif
  }
}

void memory_fi_reset_overhead(memory_fi_context_t* ctx) {
  if (ctx != nullptr) {
    ctx->overhead = {};
    memory_fi_reset_callsite_counters();
  }
}

void memory_fi_get_overhead(const memory_fi_context_t* ctx,
                            memory_fi_overhead_t* out) {
  if (ctx == nullptr || out == nullptr) {
    return;
  }
  *out = ctx->overhead;
#if defined(FI_CAMPAIGN_MEMORY)
  out->skipped_before_op_count = g_microfi_skipped_before_op_count;
  out->skipped_after_op_count = g_microfi_skipped_after_op_count;
#endif
}

#ifdef MULTI_FI_MAX_COUNT
void memory_fi_set_group(memory_fi_context_t* ctx,
                         const memory_fault_t* pool,
                         const uint16_t* pool_indices,
                         uint8_t count) {
  if (ctx == nullptr) {
    return;
  }
  uint8_t n = count;
  if (n > MULTI_FI_MAX_COUNT) {
    n = MULTI_FI_MAX_COUNT;
  }
  for (uint8_t i = 0; i < n; ++i) {
    ctx->group_faults[i] = &pool[pool_indices[i]];
    ctx->saved_group_bytes[i] = 0;
    ctx->group_injection_active[i] = false;
  }
  ctx->group_fault_count = n;
  memory_fi_publish_active_op_mask(ctx);
}

void memory_fi_clear_group(memory_fi_context_t* ctx) {
  if (ctx == nullptr) {
    return;
  }
  ctx->group_fault_count = 0;
  memory_fi_publish_active_op_mask(ctx);
}
#endif

// ---- Operator hooks for memory campaign ---------------------------------
#if defined(FI_CAMPAIGN_MEMORY)

/* Call-site gating in micro_interpreter_graph.cc reads the active-op
 * mask published by memory_fi_begin_inference(). */
/* `flatten` collapses the wrapper + memory_fi_*_op into one leaf
 * function per hook, matching the register-FI optimization. */
extern "C" __attribute__((hot, flatten)) void microfi_before_op(int op_index) {
#if defined(FI_FAULT_TRANSIENT)
  if (g_memory_ctx != nullptr && op_index >= 0) {
    memory_fi_before_op(g_memory_ctx, static_cast<uint32_t>(op_index));
  }
#else
  (void)op_index;
#endif
}

extern "C" __attribute__((hot, flatten)) void microfi_after_op(int op_index) {
#if defined(FI_FAULT_TRANSIENT)
  if (g_memory_ctx != nullptr && op_index >= 0) {
    memory_fi_after_op(g_memory_ctx, static_cast<uint32_t>(op_index));
  }
#else
  (void)op_index;
#endif
}

#endif  // FI_CAMPAIGN_MEMORY
