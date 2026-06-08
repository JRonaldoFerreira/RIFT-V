#include "memory_fi.h"

#include "fi_cycles.h"

#if defined(FI_CAMPAIGN_MEMORY)
static memory_fi_context_t* g_memory_ctx = nullptr;
#endif

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
#endif
}

void memory_fi_begin_inference(memory_fi_context_t* ctx, int32_t fault_index) {
  if (ctx == nullptr) {
    return;
  }
  ctx->active_fault_index = fault_index;
  
  ctx->injected_this_inference = false;
  ctx->saved_original_byte = 0u;
  ctx->injection_active = false;
}

void memory_fi_end_inference(memory_fi_context_t* ctx) {
  if (ctx == nullptr) {
    return;
  }

  if (ctx->injection_active && ctx->active_fault_index >= 0 &&
      static_cast<uint32_t>(ctx->active_fault_index) < ctx->fault_count) {
    const memory_fault_t* fault = &ctx->fault_list[ctx->active_fault_index];
    uint8_t* addr = memory_fi_resolve_addr(ctx, fault);
    *addr = ctx->saved_original_byte;
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
  ctx->injected_this_inference = false;
  ctx->injection_active = false;
}

const memory_fault_t* memory_fi_current_fault(const memory_fi_context_t* ctx) {
  if (ctx == nullptr || ctx->active_fault_index < 0 ||
      static_cast<uint32_t>(ctx->active_fault_index) >= ctx->fault_count) {
    return nullptr;
  }
  return &ctx->fault_list[ctx->active_fault_index];
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

void memory_fi_before_op(memory_fi_context_t* ctx, uint32_t op_index) {
  if (ctx == nullptr) {
    return;
  }

  const uint32_t entry_cycle = ctx->overhead_enabled ? fi_read_cycles() : 0u;
  uint32_t injected_now = 0u;

  do {
  /* FAST PATH — same rationale as register_fi: during a single
   * inference only one operator carries a scheduled fault, so the
   * remaining ~15 hook calls must be as cheap as possible to keep the
   * per-injection overhead comparable to microFI. */
  {
    bool has_work = false;
#ifdef MULTI_FI_MAX_COUNT
    if (ctx->group_fault_count > 0u) {
      has_work = true;
    }
#endif
    if (!has_work) {
      const int32_t fi = ctx->active_fault_index;
      if (fi < 0 || static_cast<uint32_t>(fi) >= ctx->fault_count) {
        break;
      }
      const uint32_t target_op = ctx->fault_list[fi].op_index;
      if (target_op != op_index && target_op != FI_OP_INDEX_WILDCARD) {
        break;
      }
    }
  }

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
    break;
  }
#endif

  if (ctx->active_fault_index < 0 ||
      static_cast<uint32_t>(ctx->active_fault_index) >= ctx->fault_count) {
    break;
  }
  const memory_fault_t* fault = &ctx->fault_list[ctx->active_fault_index];

  if (fault->op_index == FI_OP_INDEX_WILDCARD) {
    uint8_t* addr = memory_fi_resolve_addr(ctx, fault);
    if (!ctx->injection_active) {
      ctx->saved_original_byte = *addr;
      ctx->injection_active = true;
    }
    _force_stuck_bit(addr, ctx->saved_original_byte, fault->bit_index);
    ctx->injected_this_inference = true;
    injected_now += 1u;
    break;
  }

  if (fault->op_index != op_index) {
    break;
  }

  uint8_t* addr = memory_fi_resolve_addr(ctx, fault);
  ctx->saved_original_byte = *addr;
  *addr ^= static_cast<uint8_t>(1u << fault->bit_index);
  ctx->injection_active = true;
  ctx->injected_this_inference = true;
  injected_now += 1u;
  } while (0);

  if (ctx->overhead_enabled) {
    const uint32_t exit_cycle = fi_read_cycles();
    ctx->overhead.before_op_cycles += static_cast<uint32_t>(exit_cycle - entry_cycle);
    ctx->overhead.before_op_count  += 1u;
    ctx->overhead.injections       += injected_now;
  }
}

void memory_fi_after_op(memory_fi_context_t* ctx, uint32_t op_index) {
  if (ctx == nullptr) {
    return;
  }

  const uint32_t entry_cycle = ctx->overhead_enabled ? fi_read_cycles() : 0u;

  do {
  /* FAST PATH — bail out early when nothing was injected this op.
   * The single-fault path is keyed on `injection_active`, set by
   * before_op only when this op matched.  The multi-fault path needs
   * to check the group bitmap. */
  {
    bool has_work = false;
#ifdef MULTI_FI_MAX_COUNT
    if (ctx->group_fault_count > 0u) {
      has_work = true;
    }
#endif
    if (!has_work && !ctx->injection_active) {
      break;
    }
  }

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

    if (ctx->active_fault_index < 0 ||
        static_cast<uint32_t>(ctx->active_fault_index) >= ctx->fault_count) {
      break;
    }
    const memory_fault_t* fault = &ctx->fault_list[ctx->active_fault_index];
    if (fault->op_index != op_index) {
      break;
    }

    uint8_t* addr = memory_fi_resolve_addr(ctx, fault);
    *addr = ctx->saved_original_byte;
    ctx->injection_active = false;
  } while (0);

  if (ctx->overhead_enabled) {
    const uint32_t exit_cycle = fi_read_cycles();
    ctx->overhead.after_op_cycles += static_cast<uint32_t>(exit_cycle - entry_cycle);
    ctx->overhead.after_op_count  += 1u;
  }
}

void memory_fi_set_overhead_enabled(memory_fi_context_t* ctx, bool enabled) {
  if (ctx != nullptr) {
    ctx->overhead_enabled = enabled;
  }
}

void memory_fi_reset_overhead(memory_fi_context_t* ctx) {
  if (ctx != nullptr) {
    ctx->overhead = {};
  }
}

void memory_fi_get_overhead(const memory_fi_context_t* ctx,
                            memory_fi_overhead_t* out) {
  if (ctx == nullptr || out == nullptr) {
    return;
  }
  *out = ctx->overhead;
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
}

void memory_fi_clear_group(memory_fi_context_t* ctx) {
  if (ctx == nullptr) {
    return;
  }
  ctx->group_fault_count = 0;
}
#endif

// ---- Operator hooks for memory campaign ---------------------------------
#if defined(FI_CAMPAIGN_MEMORY)

extern "C" void microfi_before_op(int op_index) {
#if defined(FI_FAULT_TRANSIENT)
  if (g_memory_ctx != nullptr && op_index >= 0) {
    memory_fi_before_op(g_memory_ctx, static_cast<uint32_t>(op_index));
  }
#else
  (void)op_index;
#endif
}

extern "C" void microfi_after_op(int op_index) {
#if defined(FI_FAULT_TRANSIENT)
  if (g_memory_ctx != nullptr && op_index >= 0) {
    memory_fi_after_op(g_memory_ctx, static_cast<uint32_t>(op_index));
  }
#else
  (void)op_index;
#endif
}

#endif  // FI_CAMPAIGN_MEMORY
