#include "register_fi.h"

#if defined(FI_CAMPAIGN_REGISTER)

#include <stddef.h>
#ifdef REGISTER_FI_RUNTIME_CHECK
#include <stdio.h>
#endif

#include "esp_attr.h"
#include "esp_timer.h"
#include "esp_check.h"
#ifndef REGISTER_FI_USE_SAVED_SP_MICROFI
#include "riscv/rvruntime-frames.h"
#endif
#include "freertos/FreeRTOS.h"

#ifdef REGISTER_FI_USE_SAVED_SP_MICROFI
/* MicroFI approach: the SP saved by REGISTER_FI_SAVE_SP() at each
 * instrumented inference function.  Initialised to 0 (invalid). */
volatile uintptr_t g_register_fi_saved_sp = 0;
#endif

#ifndef REGISTER_FI_USE_SAVED_SP_MICROFI
/* Trampoline globals — written by the ISR, read by _fi_trampoline (asm).
 * sp and gp are NOT restored from RvExcFrame by restore_general_regs
 * in vectors.S; we redirect mepc to an IRAM trampoline that XORs the
 * live register after mret. */
extern "C" {
extern void _fi_trampoline(void);
volatile uint32_t g_fi_trampoline_mepc    = 0;
volatile uint32_t g_fi_trampoline_sp_mask = 0;
volatile uint32_t g_fi_trampoline_gp_mask = 0;
}
#endif  /* !REGISTER_FI_USE_SAVED_SP_MICROFI */

namespace {

#ifndef REGISTER_FI_USE_SAVED_SP_MICROFI
/* ---- Excframe path (DEFAULT) ---- */
/* FreeRTOS internals used to access the interrupt exception frame.
 * When a timer interrupt fires during operator execution, vectors.S
 * saves ALL 31 GPRs into an RvExcFrame on the task stack, then
 * rtos_int_enter stores that stack pointer into
 * pxCurrentTCBs[0]->pxTopOfStack before switching to the ISR stack.
 * Our ESP_TIMER_ISR callback reads this pointer to get the real
 * RvExcFrame with architecturally-named register fields. */
struct minimal_tcb_t {
  StackType_t* pxTopOfStack;
};
extern "C" void* pxCurrentTCBs[];
#endif  /* !REGISTER_FI_USE_SAVED_SP_MICROFI */

static register_fi_context_t* g_ctx = nullptr;

#ifdef REGISTER_FI_USE_SAVED_SP_MICROFI
/* ---- MicroFI path (opt-in) ---- */
/* Stack-slot descriptors: register_id is a word-index into the function
 * stack frame.  The names are informational; the actual CPU register at
 * each slot depends on the compiler prologue.  The paper simply accesses
 * "the N-th saved 32-bit word on the stack". */
static const char* kSlotNames[REGISTER_FI_MAX_STACK_SLOTS] = {
    "slot_0",  "slot_1",  "slot_2",  "slot_3",
    "slot_4",  "slot_5",  "slot_6",  "slot_7",
    "slot_8",  "slot_9",  "slot_10", "slot_11",
    "slot_12", "slot_13", "slot_14", "slot_15",
    "slot_16", "slot_17", "slot_18", "slot_19",
    "slot_20", "slot_21", "slot_22", "slot_23",
    "slot_24", "slot_25", "slot_26", "slot_27",
    "slot_28", "slot_29", "slot_30", "slot_31",
};

/* MicroFI path does not need the mepc trampoline — all slots are
 * accessible through the saved stack pointer and directly modifiable. */
IRAM_ATTR static inline void do_bitflip(uint32_t* /*frame*/, uint32_t* target,
                                         uint8_t /*reg_id*/, uint32_t mask) {
    *target ^= mask;
}
IRAM_ATTR static inline void reset_trampoline(void) {}

#else
/* ---- Excframe path (DEFAULT) ---- */
/* Uses RvExcFrame offsets via pxCurrentTCBs[0]->pxTopOfStack.
 * During the ISR, this pointer holds the interrupt exception frame
 * (set by rtos_int_enter), containing all 31 GPRs of the interrupted
 * code.  Fields are named (ra, s0, a0, etc.) — 100% correct. */
struct RegisterDescriptor {
  const char* name;
  uint32_t offset;
};

#define REG_DESC(field) {#field, static_cast<uint32_t>(offsetof(RvExcFrame, field))}

static const RegisterDescriptor kRegisterDescriptors[] = {
    REG_DESC(ra), REG_DESC(sp), REG_DESC(gp), REG_DESC(tp),
    REG_DESC(t0), REG_DESC(t1), REG_DESC(t2), REG_DESC(s0),
    REG_DESC(s1), REG_DESC(a0), REG_DESC(a1), REG_DESC(a2),
    REG_DESC(a3), REG_DESC(a4), REG_DESC(a5), REG_DESC(a6),
    REG_DESC(a7), REG_DESC(s2), REG_DESC(s3), REG_DESC(s4),
    REG_DESC(s5), REG_DESC(s6), REG_DESC(s7), REG_DESC(s8),
    REG_DESC(s9), REG_DESC(s10), REG_DESC(s11), REG_DESC(t3),
    REG_DESC(t4), REG_DESC(t5), REG_DESC(t6),
};

#undef REG_DESC

constexpr uint8_t kRegIdSp = 1;
constexpr uint8_t kRegIdGp = 2;

/* Apply a bit-flip.  For most registers this directly XORs the
 * RvExcFrame field.  For sp/gp (not restored by restore_general_regs)
 * the mask is accumulated into trampoline globals and mepc is
 * redirected so the live register is XORed after mret. */
IRAM_ATTR static inline void do_bitflip(
        RvExcFrame* frame, uint32_t* target,
        uint8_t reg_id, uint32_t mask) {
    if (reg_id == kRegIdSp) {
        g_fi_trampoline_sp_mask ^= mask;
    } else if (reg_id == kRegIdGp) {
        g_fi_trampoline_gp_mask ^= mask;
    } else {
        *target ^= mask;
        return;
    }
    /* Redirect mepc to the trampoline (once per ISR invocation). */
    const long tramp_addr = (long)(uintptr_t)_fi_trampoline;
    if (frame->mepc != tramp_addr) {
        g_fi_trampoline_mepc = static_cast<uint32_t>(frame->mepc);
        frame->mepc = tramp_addr;
    }
}

/* Reset trampoline masks — called at ISR entry. */
IRAM_ATTR static inline void reset_trampoline(void) {
    g_fi_trampoline_sp_mask = 0;
    g_fi_trampoline_gp_mask = 0;
}

#endif  /* REGISTER_FI_USE_SAVED_SP_MICROFI */

#ifdef REGISTER_FI_RUNTIME_CHECK

#ifndef REGCHECK_TARGET_REG_ID
#define REGCHECK_TARGET_REG_ID 0xFFu
#endif
#ifndef REGCHECK_TARGET_BIT
#define REGCHECK_TARGET_BIT 0xFFu
#endif

static struct {
  volatile bool valid;
  uint8_t register_id;
  uint8_t bit_index;
  uint32_t active_op_index;
  uint32_t delay_us;
#ifdef REGISTER_FI_USE_SAVED_SP_MICROFI
  uintptr_t saved_sp;     /* SP captured by REGISTER_FI_SAVE_SP() */
#else
  void* tcb_raw;          /* legacy: pxCurrentTCBs[0] */
  void* top_of_stack;     /* legacy: tcb->pxTopOfStack */
#endif
  void* frame_ptr;
  void* slot_ptr;
  void* expected_ptr;
  uint32_t value_before;
  uint32_t value_after;
} g_regcheck = {};

#ifdef REGISTER_FI_USE_SAVED_SP_MICROFI
IRAM_ATTR static uint32_t* regcheck_expected_ptr(uint32_t* stack_base, uint8_t slot_index) {
  if (stack_base == nullptr || slot_index >= REGISTER_FI_MAX_STACK_SLOTS) return nullptr;
  return &stack_base[slot_index];
}
#else
IRAM_ATTR static uint32_t* regcheck_expected_ptr(RvExcFrame* frame, uint8_t reg_id) {
  switch (reg_id) {
    case 0:  return reinterpret_cast<uint32_t*>(&frame->ra);
    case 1:  return reinterpret_cast<uint32_t*>(&frame->sp);
    case 2:  return reinterpret_cast<uint32_t*>(&frame->gp);
    case 3:  return reinterpret_cast<uint32_t*>(&frame->tp);
    case 4:  return reinterpret_cast<uint32_t*>(&frame->t0);
    case 5:  return reinterpret_cast<uint32_t*>(&frame->t1);
    case 6:  return reinterpret_cast<uint32_t*>(&frame->t2);
    case 7:  return reinterpret_cast<uint32_t*>(&frame->s0);
    case 8:  return reinterpret_cast<uint32_t*>(&frame->s1);
    case 9:  return reinterpret_cast<uint32_t*>(&frame->a0);
    case 10: return reinterpret_cast<uint32_t*>(&frame->a1);
    case 11: return reinterpret_cast<uint32_t*>(&frame->a2);
    case 12: return reinterpret_cast<uint32_t*>(&frame->a3);
    case 13: return reinterpret_cast<uint32_t*>(&frame->a4);
    case 14: return reinterpret_cast<uint32_t*>(&frame->a5);
    case 15: return reinterpret_cast<uint32_t*>(&frame->a6);
    case 16: return reinterpret_cast<uint32_t*>(&frame->a7);
    case 17: return reinterpret_cast<uint32_t*>(&frame->s2);
    case 18: return reinterpret_cast<uint32_t*>(&frame->s3);
    case 19: return reinterpret_cast<uint32_t*>(&frame->s4);
    case 20: return reinterpret_cast<uint32_t*>(&frame->s5);
    case 21: return reinterpret_cast<uint32_t*>(&frame->s6);
    case 22: return reinterpret_cast<uint32_t*>(&frame->s7);
    case 23: return reinterpret_cast<uint32_t*>(&frame->s8);
    case 24: return reinterpret_cast<uint32_t*>(&frame->s9);
    case 25: return reinterpret_cast<uint32_t*>(&frame->s10);
    case 26: return reinterpret_cast<uint32_t*>(&frame->s11);
    case 27: return reinterpret_cast<uint32_t*>(&frame->t3);
    case 28: return reinterpret_cast<uint32_t*>(&frame->t4);
    case 29: return reinterpret_cast<uint32_t*>(&frame->t5);
    case 30: return reinterpret_cast<uint32_t*>(&frame->t6);
    default: return nullptr;
  }
}
#endif  /* REGISTER_FI_USE_SAVED_SP_MICROFI */

#endif  // REGISTER_FI_RUNTIME_CHECK

constexpr uint32_t kMaxTrackedOps = sizeof(g_ctx->op_profiles) / sizeof(g_ctx->op_profiles[0]);

const register_fault_t* current_fault(const register_fi_context_t* ctx) {
  if (ctx == nullptr || ctx->active_fault_index < 0) {
    return nullptr;
  }
  const uint32_t idx = static_cast<uint32_t>(ctx->active_fault_index);
  if (idx >= ctx->fault_count) {
    return nullptr;
  }
  return &ctx->fault_list[idx];
}

#ifdef REGISTER_FI_USE_SAVED_SP_MICROFI
/* MicroFI (opt-in): register_id is a word-slot index into the function's
 * stack frame.  stack_base points to the first word at the saved SP.
 * This accesses the real stack of the last instrumented function, exactly
 * as described in the MicroFI paper. */
IRAM_ATTR uint32_t* resolve_register_slot(uint32_t* stack_base, uint8_t slot_index) {
  if (stack_base == nullptr || slot_index >= REGISTER_FI_MAX_STACK_SLOTS) {
    return nullptr;
  }
  return &stack_base[slot_index];
}
#else
/* Excframe (DEFAULT): register_id maps to a named field inside the
 * RvExcFrame interrupt exception frame.  During the ISR,
 * pxCurrentTCBs[0]->pxTopOfStack points to this frame. */
IRAM_ATTR uint32_t* resolve_register_slot(RvExcFrame* frame, uint8_t register_id) {
  if (frame == nullptr || register_id >= (sizeof(kRegisterDescriptors) / sizeof(kRegisterDescriptors[0]))) {
    return nullptr;
  }
  const uint32_t offset = kRegisterDescriptors[register_id].offset;
  return reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(frame) + offset);
}
#endif  /* REGISTER_FI_USE_SAVED_SP_MICROFI */

uint32_t compute_delay_us(register_fi_context_t* ctx, const register_fault_t* fault) {
  if (fault == nullptr) {
    return 0;
  }
  const uint32_t op_index = fault->op_index;
  if (op_index >= kMaxTrackedOps) {
    return 0;
  }
  const uint32_t mean_us = ctx->op_profiles[op_index].mean_duration_us;
  if (mean_us == 0) {
    return 0;
  }
  if (mean_us <= ctx->min_timer_delay_us + ctx->slot_guard_us) {
    return 0;
  }

  const uint32_t slot_count = fault->time_slot_count == 0 ? 1u : fault->time_slot_count;
  const float center = (static_cast<float>(fault->time_slot_index) + 0.5f) / static_cast<float>(slot_count);
  uint32_t delay_us = static_cast<uint32_t>(static_cast<float>(mean_us) * center);
  if (delay_us < ctx->min_timer_delay_us) {
    delay_us = ctx->min_timer_delay_us;
  }
  const uint32_t upper_bound = (mean_us > ctx->slot_guard_us) ? (mean_us - ctx->slot_guard_us) : mean_us;
  if (delay_us > upper_bound) {
    delay_us = upper_bound;
  }
  return delay_us;
}

IRAM_ATTR void register_timer_isr(void* arg) {
  auto* ctx = static_cast<register_fi_context_t*>(arg);
  if (ctx == nullptr || !ctx->op_active) {
    return;
  }

#ifdef REGISTER_FI_USE_SAVED_SP_MICROFI
  /* ------------------------------------------------------------------ */
  /* MicroFI approach (opt-in) — faithful to the paper.                  */
  /*                                                                     */
  /* The saved SP (captured via REGISTER_FI_SAVE_SP() at each            */
  /* instrumented function entry) points to the base of the function's   */
  /* stack frame.  We treat this as a uint32_t[] array.                  */
  /*                                                                     */
  /* NOTE: This path has known limitations — see register_fi.h header.   */
  /* ------------------------------------------------------------------ */
  const uintptr_t saved_sp = g_register_fi_saved_sp;
  if (saved_sp == 0u) {
    ctx->last_injection_status = REGISTER_FI_INVALID_FRAME;
    ctx->timer_armed = false;
    return;
  }
  auto* frame = reinterpret_cast<uint32_t*>(saved_sp);
#else
  /* ------------------------------------------------------------------ */
  /* Excframe path (DEFAULT) — interrupt exception frame.                */
  /*                                                                     */
  /* pxCurrentTCBs[0]->pxTopOfStack points to the RvExcFrame that        */
  /* _interrupt_handler (vectors.S) saved on the task stack when the      */
  /* timer interrupt fired.  rtos_int_enter stored it before switching    */
  /* to the ISR stack.  Contains ALL 31 GPRs of the interrupted code.    */
  /* Modifying a field here (e.g. frame->s3) means the corrupted value   */
  /* goes directly into the CPU register when mret executes.             */
  /* ------------------------------------------------------------------ */
  void* tcb_raw = pxCurrentTCBs[0];
  if (tcb_raw == nullptr) {
    ctx->last_injection_status = REGISTER_FI_INVALID_FRAME;
    ctx->timer_armed = false;
    return;
  }

  auto* tcb = reinterpret_cast<minimal_tcb_t*>(tcb_raw);
  if (tcb->pxTopOfStack == nullptr) {
    ctx->last_injection_status = REGISTER_FI_INVALID_FRAME;
    ctx->timer_armed = false;
    return;
  }

  auto* frame = reinterpret_cast<RvExcFrame*>(tcb->pxTopOfStack);
#endif  /* REGISTER_FI_USE_SAVED_SP_MICROFI */

  reset_trampoline();

#ifdef MULTI_FI_MAX_COUNT
  if (ctx->op_fault_count > 0) {
    // Inject only the current batch: faults whose delay equals the
    // delay that was used to arm this timer firing.
    const uint8_t start = ctx->op_fault_next_idx;
    const uint32_t current_delay = ctx->op_fault_delays[start];
    uint8_t end = start;
    while (end < ctx->op_fault_count && ctx->op_fault_delays[end] == current_delay) {
      ++end;
    }

    for (uint8_t i = start; i < end; ++i) {
      const volatile register_fault_t* f = ctx->op_faults[i];
      uint32_t* target = resolve_register_slot(frame, f->register_id);
      if (target != nullptr) {
        uint32_t mask = (1u << f->bit_index);
#if !defined(FI_FAULT_TRANSIENT)
        uint8_t gi = 0;
        for (uint8_t g = 0; g < ctx->group_fault_count; ++g) {
          if (ctx->group_faults[g] == f) { gi = g; break; }
        }
        if (!ctx->permanent_injected[gi]) {
          ctx->permanent_original_bits[gi] = (*target & mask) != 0;
          ctx->permanent_injected[gi] = true;
          do_bitflip(frame, target, f->register_id, mask);
        } else {
          bool current_bit = (*target & mask) != 0;
          if (current_bit == ctx->permanent_original_bits[gi]) {
            do_bitflip(frame, target, f->register_id, mask);
          }
        }
#else
        do_bitflip(frame, target, f->register_id, mask);
#endif
      }
    }

    ctx->op_fault_next_idx = end;
    ctx->last_injection_status = REGISTER_FI_INJECTED;

    // If more fault batches remain, re-arm the timer for the next
    // batch using the delta between delays.
    if (end < ctx->op_fault_count) {
      uint32_t next_delay = ctx->op_fault_delays[end];
      uint32_t delta = (next_delay > current_delay) ? (next_delay - current_delay) : 1u;
      ctx->current_delay_us = next_delay;
      auto* timer = static_cast<esp_timer_handle_t>(ctx->timer_handle);
      if (esp_timer_start_once(timer, delta) == ESP_OK) {
        ctx->timer_armed = true;
        ctx->last_injection_status = REGISTER_FI_ARMED;
      } else {
        ctx->timer_armed = false;
      }
    } else {
      ctx->timer_armed = false;
    }
    return;
  }
#endif

  const register_fault_t* fault = current_fault(ctx);
  if (fault == nullptr) {
    ctx->last_injection_status = REGISTER_FI_INVALID_FRAME;
    ctx->timer_armed = false;
    return;
  }

  uint32_t* target = resolve_register_slot(frame, fault->register_id);
  if (target == nullptr) {
    ctx->last_injection_status = REGISTER_FI_INVALID_REGISTER;
    ctx->timer_armed = false;
    return;
  }

#ifdef REGISTER_FI_RUNTIME_CHECK
  if (!g_regcheck.valid) {
    const bool reg_ok = (REGCHECK_TARGET_REG_ID == 0xFFu) ||
                        (fault->register_id == REGCHECK_TARGET_REG_ID);
    const bool bit_ok = (REGCHECK_TARGET_BIT == 0xFFu) ||
                        (fault->bit_index == REGCHECK_TARGET_BIT);
    if (reg_ok && bit_ok) {
      g_regcheck.register_id = fault->register_id;
      g_regcheck.bit_index = fault->bit_index;
      g_regcheck.active_op_index = static_cast<uint32_t>(ctx->active_op_index);
      g_regcheck.delay_us = ctx->current_delay_us;
#ifdef REGISTER_FI_USE_SAVED_SP_MICROFI
      g_regcheck.saved_sp = saved_sp;
#else
      g_regcheck.tcb_raw = tcb_raw;
      g_regcheck.top_of_stack = reinterpret_cast<void*>(tcb->pxTopOfStack);
#endif
      g_regcheck.frame_ptr = frame;
      g_regcheck.slot_ptr = target;
      g_regcheck.expected_ptr = regcheck_expected_ptr(frame, fault->register_id);
      g_regcheck.value_before = *target;
    }
  }
#endif

#if !defined(FI_FAULT_TRANSIENT)
  uint32_t mask = (1u << fault->bit_index);
  if (!ctx->permanent_active) {
    ctx->permanent_original_bit = (*target & mask) != 0;
    ctx->permanent_active = true;
    do_bitflip(frame, target, fault->register_id, mask);
  } else {
    bool current = (*target & mask) != 0;
    if (current == ctx->permanent_original_bit) {
      do_bitflip(frame, target, fault->register_id, mask);
    }
  }
#else
  do_bitflip(frame, target, fault->register_id, (1u << fault->bit_index));
#endif

#ifdef REGISTER_FI_RUNTIME_CHECK
  if (!g_regcheck.valid && g_regcheck.frame_ptr != nullptr) {
    g_regcheck.value_after = *target;
    g_regcheck.valid = true;
  }
#endif

  ctx->last_injection_status = REGISTER_FI_INJECTED;
  ctx->timer_armed = false;
}

}  // namespace

void register_fi_init(register_fi_context_t* ctx,
                      const register_fault_t* fault_list,
                      uint32_t fault_count,
                      uint32_t min_timer_delay_us,
                      uint32_t slot_guard_us) {
  if (ctx == nullptr) {
    return;
  }
  ctx->fault_list = fault_list;
  ctx->fault_count = fault_count;
  ctx->active_fault_index = -1;
  ctx->active_op_index = -1;
  ctx->op_active = false;
  ctx->timer_armed = false;
  ctx->current_delay_us = 0;
  ctx->last_injection_status = REGISTER_FI_IDLE;
  ctx->profiling_enabled = false;
  for (uint32_t i = 0; i < kMaxTrackedOps; ++i) {
    ctx->op_start_time_us[i] = 0;
    ctx->op_profiles[i] = {0u, 0u, 0u};
  }
  ctx->min_timer_delay_us = min_timer_delay_us;
  ctx->slot_guard_us = slot_guard_us;
  ctx->timer_handle = nullptr;
  ctx->permanent_active = false;
  ctx->permanent_original_bit = false;

#ifdef REGISTER_FI_USE_SAVED_SP_MICROFI
  g_register_fi_saved_sp = 0;
#endif

#ifdef REGISTER_FI_RUNTIME_CHECK
  g_regcheck = {};
#endif

#ifdef MULTI_FI_MAX_COUNT
  ctx->group_fault_count = 0;
  ctx->op_fault_count = 0;
  ctx->op_fault_next_idx = 0;
  for (uint8_t i = 0; i < MULTI_FI_MAX_COUNT; ++i) {
    ctx->group_faults[i] = nullptr;
    ctx->op_faults[i] = nullptr;
    ctx->op_fault_delays[i] = 0;
    ctx->permanent_injected[i] = false;
    ctx->permanent_original_bits[i] = false;
  }
#endif

  const esp_timer_create_args_t timer_args = {
      .callback = &register_timer_isr,
      .arg = ctx,
      .dispatch_method = ESP_TIMER_ISR,
      .name = "registerfi",
      .skip_unhandled_events = false,
  };
  esp_timer_handle_t timer = nullptr;
  if (esp_timer_create(&timer_args, &timer) == ESP_OK) {
    ctx->timer_handle = timer;
  }
  g_ctx = ctx;
}

void register_fi_deinit(register_fi_context_t* ctx) {
  if (ctx == nullptr) {
    return;
  }
  if (ctx->timer_handle != nullptr) {
    auto* timer = static_cast<esp_timer_handle_t>(ctx->timer_handle);
    if (esp_timer_is_active(timer)) {
      esp_timer_stop(timer);
    }
    esp_timer_delete(timer);
    ctx->timer_handle = nullptr;
  }
  if (g_ctx == ctx) {
    g_ctx = nullptr;
  }
}

void register_fi_set_profiling(register_fi_context_t* ctx, bool enabled) {
  if (ctx != nullptr) {
    ctx->profiling_enabled = enabled;
  }
}

void register_fi_begin_inference(register_fi_context_t* ctx, int32_t fault_index) {
  if (ctx == nullptr) {
    return;
  }
  ctx->active_fault_index = fault_index;
  ctx->active_op_index = -1;
  ctx->op_active = false;
  ctx->timer_armed = false;
  ctx->current_delay_us = 0;
  ctx->last_injection_status = REGISTER_FI_IDLE;
  ctx->permanent_active = false;
  ctx->permanent_original_bit = false;
#ifdef REGISTER_FI_USE_SAVED_SP_MICROFI
  g_register_fi_saved_sp = 0;
#endif
#ifdef MULTI_FI_MAX_COUNT
  ctx->op_fault_next_idx = 0;
  for (uint8_t i = 0; i < MULTI_FI_MAX_COUNT; ++i) {
    ctx->op_fault_delays[i] = 0;
    ctx->permanent_injected[i] = false;
    ctx->permanent_original_bits[i] = false;
  }
#endif
}

void register_fi_end_inference(register_fi_context_t* ctx) {
  if (ctx == nullptr) {
    return;
  }
  if (ctx->timer_handle != nullptr && ctx->timer_armed) {
    auto* timer = static_cast<esp_timer_handle_t>(ctx->timer_handle);
    if (esp_timer_is_active(timer)) {
      esp_timer_stop(timer);
    }
  }
  if (ctx->last_injection_status == REGISTER_FI_ARMED) {
    ctx->last_injection_status = REGISTER_FI_TIMER_NOT_FIRED;
  }
  ctx->timer_armed = false;
  ctx->op_active = false;
  ctx->active_op_index = -1;
  ctx->active_fault_index = -1;
  ctx->current_delay_us = 0;
  ctx->permanent_active = false;
  ctx->permanent_original_bit = false;
}

void register_fi_before_op(register_fi_context_t* ctx, uint32_t op_index) {
  if (ctx == nullptr || op_index >= kMaxTrackedOps) {
    return;
  }

  if (ctx->profiling_enabled) {
    ctx->op_start_time_us[op_index] = static_cast<uint64_t>(esp_timer_get_time());
  }

#ifdef MULTI_FI_MAX_COUNT
  if (ctx->group_fault_count > 0) {
    ctx->op_fault_count = 0;
    bool has_faults = false;

    // Collect faults targeting this operator with their individual delays.
    for (uint8_t i = 0; i < ctx->group_fault_count; ++i) {
      const register_fault_t* f = ctx->group_faults[i];
      if (f->op_index == op_index) {
        uint8_t n = ctx->op_fault_count;
        ctx->op_faults[n] = f;
        ctx->op_fault_delays[n] = compute_delay_us(ctx, f);
        ctx->op_fault_count = n + 1;
        has_faults = true;
      }
    }

#if !defined(FI_FAULT_TRANSIENT)
    // For permanent mode, also re-inject faults that were already
    // activated on earlier operators (to sustain the corruption).
    for (uint8_t i = 0; i < ctx->group_fault_count; ++i) {
      if (ctx->permanent_injected[i] && ctx->group_faults[i]->op_index != op_index) {
        if (ctx->op_fault_count < MULTI_FI_MAX_COUNT) {
          uint8_t n = ctx->op_fault_count;
          ctx->op_faults[n] = ctx->group_faults[i];
          ctx->op_fault_delays[n] = ctx->min_timer_delay_us;
          ctx->op_fault_count = n + 1;
          has_faults = true;
        }
      }
    }
#endif

    if (!has_faults || ctx->timer_handle == nullptr) {
      return;
    }

    // Insertion-sort op_faults by delay (ascending) so the ISR can
    // fire them in chronological order.
    for (uint8_t i = 1; i < ctx->op_fault_count; ++i) {
      const volatile register_fault_t* tmpf = ctx->op_faults[i];
      uint32_t tmpd = ctx->op_fault_delays[i];
      int8_t j = static_cast<int8_t>(i) - 1;
      while (j >= 0 && ctx->op_fault_delays[j] > tmpd) {
        ctx->op_faults[j + 1] = ctx->op_faults[j];
        ctx->op_fault_delays[j + 1] = ctx->op_fault_delays[j];
        --j;
      }
      ctx->op_faults[j + 1] = tmpf;
      ctx->op_fault_delays[j + 1] = tmpd;
    }

    ctx->active_op_index = static_cast<int32_t>(op_index);
    ctx->op_active = true;
    ctx->op_fault_next_idx = 0;

    uint32_t first_delay = ctx->op_fault_delays[0];
    if (first_delay == 0u) {
      first_delay = ctx->min_timer_delay_us;
      if (first_delay == 0u) {
        ctx->last_injection_status = REGISTER_FI_TIMING_WINDOW_TOO_SHORT;
        return;
      }
    }

    ctx->current_delay_us = first_delay;
    auto* timer = static_cast<esp_timer_handle_t>(ctx->timer_handle);
    if (esp_timer_start_once(timer, first_delay) == ESP_OK) {
      ctx->timer_armed = true;
      ctx->last_injection_status = REGISTER_FI_ARMED;
    } else {
      ctx->last_injection_status = REGISTER_FI_TIMER_NOT_FIRED;
    }
    return;
  }
#endif

#if !defined(FI_FAULT_TRANSIENT)
  if (ctx->permanent_active && ctx->timer_handle != nullptr) {
    ctx->active_op_index = static_cast<int32_t>(op_index);
    ctx->op_active = true;
    ctx->current_delay_us = ctx->min_timer_delay_us;
    auto* timer = static_cast<esp_timer_handle_t>(ctx->timer_handle);
    if (esp_timer_start_once(timer, ctx->min_timer_delay_us) == ESP_OK) {
      ctx->timer_armed = true;
      ctx->last_injection_status = REGISTER_FI_ARMED;
    }
    return;
  }
#endif

  const register_fault_t* fault = current_fault(ctx);
  if (fault == nullptr || fault->op_index != op_index || ctx->timer_handle == nullptr) {
    return;
  }

  uint32_t delay_us = compute_delay_us(ctx, fault);
  ctx->active_op_index = static_cast<int32_t>(op_index);
  ctx->op_active = true;
  ctx->current_delay_us = delay_us;

  if (delay_us == 0u) {
    if (fault->time_slot_index == 0u && ctx->min_timer_delay_us > 0u) {
      delay_us = ctx->min_timer_delay_us;
      ctx->current_delay_us = delay_us;
    } else {
      ctx->last_injection_status = REGISTER_FI_TIMING_WINDOW_TOO_SHORT;
      return;
    }
  }

  auto* timer = static_cast<esp_timer_handle_t>(ctx->timer_handle);
  if (esp_timer_start_once(timer, delay_us) == ESP_OK) {
    ctx->timer_armed = true;
    ctx->last_injection_status = REGISTER_FI_ARMED;
  } else {
    ctx->last_injection_status = REGISTER_FI_TIMER_NOT_FIRED;
  }
}

void register_fi_after_op(register_fi_context_t* ctx, uint32_t op_index) {
  if (ctx == nullptr || op_index >= kMaxTrackedOps) {
    return;
  }

  if (ctx->profiling_enabled) {
    const uint64_t start_us = ctx->op_start_time_us[op_index];
    if (start_us != 0u) {
      const uint64_t elapsed = static_cast<uint64_t>(esp_timer_get_time()) - start_us;
      ctx->op_profiles[op_index].count += 1u;
      ctx->op_profiles[op_index].total_duration_us += elapsed;
      ctx->op_profiles[op_index].mean_duration_us =
          static_cast<uint32_t>(ctx->op_profiles[op_index].total_duration_us / ctx->op_profiles[op_index].count);
      ctx->op_start_time_us[op_index] = 0u;
    }
  }

  if (ctx->active_op_index != static_cast<int32_t>(op_index)) {
    return;
  }

#if !defined(FI_FAULT_TRANSIENT)
  if (ctx->permanent_active) {
    if (ctx->timer_handle != nullptr && ctx->timer_armed) {
      auto* timer = static_cast<esp_timer_handle_t>(ctx->timer_handle);
      if (esp_timer_is_active(timer)) {
        esp_timer_stop(timer);
      }
      ctx->timer_armed = false;
    }
    ctx->op_active = false;
    ctx->active_op_index = -1;
#ifdef MULTI_FI_MAX_COUNT
    ctx->op_fault_count = 0;
    ctx->op_fault_next_idx = 0;
#endif
    return;
  }
#endif

  if (ctx->timer_handle != nullptr && ctx->timer_armed) {
    auto* timer = static_cast<esp_timer_handle_t>(ctx->timer_handle);
    if (esp_timer_is_active(timer)) {
      esp_timer_stop(timer);
    }
    ctx->timer_armed = false;
    if (ctx->last_injection_status == REGISTER_FI_ARMED) {
      ctx->last_injection_status = REGISTER_FI_TIMER_NOT_FIRED;
    }
  }

  ctx->op_active = false;
  ctx->active_op_index = -1;
#ifdef MULTI_FI_MAX_COUNT
  ctx->op_fault_count = 0;
  ctx->op_fault_next_idx = 0;
#endif
}

uint32_t register_fi_get_mean_duration_us(const register_fi_context_t* ctx, uint32_t op_index) {
  if (ctx == nullptr || op_index >= kMaxTrackedOps) {
    return 0u;
  }
  return ctx->op_profiles[op_index].mean_duration_us;
}

uint32_t register_fi_get_current_delay_us(const register_fi_context_t* ctx) {
  return ctx == nullptr ? 0u : ctx->current_delay_us;
}

register_fi_injection_status_t register_fi_get_last_status(const register_fi_context_t* ctx) {
  return ctx == nullptr ? REGISTER_FI_IDLE : ctx->last_injection_status;
}

const char* register_fi_status_name(register_fi_injection_status_t status) {
  switch (status) {
    case REGISTER_FI_IDLE:
      return "idle";
    case REGISTER_FI_ARMED:
      return "armed";
    case REGISTER_FI_INJECTED:
      return "injected";
    case REGISTER_FI_TIMER_NOT_FIRED:
      return "timer_not_fired";
    case REGISTER_FI_TIMING_WINDOW_TOO_SHORT:
      return "timing_window_too_short";
    case REGISTER_FI_INVALID_FRAME:
      return "invalid_frame";
    case REGISTER_FI_INVALID_REGISTER:
      return "invalid_register";
    default:
      return "unknown";
  }
}

const char* register_fi_register_name(uint8_t register_id) {
#ifdef REGISTER_FI_USE_SAVED_SP_MICROFI
  if (register_id >= REGISTER_FI_MAX_STACK_SLOTS) {
    return "invalid";
  }
  return kSlotNames[register_id];
#else
  if (register_id >= (sizeof(kRegisterDescriptors) / sizeof(kRegisterDescriptors[0]))) {
    return "invalid";
  }
  return kRegisterDescriptors[register_id].name;
#endif
}

#ifdef REGISTER_FI_RUNTIME_CHECK
void register_fi_runtime_check_print(void) {
  if (!g_regcheck.valid) {
    return;
  }
  const bool match = (g_regcheck.slot_ptr == g_regcheck.expected_ptr);
  const bool changed = (g_regcheck.value_before != g_regcheck.value_after);
#ifdef REGISTER_FI_USE_SAVED_SP_MICROFI
  printf("[REGMAP-RUNTIME-MICROFI] slot_id=%u(%s) bit=%u op=%lu delay=%lu "
         "saved_sp=0x%08lx stack_base=%p slot=%p expected=%p match=%s "
         "before=0x%08lx after=0x%08lx changed=%s\n",
         static_cast<unsigned>(g_regcheck.register_id),
         register_fi_register_name(g_regcheck.register_id),
         static_cast<unsigned>(g_regcheck.bit_index),
         static_cast<unsigned long>(g_regcheck.active_op_index),
         static_cast<unsigned long>(g_regcheck.delay_us),
         static_cast<unsigned long>(g_regcheck.saved_sp),
         g_regcheck.frame_ptr,
         g_regcheck.slot_ptr,
         g_regcheck.expected_ptr,
         match ? "YES" : "NO",
         static_cast<unsigned long>(g_regcheck.value_before),
         static_cast<unsigned long>(g_regcheck.value_after),
         changed ? "YES" : "NO");
#else
  printf("[REGMAP-RUNTIME-LEGACY] reg_id=%u(%s) bit=%u op=%lu delay=%lu "
         "tcb=%p tos=%p frame=%p slot=%p expected=%p match=%s "
         "before=0x%08lx after=0x%08lx changed=%s\n",
         static_cast<unsigned>(g_regcheck.register_id),
         register_fi_register_name(g_regcheck.register_id),
         static_cast<unsigned>(g_regcheck.bit_index),
         static_cast<unsigned long>(g_regcheck.active_op_index),
         static_cast<unsigned long>(g_regcheck.delay_us),
         g_regcheck.tcb_raw,
         g_regcheck.top_of_stack,
         g_regcheck.frame_ptr,
         g_regcheck.slot_ptr,
         g_regcheck.expected_ptr,
         match ? "YES" : "NO",
         static_cast<unsigned long>(g_regcheck.value_before),
         static_cast<unsigned long>(g_regcheck.value_after),
         changed ? "YES" : "NO");
#endif
  fflush(stdout);
  g_regcheck = {};
}
#endif

#ifdef MULTI_FI_MAX_COUNT
void register_fi_set_group(register_fi_context_t* ctx,
                           const register_fault_t* pool,
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
  }
  ctx->group_fault_count = n;
  ctx->op_fault_count = 0;
}

void register_fi_clear_group(register_fi_context_t* ctx) {
  if (ctx == nullptr) {
    return;
  }
  ctx->group_fault_count = 0;
  ctx->op_fault_count = 0;
}
#endif

extern "C" void microfi_before_op(int op_index) {
  if (g_ctx != nullptr && op_index >= 0) {
    /* MicroFI: capture SP at the operator hook – closest instrumented
     * point before the actual kernel runs.  The ISR will index into
     * the stack frame from this saved SP value. */
    REGISTER_FI_SAVE_SP();
    register_fi_before_op(g_ctx, static_cast<uint32_t>(op_index));
  }
}

extern "C" void microfi_after_op(int op_index) {
  if (g_ctx != nullptr && op_index >= 0) {
    register_fi_after_op(g_ctx, static_cast<uint32_t>(op_index));
  }
}

#else  // !FI_CAMPAIGN_REGISTER

#if !defined(FI_CAMPAIGN_MEMORY)
extern "C" void microfi_before_op(int) {}
extern "C" void microfi_after_op(int) {}
#endif

#endif  // FI_CAMPAIGN_REGISTER
