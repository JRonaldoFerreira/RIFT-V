#pragma once
#include <stdbool.h>
#include <stdint.h>

/*
 * RegisterFI — Interrupt Exception Frame approach
 *
 * RegisterFI uses the RISC-V interrupt exception frame to inject faults
 * into real, architecturally-named CPU registers:
 *
 *   1. The esp_timer callback fires via ESP_TIMER_ISR (interrupt context).
 *   2. The interrupt entry code (_interrupt_handler in vectors.S) has
 *      already saved ALL 31 GPRs into an RvExcFrame on the task stack.
 *   3. rtos_int_enter (portasm.S) stored that stack pointer into
 *      pxCurrentTCBs[0]->pxTopOfStack.
 *   4. Our ISR callback reads that pointer, casts to RvExcFrame*, and
 *      modifies the named field (e.g. frame->s3 ^= (1 << bit)).
 *   5. On interrupt return, restore_general_regs + mret load the
 *      (corrupted) value back into the real CPU register.
 *
 * This approach:
 *   - Provides 100% correct architectural register names (ra, s0, a0, …)
 *   - Corrupts the register DURING operator execution (faithful fault model)
 *   - Is independent of compiler prologue layout
 *
 * sp and gp are not restored by restore_general_regs in vectors.S, so
 * faults targeting those registers are applied via _fi_trampoline (asm)
 * which XORs the live register after mret.
 */

#ifdef __cplusplus
extern "C" {
#endif


#if defined(FI_CAMPAIGN_REGISTER)
#include "register_fault_list.h"
#else
typedef struct {
  int32_t fault_id;
  uint32_t op_index;
  uint8_t time_slot_index;
  uint8_t time_slot_count;
  uint8_t register_id;
  uint8_t bit_index;
} register_fault_t;
#endif

typedef enum {
  REGISTER_FI_IDLE = 0,
  REGISTER_FI_ARMED,
  REGISTER_FI_INJECTED,
  REGISTER_FI_TIMER_NOT_FIRED,
  REGISTER_FI_TIMING_WINDOW_TOO_SHORT,
  REGISTER_FI_INVALID_FRAME,
  REGISTER_FI_INVALID_REGISTER,
} register_fi_injection_status_t;

typedef struct {
  uint32_t count;
  uint64_t total_duration_us;
  uint32_t mean_duration_us;
} register_fi_op_profile_t;

/* Per-injection overhead accumulators — microFI-style accounting.
 *
 * The microFI paper reports "~1300 CPU cycles or 8.1 µs per injection"
 * (operation monitoring + timer set-up + bit flip + context switching).
 * The fields below break that same total down into the individual
 * stages so the two frameworks can be compared component-by-component.
 *
 *   before_op_cycles  → operation monitoring + timer set-up
 *   isr_cycles        → bit flip (work done inside register_timer_isr)
 *   isr_entry_cycles  → context switching (interrupt dispatch latency
 *                        beyond the requested delay)
 *   after_op_cycles   → operation monitoring + timer tear-down
 *
 * `injections` counts the number of bit-flips actually applied during
 * the campaign, so the per-injection averages are simply
 * `total_cycles / injections`. */
/* Counters are 32-bit (not 64-bit) to keep the per-hook accumulator
 * update down to a single load/add/store on RV32 — 64-bit adds compile
 * to 6+ instructions plus an `sltu` for carry, which alone accounts for
 * ~10 cycles of every hook invocation on the C6.  At 160 MHz, even
 * 4 billion cycles is only ~25 s, which is far longer than any single
 * fault-injection campaign run between resets. */
typedef struct {
  uint32_t before_op_cycles;
  uint32_t before_op_count;
  uint32_t after_op_cycles;
  uint32_t after_op_count;
  uint32_t isr_cycles;
  uint32_t isr_count;
  uint32_t isr_entry_cycles;
  uint32_t isr_entry_count;
  uint32_t injections;
} register_fi_overhead_t;

typedef struct {
  const register_fault_t* fault_list;
  uint32_t fault_count;
  volatile int32_t active_fault_index;
  volatile int32_t active_op_index;
  volatile bool op_active;
  volatile bool timer_armed;
  volatile uint32_t current_delay_us;
  volatile register_fi_injection_status_t last_injection_status;

  bool profiling_enabled;
  uint64_t op_start_time_us[32];
  register_fi_op_profile_t op_profiles[32];
  uint32_t min_timer_delay_us;
  uint32_t slot_guard_us;
  void* timer_handle;

  volatile bool permanent_active;
  volatile bool permanent_original_bit;

  /* Overhead accounting — see register_fi_overhead_t. Updated from
   * both task and ISR context; the writes are monotonic and the host
   * only reads the snapshot after the campaign, so no atomicity
   * primitives are needed. */
  bool overhead_enabled;
  register_fi_overhead_t overhead;
  volatile uint32_t arm_cycles;        /* mcycle taken when arming  */
  volatile uint32_t arm_delay_cycles;  /* expected delay in cycles  */

#ifdef MULTI_FI_MAX_COUNT
  const register_fault_t* group_faults[MULTI_FI_MAX_COUNT];
  uint8_t group_fault_count;
  volatile const register_fault_t* op_faults[MULTI_FI_MAX_COUNT];
  volatile uint8_t op_fault_count;
  volatile uint32_t op_fault_delays[MULTI_FI_MAX_COUNT];
  volatile uint8_t op_fault_next_idx;
  volatile bool permanent_injected[MULTI_FI_MAX_COUNT];
  volatile bool permanent_original_bits[MULTI_FI_MAX_COUNT];
#endif
} register_fi_context_t;

void register_fi_init(register_fi_context_t* ctx,
                      const register_fault_t* fault_list,
                      uint32_t fault_count,
                      uint32_t min_timer_delay_us,
                      uint32_t slot_guard_us);
void register_fi_deinit(register_fi_context_t* ctx);
void register_fi_begin_inference(register_fi_context_t* ctx, int32_t fault_index);
void register_fi_end_inference(register_fi_context_t* ctx);
void register_fi_before_op(register_fi_context_t* ctx, uint32_t op_index);
void register_fi_after_op(register_fi_context_t* ctx, uint32_t op_index);
void register_fi_set_profiling(register_fi_context_t* ctx, bool enabled);
uint32_t register_fi_get_mean_duration_us(const register_fi_context_t* ctx, uint32_t op_index);
uint32_t register_fi_get_current_delay_us(const register_fi_context_t* ctx);
register_fi_injection_status_t register_fi_get_last_status(const register_fi_context_t* ctx);
const char* register_fi_status_name(register_fi_injection_status_t status);
const char* register_fi_register_name(uint8_t register_id);

/* Overhead-accounting API. */
void register_fi_set_overhead_enabled(register_fi_context_t* ctx, bool enabled);
void register_fi_reset_overhead(register_fi_context_t* ctx);
void register_fi_get_overhead(const register_fi_context_t* ctx,
                              register_fi_overhead_t* out);

#ifdef MULTI_FI_MAX_COUNT
void register_fi_set_group(register_fi_context_t* ctx,
                           const register_fault_t* pool,
                           const uint16_t* pool_indices,
                           uint8_t count);
void register_fi_clear_group(register_fi_context_t* ctx);
#endif

// Hooks called from the patched TFLM loop.
void microfi_before_op(int op_index);
void microfi_after_op(int op_index);

#ifdef REGISTER_FI_RUNTIME_CHECK
void register_fi_runtime_check_print(void);
#endif

#ifdef __cplusplus
}
#endif
