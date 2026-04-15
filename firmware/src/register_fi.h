#pragma once
#include <stdbool.h>
#include <stdint.h>

/*
 * RegisterFI — Interrupt Exception Frame approach (DEFAULT)
 *
 * By default, RegisterFI uses the RISC-V interrupt exception frame to
 * inject faults into real, architecturally-named CPU registers:
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
 * REGISTER_FI_USE_SAVED_SP_MICROFI (optional, opt-in)
 *
 * When explicitly defined, switches to the MicroFI-paper saved-SP approach
 * where a global captures the stack pointer at the operator hook entry and
 * the ISR indexes into stack-word slots.  This path has known limitations
 * (slot names are not real register names; the captured SP may not match
 * the operator kernel's frame).  Define it only for comparison/research.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of uint32_t stack-word slots addressable by register_id.
 * Covers 32 × 4 = 128 bytes of stack frame, which is sufficient for the
 * callee-saved registers + locals saved by a typical RISC-V prologue.
 */
#define REGISTER_FI_MAX_STACK_SLOTS 32u

#ifdef REGISTER_FI_USE_SAVED_SP_MICROFI
/*
 * g_register_fi_saved_sp
 *   Volatile global written by REGISTER_FI_SAVE_SP() at each instrumented
 *   point in the inference path.  Read by the timer ISR to locate the
 *   function's stack frame.  Zero means "no valid frame captured yet".
 */
extern volatile uintptr_t g_register_fi_saved_sp;

/*
 * REGISTER_FI_SAVE_SP()
 *   MicroFI-style macro: captures the current RISC-V stack pointer into
 *   g_register_fi_saved_sp.  Must be called from task context (not ISR).
 *   Designed to be placed at the beginning of every function / hook on
 *   the inference path so that the ISR always sees a recent SP value.
 *
 *   On RISC-V  (ESP32-C3):  mv %0, sp
 */
#define REGISTER_FI_SAVE_SP() do {       \
    uintptr_t _microfi_sp_tmp;           \
    __asm__ volatile("mv %0, sp"        \
                     : "=r"(_microfi_sp_tmp)); \
    g_register_fi_saved_sp = _microfi_sp_tmp;  \
} while (0)

#else  /* !REGISTER_FI_USE_SAVED_SP_MICROFI */
#define REGISTER_FI_SAVE_SP() ((void)0)
#endif /* REGISTER_FI_USE_SAVED_SP_MICROFI */

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
