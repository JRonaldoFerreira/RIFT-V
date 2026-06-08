#pragma once
/*
 * fi_cycles.h — CPU cycle accounting for fault-injection overhead.
 *
 * Provides a portable, header-only API to sample the RISC-V cycle
 * counter (`mcycle` CSR) on ESP32-C3 / C6 / H2 / P4 targets so the
 * register- and memory-FI subsystems can quantify their own runtime
 * cost in **CPU cycles**, matching the methodology used by microFI
 * to report "~1300 CPU cycles or 8.1 µs per injection".
 *
 * Why a dedicated header (instead of `esp_cpu_get_cycle_count()`):
 *   - Zero IDF version dependency (the same prototype lived in
 *     different headers across IDF 4.x / 5.x).
 *   - Single inline-asm read of `mcycle` — minimum perturbation of
 *     the very code path we are trying to measure.
 *   - Works identically from C / C++ / ISR context.
 *
 * On the ESP32-C6 the CPU runs at 160 MHz by default, so 1 cycle
 * ≈ 6.25 ns and 1 µs = 160 cycles.  The conversion uses the value
 * reported by `esp_clk_cpu_freq()` once at boot and cached in
 * `g_fi_cpu_freq_hz`.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CPU frequency in Hz, cached at boot (see fi_cycles_init). */
extern volatile uint32_t g_fi_cpu_freq_hz;

/* One-shot initialization — should be called once during boot. */
void fi_cycles_init(void);

/* Read the 32-bit cycle counter.
 *
 * The ESP32 RISC-V targets (C3, C5, C6, H2, P4) do NOT implement the
 * standard `mcycle` machine-mode CSR — instead Espressif exposes a
 * per-chip performance counter through the **machine performance
 * counter** CSR `0x7E2` (a.k.a. `CSR_PCCR_MACHINE` in ESP-IDF).
 * Reading the wrong CSR raises an "Illegal instruction" exception,
 * so we read 0x7E2 directly with the same instruction
 * `esp_cpu_get_cycle_count()` would emit — only this time fully
 * inlined.
 *
 * The inline form is critical for fault-injection overhead bookkeeping:
 * with an out-of-line wrapper the function-call ABI (~50 cycles per
 * call) accounts for >25% of the per-hook cost on a no-op path.
 * Inlining drops `fi_read_cycles()` to a single `csrr` instruction
 * (1-2 cycles), removing the bookkeeping bias when comparing the
 * framework's per-injection cost against the microFI paper's "around
 * 1300 CPU cycles" figure.
 *
 * At 160 MHz the 32-bit counter wraps every ~26.8 seconds, more than
 * enough for single-injection overhead bookkeeping.  Callers that may
 * straddle a wrap should subtract using `uint32_t` arithmetic, which
 * already handles the wrap correctly. */
#if defined(__riscv) || defined(__riscv__)

#define FI_CYCLES_CSR_PCCR_MACHINE 0x7E2

static inline __attribute__((always_inline)) uint32_t fi_read_cycles(void) {
  uint32_t v;
  __asm__ volatile("csrr %0, %1" : "=r"(v) : "i"(FI_CYCLES_CSR_PCCR_MACHINE));
  return v;
}

#else
/* Fallback for non-RISC-V targets: defer to the out-of-line wrapper
 * implemented in `fi_cycles.cc`, which calls the right per-arch ESP-IDF
 * helper.  Xtensa targets use the CCOUNT register. */
uint32_t fi_read_cycles(void);
#endif

/* Convert a cycle count to microseconds using the cached CPU freq.
 * Returns 0 if `fi_cycles_init` has not run yet. */
static inline __attribute__((always_inline)) uint64_t fi_cycles_to_us(uint64_t cycles) {
  const uint32_t hz = g_fi_cpu_freq_hz;
  if (hz == 0u) return 0u;
  return (cycles * 1000000ull) / hz;
}

#ifdef __cplusplus
}
#endif
