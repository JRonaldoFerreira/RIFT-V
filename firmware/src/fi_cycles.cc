#include "fi_cycles.h"

#include "esp_attr.h"
#include "esp_log.h"
#include "esp_cpu.h"

/* Different IDF versions place the CPU-frequency helper in different
 * headers (`esp_clk.h`, `esp_private/esp_clk.h`, ...). Forward-declare
 * locally so this file does not pull a public header that may not be
 * exposed by some board variants. The signature is stable across IDF
 * 4.x and 5.x. */
extern "C" int esp_clk_cpu_freq(void);

volatile uint32_t g_fi_cpu_freq_hz = 0u;

#if !(defined(__riscv) || defined(__riscv__))
/* Non-RISC-V fallback (e.g. Xtensa targets).  On RISC-V the header
 * provides an `always_inline` definition that reads CSR 0x7E2 directly
 * — this is critical for keeping the per-hook bookkeeping cheap. */
extern "C" IRAM_ATTR uint32_t fi_read_cycles(void) {
  return static_cast<uint32_t>(esp_cpu_get_cycle_count());
}
#endif

extern "C" void fi_cycles_init(void) {
  const int hz = esp_clk_cpu_freq();
  if (hz > 0) {
    g_fi_cpu_freq_hz = static_cast<uint32_t>(hz);
  } else {
    /* Reasonable default for ESP32-C6 / C3 / H2 default configuration. */
    g_fi_cpu_freq_hz = 160000000u;
  }
  /* Quick self-test: read the counter twice and make sure it advances.
   * Catches silently-broken cycle-counter back-ends early. */
  const uint32_t c0 = fi_read_cycles();
  const uint32_t c1 = fi_read_cycles();
  ESP_LOGI("fi_cycles",
           "CPU cycle counter initialised (freq=%u Hz, ns/cycle=%.3f, "
           "self-test Δ=%u cycles)",
           static_cast<unsigned>(g_fi_cpu_freq_hz),
           1e9 / static_cast<float>(g_fi_cpu_freq_hz),
           static_cast<unsigned>(c1 - c0));
}
