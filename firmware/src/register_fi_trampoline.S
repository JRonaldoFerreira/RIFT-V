/*
 * register_fi_trampoline.S — MEPC-redirect trampoline for sp/gp fault injection
 *
 * ESP-IDF's restore_general_regs (vectors.S) does NOT reload sp or gp
 * from the RvExcFrame on interrupt return:
 *   - sp is recovered by `addi sp, sp, CONTEXT_SIZE`
 *   - gp is never restored (saved only for GDB)
 *
 * When the ISR needs to inject a bit-flip into sp or gp it sets up XOR
 * masks in globals and redirects frame->mepc here.  After the first
 * mret from the interrupt handler we land in this trampoline which:
 *   1. XORs sp with g_fi_trampoline_sp_mask  (0 → no-op)
 *   2. XORs gp with g_fi_trampoline_gp_mask  (0 → no-op)
 *   3. Writes the original return address to mepc
 *   4. Restores the scratch register and does mret
 *
 * Scratch register: mscratch CSR (unused by ESP-IDF interrupt handler).
 * Runs in machine mode (ESP32-C3 has no U-mode in FreeRTOS).
 *
 * Globals (defined in register_fi.cc, C linkage):
 *   g_fi_trampoline_sp_mask  — XOR mask for sp
 *   g_fi_trampoline_gp_mask  — XOR mask for gp
 *   g_fi_trampoline_mepc     — original mepc to resume execution
 */

#if defined(FI_CAMPAIGN_REGISTER) && !defined(REGISTER_FI_USE_SAVED_SP_MICROFI)

    .section .iram1,"ax"
    .align 2

    .global _fi_trampoline
    .type   _fi_trampoline, @function
_fi_trampoline:
    /* Save t0 — only scratch register we need */
    csrw    mscratch, t0

    /* Apply sp fault mask (XOR with 0 is a harmless no-op) */
    la      t0, g_fi_trampoline_sp_mask
    lw      t0, 0(t0)
    xor     sp, sp, t0

    /* Apply gp fault mask */
    la      t0, g_fi_trampoline_gp_mask
    lw      t0, 0(t0)
    xor     gp, gp, t0

    /* Set mepc to the original return address */
    la      t0, g_fi_trampoline_mepc
    lw      t0, 0(t0)
    csrw    mepc, t0

    /* Restore t0 and return to original code */
    csrr    t0, mscratch
    mret

    .size _fi_trampoline, .-_fi_trampoline

#endif /* FI_CAMPAIGN_REGISTER && !REGISTER_FI_USE_SAVED_SP_MICROFI */
