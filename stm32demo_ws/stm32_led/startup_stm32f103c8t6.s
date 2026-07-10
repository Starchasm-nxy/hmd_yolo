/* Startup file for STM32F103C8T6 (Cortex-M3) */

.syntax unified
.cpu    cortex-m3
.thumb

/* Exported symbols */
.global _estack
.global Reset_Handler

/* Weak aliases — user can override any ISR */
.weak   NMI_Handler
.weak   HardFault_Handler
.weak   MemManage_Handler
.weak   BusFault_Handler
.weak   UsageFault_Handler
.weak   SVC_Handler
.weak   DebugMon_Handler
.weak   PendSV_Handler
.weak   SysTick_Handler

/* .vectors section goes first in Flash */
.section .vectors, "a"

    .word _estack               /* Initial stack pointer */
    .word Reset_Handler         /* Reset */
    .word NMI_Handler           /* NMI */
    .word HardFault_Handler     /* HardFault */
    .word MemManage_Handler     /* MemManage */
    .word BusFault_Handler      /* BusFault */
    .word UsageFault_Handler    /* UsageFault */
    .word 0                     /* Reserved */
    .word 0                     /* Reserved */
    .word 0                     /* Reserved */
    .word 0                     /* Reserved */
    .word SVC_Handler           /* SVCall */
    .word DebugMon_Handler      /* Debug monitor */
    .word 0                     /* Reserved */
    .word PendSV_Handler        /* PendSV */
    .word SysTick_Handler       /* SysTick */

.section .text

/* ── Reset Handler ──────────────────────────────────────── */
.type Reset_Handler, %function
Reset_Handler:
    ldr   r0, =_estack
    mov   sp, r0                /* set stack pointer */

    /* Copy .data section from Flash to SRAM */
    ldr   r0, =_sdata
    ldr   r1, =_edata
    ldr   r2, =_sidata
    cmp   r0, r1
    beq   2f
1:  ldr   r3, [r2], #4
    str   r3, [r0], #4
    cmp   r0, r1
    bne   1b
2:

    /* Zero-fill .bss section */
    ldr   r0, =_sbss
    ldr   r1, =_ebss
    mov   r2, #0
    cmp   r0, r1
    beq   4f
3:  str   r2, [r0], #4
    cmp   r0, r1
    bne   3b
4:

    bl    main                  /* call main() */
    b     .                     /* trap if main returns */

/* ── Default interrupt handlers (infinite loop) ─────────── */
.thumb_func
NMI_Handler:
.thumb_func
HardFault_Handler:
.thumb_func
MemManage_Handler:
.thumb_func
BusFault_Handler:
.thumb_func
UsageFault_Handler:
.thumb_func
SVC_Handler:
.thumb_func
DebugMon_Handler:
.thumb_func
PendSV_Handler:
.thumb_func
SysTick_Handler:
    b    .
