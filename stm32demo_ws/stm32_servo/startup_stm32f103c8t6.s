/* Startup file for STM32F103C8T6 (Cortex-M3) */

.syntax unified
.cpu    cortex-m3
.thumb

/* Exported symbols */
.global _estack
.global Reset_Handler

/* Weak aliases — system exceptions */
.weak   NMI_Handler
.weak   HardFault_Handler
.weak   MemManage_Handler
.weak   BusFault_Handler
.weak   UsageFault_Handler
.weak   SVC_Handler
.weak   DebugMon_Handler
.weak   PendSV_Handler
.weak   SysTick_Handler

/* Weak aliases — external interrupts (user can override) */
.weak   EXTI0_IRQHandler
.weak   EXTI1_IRQHandler
.weak   Default_Handler

/* ── Vector table (must be first in Flash) ────────────────── */
.section .vectors, "a"

    /* System exception vectors (positions 0–15) */
    .word _estack               /* 0:  Initial SP */
    .word Reset_Handler         /* 1:  Reset */
    .word NMI_Handler           /* 2:  NMI */
    .word HardFault_Handler     /* 3:  HardFault */
    .word MemManage_Handler     /* 4:  MemManage */
    .word BusFault_Handler      /* 5:  BusFault */
    .word UsageFault_Handler    /* 6:  UsageFault */
    .word 0                     /* 7:  Reserved */
    .word 0                     /* 8:  Reserved */
    .word 0                     /* 9:  Reserved */
    .word 0                     /* 10: Reserved */
    .word SVC_Handler           /* 11: SVCall */
    .word DebugMon_Handler      /* 12: Debug monitor */
    .word 0                     /* 13: Reserved */
    .word PendSV_Handler        /* 14: PendSV */
    .word SysTick_Handler       /* 15: SysTick */

    /* External interrupt vectors (positions 16–43 for STM32F103C8T6) */
    .word Default_Handler       /* 16: WWDG */
    .word Default_Handler       /* 17: PVD */
    .word Default_Handler       /* 18: TAMPER */
    .word Default_Handler       /* 19: RTC */
    .word Default_Handler       /* 20: FLASH */
    .word Default_Handler       /* 21: RCC */
    .word EXTI0_IRQHandler      /* 22: EXTI0 */
    .word EXTI1_IRQHandler      /* 23: EXTI1 */
    .word Default_Handler       /* 24: EXTI2 */
    .word Default_Handler       /* 25: EXTI3 */
    .word Default_Handler       /* 26: EXTI4 */
    .word Default_Handler       /* 27: DMA1_Channel1 */
    .word Default_Handler       /* 28: DMA1_Channel2 */
    .word Default_Handler       /* 29: DMA1_Channel3 */
    .word Default_Handler       /* 30: DMA1_Channel4 */
    .word Default_Handler       /* 31: DMA1_Channel5 */
    .word Default_Handler       /* 32: DMA1_Channel6 */
    .word Default_Handler       /* 33: DMA1_Channel7 */
    .word Default_Handler       /* 34: ADC1_2 */
    .word Default_Handler       /* 35: USB_HP_CAN1_TX */
    .word Default_Handler       /* 36: USB_LP_CAN1_RX0 */
    .word Default_Handler       /* 37: CAN1_RX1 */
    .word Default_Handler       /* 38: CAN1_SCE */
    .word Default_Handler       /* 39: EXTI9_5 */
    .word Default_Handler       /* 40: TIM1_BRK */
    .word Default_Handler       /* 41: TIM1_UP */
    .word Default_Handler       /* 42: TIM1_TRG_COM */
    .word Default_Handler       /* 43: TIM1_CC */

.section .text

/* ── Reset Handler ────────────────────────────────────────── */
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

/* ── Default handler (infinite loop) ──────────────────────── */
.thumb_func
Default_Handler:
    b    .

/* ── System exception handlers ────────────────────────────── */
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
