/* LED Blink on PA5 — STM32F103C8T6 bare-metal */

/* ── Peripheral register addresses ─────────────────────── */

/* RCC (Reset & Clock Control) */
#define RCC_BASE        0x40021000UL
#define RCC_APB2ENR     (*(volatile unsigned int *)(RCC_BASE + 0x18))

/* GPIOA */
#define GPIOA_BASE      0x40010800UL
#define GPIOA_CRL       (*(volatile unsigned int *)(GPIOA_BASE + 0x00))
#define GPIOA_CRH       (*(volatile unsigned int *)(GPIOA_BASE + 0x04))
#define GPIOA_ODR       (*(volatile unsigned int *)(GPIOA_BASE + 0x0C))
#define GPIOA_BSRR      (*(volatile unsigned int *)(GPIOA_BASE + 0x10))

/* ── Bit definitions ───────────────────────────────────── */
#define RCC_APB2ENR_IOPAEN  (1U << 2)   /* GPIOA clock enable */

/* ── Simple busy-wait delay ~200ms at 8 MHz HSI ────────── */
static void delay(unsigned int count)
{
    while (count--) {
        __asm__ volatile ("nop");
    }
}

/* ── Main ───────────────────────────────────────────────── */
int main(void)
{
    /* 1. Enable GPIOA clock */
    RCC_APB2ENR |= RCC_APB2ENR_IOPAEN;

    /* 2. Configure PA5 as 2 MHz push-pull output
     *    GPIOA_CRL controls pins 0–7 (4 bits each).
     *    PA5 is at bits [23:20]:
     *      CNF5  = 00 (general purpose push-pull)
     *      MODE5 = 10 (output, max speed 2 MHz)
     *    Clear bits then set MODE5_1.
     */
    GPIOA_CRL &= ~(0xFU << 20);         /* clear bits for pin 5 */
    GPIOA_CRL |=  (0x2U << 20);         /* MODE5 = 10               */

    /* 3. Blink forever */
    while (1) {
        /* Turn LED on: set bit 5 in BSRR (lower 16 bits = set) */
        GPIOA_BSRR = (1U << 5);
        delay(400000);

        /* Turn LED off: reset bit 5 in BSRR (upper 16 bits = reset) */
        GPIOA_BSRR = (1U << (5 + 16));
        delay(400000);
    }
}
