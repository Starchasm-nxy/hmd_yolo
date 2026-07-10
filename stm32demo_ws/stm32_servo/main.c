/* ───────────────────────────────────────────────────────────────
 * STM32F103C8T6 — Servo + rotary encoder + SSD1306 OLED
 * ───────────────────────────────────────────────────────────────
 * Pin connections:
 *   PB0, PB1  — Rotary encoder A/B (EXTI falling-edge interrupts)
 *   PA1       — SG90 servo PWM (TIM2 CH2)
 *   PB7       — OLED VCC (power control, push-pull HIGH)
 *   PB8 (SCL),
 *   PB9 (SDA) — SSD1306 OLED I2C (bit-banged)
 *
 * Reference: Keil project "1-1编码器控制舵机（中断" — Encoder.c / PWM.c
 *   Encoder: falling-edge EXTI, count only when both pins are LOW
 *   Servo:   TIM2 CH2, PSC=71, ARR=19999, CCR = angle/180*2000+500
 * Clock: HSE 8MHz → PLL ×9 → 72MHz SYSCLK
 * ─────────────────────────────────────────────────────────────── */

#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════
 * Register definitions
 * ══════════════════════════════════════════════════════════════ */

/* ── RCC ──────────────────────────────────────────────────── */
#define RCC_BASE            0x40021000UL
#define RCC_CR             (*(volatile uint32_t *)(RCC_BASE + 0x00))
#define RCC_CFGR           (*(volatile uint32_t *)(RCC_BASE + 0x04))
#define RCC_APB2ENR        (*(volatile uint32_t *)(RCC_BASE + 0x18))
#define RCC_APB1ENR        (*(volatile uint32_t *)(RCC_BASE + 0x1C))

/* RCC_CR */
#define RCC_CR_HSION       (1U << 0)
#define RCC_CR_HSIRDY      (1U << 1)
#define RCC_CR_HSEON       (1U << 16)
#define RCC_CR_HSERDY      (1U << 17)
#define RCC_CR_PLLON       (1U << 24)
#define RCC_CR_PLLRDY      (1U << 25)

/* RCC_CFGR */
#define RCC_CFGR_SW_MSK    (3U << 0)
#define RCC_CFGR_SW_PLL    (2U << 0)
#define RCC_CFGR_SWS_MSK   (3U << 2)
#define RCC_CFGR_SWS_PLL   (2U << 2)
#define RCC_CFGR_PPRE1_DIV2 (4U << 8)

/* RCC_APB2ENR */
#define RCC_APB2ENR_AFIOEN (1U << 0)
#define RCC_APB2ENR_IOPAEN (1U << 2)
#define RCC_APB2ENR_IOPBEN (1U << 3)

/* RCC_APB1ENR */
#define RCC_APB1ENR_TIM2EN (1U << 0)

/* ── FLASH ────────────────────────────────────────────────── */
#define FLASH_BASE          0x40022000UL
#define FLASH_ACR          (*(volatile uint32_t *)(FLASH_BASE + 0x00))
#define FLASH_ACR_LATENCY_2 (2U << 0)

/* ── GPIOA ────────────────────────────────────────────────── */
#define GPIOA_BASE          0x40010800UL
#define GPIOA_CRL          (*(volatile uint32_t *)(GPIOA_BASE + 0x00))

/* ── GPIOB ────────────────────────────────────────────────── */
#define GPIOB_BASE          0x40010C00UL
#define GPIOB_CRL          (*(volatile uint32_t *)(GPIOB_BASE + 0x00))
#define GPIOB_CRH          (*(volatile uint32_t *)(GPIOB_BASE + 0x04))
#define GPIOB_IDR          (*(volatile uint32_t *)(GPIOB_BASE + 0x08))
#define GPIOB_ODR          (*(volatile uint32_t *)(GPIOB_BASE + 0x0C))
#define GPIOB_BSRR         (*(volatile uint32_t *)(GPIOB_BASE + 0x10))

/* ── AFIO ─────────────────────────────────────────────────── */
#define AFIO_BASE           0x40010000UL
#define AFIO_EXTICR1       (*(volatile uint32_t *)(AFIO_BASE + 0x08))

/* ── EXTI ─────────────────────────────────────────────────── */
#define EXTI_BASE           0x40010400UL
#define EXTI_IMR           (*(volatile uint32_t *)(EXTI_BASE + 0x00))
#define EXTI_RTSR          (*(volatile uint32_t *)(EXTI_BASE + 0x08))
#define EXTI_FTSR          (*(volatile uint32_t *)(EXTI_BASE + 0x0C))
#define EXTI_PR            (*(volatile uint32_t *)(EXTI_BASE + 0x14))

/* ── NVIC (Cortex-M3) ─────────────────────────────────────── */
#define NVIC_BASE           0xE000E100UL
#define NVIC_ISER0         (*(volatile uint32_t *)(NVIC_BASE + 0x00))

/* EXTI interrupt numbers in NVIC:
 *   EXTI0 = 6,  EXTI1 = 7 (positions in ISER0) */

/* ── TIM2 (APB1, timer clock = 72MHz) ─────────────────────── */
#define TIM2_BASE           0x40000000UL
#define TIM2_CR1           (*(volatile uint32_t *)(TIM2_BASE + 0x00))
#define TIM2_EGR           (*(volatile uint32_t *)(TIM2_BASE + 0x14))
#define TIM2_CCMR1         (*(volatile uint32_t *)(TIM2_BASE + 0x18))
#define TIM2_CCER          (*(volatile uint32_t *)(TIM2_BASE + 0x20))
#define TIM2_CNT           (*(volatile uint32_t *)(TIM2_BASE + 0x24))
#define TIM2_PSC           (*(volatile uint32_t *)(TIM2_BASE + 0x28))
#define TIM2_ARR           (*(volatile uint32_t *)(TIM2_BASE + 0x2C))
#define TIM2_CCR2          (*(volatile uint32_t *)(TIM2_BASE + 0x38))

/* ── SysTick (Cortex-M3) ──────────────────────────────────── */
#define SYSTICK_BASE        0xE000E010UL
#define SYSTICK_CSR        (*(volatile uint32_t *)(SYSTICK_BASE + 0x00))
#define SYSTICK_RVR        (*(volatile uint32_t *)(SYSTICK_BASE + 0x04))
#define SYSTICK_CVR        (*(volatile uint32_t *)(SYSTICK_BASE + 0x08))
#define SYSTICK_CSR_ENABLE  (1U << 0)
#define SYSTICK_CSR_TICKINT (1U << 1)
#define SYSTICK_CSR_CLKSRC  (1U << 2)   /* 1 = AHB clock (72MHz) */

/* ═══════════════════════════════════════════════════════════════
 * Global state shared with ISRs
 * ══════════════════════════════════════════════════════════════ */
static volatile int32_t  enc_count   = 0;    /* encoder position  */
static volatile uint32_t ms_ticks    = 0;    /* SysTick millis    */

/* Remote angle from PC (OpenOCD writes here via telnet: mww 0x20000000 <val>)
 * Placed at fixed SRAM address 0x20000000 by linker script.
 * Value > 180 means "no remote command" → encoder mode. */
static volatile uint32_t remote_angle __attribute__((section(".remote")));

/* ═══════════════════════════════════════════════════════════════
 * System clock: HSE 8MHz → PLL ×9 → 72MHz
 * ══════════════════════════════════════════════════════════════ */
static void clock_init(void)
{
    /* Enable HSE */
    RCC_CR |= RCC_CR_HSEON;
    while (!(RCC_CR & RCC_CR_HSERDY)) { __asm__ volatile ("nop"); }

    /* Flash: 2 wait states (required for 72 MHz) */
    FLASH_ACR |= FLASH_ACR_LATENCY_2;

    /* AHB=/1, APB1=/2 (→36MHz; timers ×2→72MHz), APB2=/1 */
    RCC_CFGR &= ~(0xFU << 4);              /* HPRE = /1  */
    RCC_CFGR &= ~(0x7U << 8);
    RCC_CFGR |= RCC_CFGR_PPRE1_DIV2;       /* PPRE1 = /2 */
    RCC_CFGR &= ~(0x7U << 11);             /* PPRE2 = /1 */

    /* PLL: HSE ×9 = 72MHz */
    RCC_CFGR &= ~(0xFU << 18);             /* clear PLLMUL */
    RCC_CFGR |= (1U << 16)                /* PLLSRC = HSE */
             |  (7U << 18);               /* PLLMUL = ×9  */

    /* Enable PLL and wait for lock */
    RCC_CR |= RCC_CR_PLLON;
    while (!(RCC_CR & RCC_CR_PLLRDY)) { __asm__ volatile ("nop"); }

    /* Switch to PLL */
    RCC_CFGR = (RCC_CFGR & ~RCC_CFGR_SW_MSK) | RCC_CFGR_SW_PLL;
    while ((RCC_CFGR & RCC_CFGR_SWS_MSK) != RCC_CFGR_SWS_PLL) {
        __asm__ volatile ("nop");
    }
}

/* ═══════════════════════════════════════════════════════════════
 * SysTick — 1 ms tick
 * ══════════════════════════════════════════════════════════════ */
void SysTick_Handler(void)
{
    ms_ticks++;
}

static void delay_init(void)
{
    SYSTICK_RVR = 72000 - 1;   /* 72MHz / 72000 = 1 kHz */
    SYSTICK_CVR = 0;
    SYSTICK_CSR = SYSTICK_CSR_ENABLE | SYSTICK_CSR_TICKINT | SYSTICK_CSR_CLKSRC;
}

static void delay_ms(uint32_t ms)
{
    uint32_t start = ms_ticks;
    while ((ms_ticks - start) < ms) {
        __asm__ volatile ("wfi");
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Rotary encoder — EXTI falling-edge on PB0 & PB1
 * ─────────────────────────────────────────────────────────────
 * Same approach as Keil reference Encoder.c:
 *   Trigger on falling edge only.
 *   If both pins are LOW when the interrupt fires → valid step.
 *   EXTI0 (A fell) + both low → CCW  (count--)
 *   EXTI1 (B fell) + both low → CW   (count++)
 * (Swap ++/-- if your encoder direction feels reversed.)
 * ══════════════════════════════════════════════════════════════ */

void EXTI0_IRQHandler(void)
{
    if (EXTI_PR & (1U << 0)) {                       /* EXTI0 pending */
        if (((GPIOB_IDR >> 0) & 1) == 0) {           /* PB0 (A) is LOW  */
            if (((GPIOB_IDR >> 1) & 1) == 0) {       /* PB1 (B) is LOW  */
                enc_count--;
            }
        }
        EXTI_PR = (1U << 0);                         /* clear pending */
    }
}

void EXTI1_IRQHandler(void)
{
    if (EXTI_PR & (1U << 1)) {                       /* EXTI1 pending */
        if (((GPIOB_IDR >> 1) & 1) == 0) {           /* PB1 (B) is LOW  */
            if (((GPIOB_IDR >> 0) & 1) == 0) {       /* PB0 (A) is LOW  */
                enc_count++;
            }
        }
        EXTI_PR = (1U << 1);                         /* clear pending */
    }
}

static void encoder_init(void)
{
    /* Enable clocks */
    RCC_APB2ENR |= RCC_APB2ENR_IOPBEN | RCC_APB2ENR_AFIOEN;

    /* PB0, PB1 as inputs with pull-ups
     * CNF=10 (input with PU/PD), MODE=00 (input) → 0x8 per pin */
    GPIOB_CRL &= ~(0xFFU << 0);
    GPIOB_CRL |=  (0x88U << 0);
    GPIOB_ODR |=  (1U << 0) | (1U << 1);  /* pull-up (ODR=1 in input mode) */

    /* Map EXTI0 → PB0, EXTI1 → PB1
     * AFIO_EXTICR1: EXTI0 @ bits[3:0], EXTI1 @ bits[7:4]
     * Port B = 0001 */
    AFIO_EXTICR1 &= ~((0xFU << 0) | (0xFU << 4));
    AFIO_EXTICR1 |=  (0x1U << 0) | (0x1U << 4);

    /* Trigger on falling edge only (match Keil reference) */
    EXTI_FTSR |= (1U << 0) | (1U << 1);

    /* Unmask EXTI0 and EXTI1 */
    EXTI_IMR  |= (1U << 0) | (1U << 1);

    /* Enable NVIC interrupts: EXTI0=pos6, EXTI1=pos7 */
    NVIC_ISER0 = (1U << 6) | (1U << 7);
}

/* ═══════════════════════════════════════════════════════════════
 * I2C bit-bang on PB8 (SCL) / PB9 (SDA)
 * ══════════════════════════════════════════════════════════════ */

static void i2c_delay(void)
{
    for (volatile int i = 0; i < 30; i++) {
        __asm__ volatile ("nop");
    }
}

static void i2c_scl_lo(void) { GPIOB_BSRR = (1U << (8 + 16)); }
static void i2c_scl_hi(void) { GPIOB_BSRR = (1U << 8); }
static void i2c_sda_lo(void) { GPIOB_BSRR = (1U << (9 + 16)); }
static void i2c_sda_hi(void) { GPIOB_BSRR = (1U << 9); }

static void i2c_init(void)
{
    RCC_APB2ENR |= RCC_APB2ENR_IOPBEN;

    /* PB8, PB9: open-drain output, 50MHz  (CRH bits [3:0] and [7:4])
     * CNF=01 (GP open-drain), MODE=11 → 0x7 per pin */
    GPIOB_CRH &= ~(0xFFU << 0);
    GPIOB_CRH |=  (0x77U << 0);

    i2c_scl_hi();
    i2c_sda_hi();
}

static void i2c_start(void)
{
    i2c_sda_hi();
    i2c_scl_hi();
    i2c_delay();
    i2c_sda_lo();
    i2c_delay();
    i2c_scl_lo();
}

static void i2c_stop(void)
{
    i2c_sda_lo();
    i2c_scl_hi();
    i2c_delay();
    i2c_sda_hi();
    i2c_delay();
}

static void i2c_write_byte(uint8_t data)
{
    for (int i = 7; i >= 0; i--) {
        if (data & (1U << i))
            i2c_sda_hi();
        else
            i2c_sda_lo();
        i2c_delay();
        i2c_scl_hi();
        i2c_delay();
        i2c_scl_lo();
    }
    /* ACK clock — ignore slave ACK */
    i2c_sda_hi();
    i2c_delay();
    i2c_scl_hi();
    i2c_delay();
    i2c_scl_lo();
}

/* ═══════════════════════════════════════════════════════════════
 * SSD1306 128×64 OLED
 * I2C addr: 0x3C (7-bit) → write byte = 0x78
 * Control byte: 0x00 = command, 0x40 = data
 * ══════════════════════════════════════════════════════════════ */

#define OLED_ADDR  0x78

static void oled_cmd(uint8_t cmd)
{
    i2c_start();
    i2c_write_byte(OLED_ADDR);
    i2c_write_byte(0x00);
    i2c_write_byte(cmd);
    i2c_stop();
}

static void oled_data_burst(const uint8_t *data, uint16_t len)
{
    i2c_start();
    i2c_write_byte(OLED_ADDR);
    i2c_write_byte(0x40);
    for (uint16_t i = 0; i < len; i++)
        i2c_write_byte(data[i]);
    i2c_stop();
}

static void oled_set_cursor(uint8_t page, uint8_t col)
{
    oled_cmd(0xB0 | (page & 0x07));
    oled_cmd(0x00 | (col & 0x0F));
    oled_cmd(0x10 | ((col >> 4) & 0x0F));
}

static void oled_clear(void)
{
    for (uint8_t p = 0; p < 8; p++) {
        oled_set_cursor(p, 0);
        i2c_start();
        i2c_write_byte(OLED_ADDR);
        i2c_write_byte(0x40);
        for (uint16_t c = 0; c < 128; c++)
            i2c_write_byte(0x00);
        i2c_stop();
    }
}

static void oled_init(void)
{
    /* PB7 as push-pull output for OLED VCC power */
    GPIOB_CRL &= ~(0xFU << 28);
    GPIOB_CRL |=  (0x3U << 28);      /* CNF=00 (GP PP), MODE=11 (50MHz) */

    /* Power up sequence: PB7 low→delay→high→delay */
    GPIOB_BSRR = (1U << (7 + 16));   /* PB7 low  */
    delay_ms(10);
    GPIOB_BSRR = (1U << 7);          /* PB7 high */
    delay_ms(10);

    /* ── SSD1306 init sequence ── */
    static const uint8_t init_seq[] = {
        0xAE,       /* display off */
        0xD5, 0x80, /* clock div */
        0xA8, 0x3F, /* mux ratio (64) */
        0xD3, 0x00, /* display offset */
        0x40,       /* start line */
        0x8D, 0x14, /* charge pump on */
        0x20, 0x00, /* horizontal addressing */
        0xA1,       /* segment remap */
        0xC8,       /* COM scan direction */
        0xDA, 0x12, /* COM pins */
        0x81, 0xCF, /* contrast */
        0xD9, 0xF1, /* pre-charge */
        0xDB, 0x40, /* VCOMH level */
        0xA4,       /* display on resume */
        0xA6,       /* normal (not inverted) */
        0xAF,       /* display ON */
    };
    for (uint32_t i = 0; i < sizeof(init_seq); i++)
        oled_cmd(init_seq[i]);
}

/* ═══════════════════════════════════════════════════════════════
 * 8×8 Font — column-major (each byte = one 8-pixel vertical column)
 * Covers ASCII 0x20 (space) – 0x5A ('Z')
 * Index: font[(ascii - 0x20) * 8 + column]
 * ══════════════════════════════════════════════════════════════ */

static const uint8_t font8x8[] = {
    /* 0x20   SPACE */
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    /* 0x21 ! */
    0x00,0x00,0x00,0x00,0x5F,0x00,0x00,0x00,
    /* 0x22 " */
    0x00,0x00,0x00,0x07,0x00,0x07,0x00,0x00,
    /* 0x23 # */
    0x00,0x14,0x7F,0x14,0x7F,0x14,0x00,0x00,
    /* 0x24 $ */
    0x00,0x24,0x2A,0x7F,0x2A,0x12,0x00,0x00,
    /* 0x25 % */
    0x00,0x23,0x13,0x08,0x64,0x62,0x00,0x00,
    /* 0x26 & */
    0x00,0x36,0x49,0x55,0x22,0x50,0x00,0x00,
    /* 0x27 ' */
    0x00,0x00,0x00,0x07,0x00,0x00,0x00,0x00,
    /* 0x28 ( */
    0x00,0x00,0x1C,0x22,0x41,0x00,0x00,0x00,
    /* 0x29 ) */
    0x00,0x00,0x41,0x22,0x1C,0x00,0x00,0x00,
    /* 0x2A * */
    0x00,0x08,0x2A,0x1C,0x2A,0x08,0x00,0x00,
    /* 0x2B + */
    0x00,0x08,0x08,0x3E,0x08,0x08,0x00,0x00,
    /* 0x2C , */
    0x00,0x00,0x50,0x30,0x00,0x00,0x00,0x00,
    /* 0x2D - */
    0x00,0x08,0x08,0x08,0x08,0x08,0x00,0x00,
    /* 0x2E . */
    0x00,0x00,0x60,0x60,0x00,0x00,0x00,0x00,
    /* 0x2F / */
    0x00,0x20,0x10,0x08,0x04,0x02,0x00,0x00,
    /* 0x30 0 */
    0x00,0x3E,0x51,0x49,0x45,0x3E,0x00,0x00,
    /* 0x31 1 */
    0x00,0x00,0x42,0x7F,0x40,0x00,0x00,0x00,
    /* 0x32 2 */
    0x00,0x62,0x51,0x49,0x49,0x46,0x00,0x00,
    /* 0x33 3 */
    0x00,0x22,0x41,0x49,0x49,0x36,0x00,0x00,
    /* 0x34 4 */
    0x00,0x18,0x14,0x12,0x7F,0x10,0x00,0x00,
    /* 0x35 5 */
    0x00,0x27,0x45,0x45,0x45,0x39,0x00,0x00,
    /* 0x36 6 */
    0x00,0x3C,0x4A,0x49,0x49,0x30,0x00,0x00,
    /* 0x37 7 */
    0x00,0x01,0x71,0x09,0x05,0x03,0x00,0x00,
    /* 0x38 8 */
    0x00,0x36,0x49,0x49,0x49,0x36,0x00,0x00,
    /* 0x39 9 */
    0x00,0x06,0x49,0x49,0x29,0x1E,0x00,0x00,
    /* 0x3A : */
    0x00,0x00,0x36,0x36,0x00,0x00,0x00,0x00,
    /* 0x3B ; */
    0x00,0x00,0x56,0x36,0x00,0x00,0x00,0x00,
    /* 0x3C < */
    0x00,0x08,0x14,0x22,0x41,0x00,0x00,0x00,
    /* 0x3D = */
    0x00,0x14,0x14,0x14,0x14,0x14,0x00,0x00,
    /* 0x3E > */
    0x00,0x41,0x22,0x14,0x08,0x00,0x00,0x00,
    /* 0x3F ? */
    0x00,0x02,0x01,0x51,0x09,0x06,0x00,0x00,
    /* 0x40 @ */
    0x00,0x32,0x49,0x79,0x41,0x3E,0x00,0x00,
    /* 0x41 A */
    0x00,0x7E,0x09,0x09,0x09,0x7E,0x00,0x00,
    /* 0x42 B */
    0x00,0x7F,0x49,0x49,0x49,0x36,0x00,0x00,
    /* 0x43 C */
    0x00,0x3E,0x41,0x41,0x41,0x22,0x00,0x00,
    /* 0x44 D */
    0x00,0x7F,0x41,0x41,0x22,0x1C,0x00,0x00,
    /* 0x45 E */
    0x00,0x7F,0x49,0x49,0x49,0x41,0x00,0x00,
    /* 0x46 F */
    0x00,0x7F,0x09,0x09,0x09,0x01,0x00,0x00,
    /* 0x47 G */
    0x00,0x3E,0x41,0x49,0x49,0x7A,0x00,0x00,
    /* 0x48 H */
    0x00,0x7F,0x08,0x08,0x08,0x7F,0x00,0x00,
    /* 0x49 I */
    0x00,0x00,0x41,0x7F,0x41,0x00,0x00,0x00,
    /* 0x4A J */
    0x00,0x20,0x40,0x41,0x3F,0x01,0x00,0x00,
    /* 0x4B K */
    0x00,0x7F,0x08,0x14,0x22,0x41,0x00,0x00,
    /* 0x4C L */
    0x00,0x7F,0x40,0x40,0x40,0x40,0x00,0x00,
    /* 0x4D M */
    0x00,0x7F,0x02,0x0C,0x02,0x7F,0x00,0x00,
    /* 0x4E N */
    0x00,0x7F,0x04,0x08,0x10,0x7F,0x00,0x00,
    /* 0x4F O */
    0x00,0x3E,0x41,0x41,0x41,0x3E,0x00,0x00,
    /* 0x50 P */
    0x00,0x7F,0x09,0x09,0x09,0x06,0x00,0x00,
    /* 0x51 Q */
    0x00,0x3E,0x41,0x51,0x21,0x5E,0x00,0x00,
    /* 0x52 R */
    0x00,0x7F,0x09,0x19,0x29,0x46,0x00,0x00,
    /* 0x53 S */
    0x00,0x26,0x49,0x49,0x49,0x32,0x00,0x00,
    /* 0x54 T */
    0x00,0x01,0x01,0x7F,0x01,0x01,0x00,0x00,
    /* 0x55 U */
    0x00,0x3F,0x40,0x40,0x40,0x3F,0x00,0x00,
    /* 0x56 V */
    0x00,0x07,0x18,0x60,0x18,0x07,0x00,0x00,
    /* 0x57 W */
    0x00,0x7F,0x20,0x18,0x20,0x7F,0x00,0x00,
    /* 0x58 X */
    0x00,0x63,0x14,0x08,0x14,0x63,0x00,0x00,
    /* 0x59 Y */
    0x00,0x03,0x04,0x78,0x04,0x03,0x00,0x00,
    /* 0x5A Z */
    0x00,0x61,0x51,0x49,0x45,0x43,0x00,0x00,
};

static void oled_glyph(char c, uint8_t page, uint8_t col)
{
    if (c < 0x20 || c > 0x5A) c = 0x20;
    uint16_t idx = (uint16_t)(c - 0x20) * 8;
    oled_set_cursor(page, col);
    oled_data_burst(&font8x8[idx], 8);
}

static void oled_str(const char *s, uint8_t page, uint8_t col)
{
    while (*s) {
        oled_glyph(*s++, page, col);
        col += 8;
    }
}

/* (inline-number display used in main loop below) */

/* ═══════════════════════════════════════════════════════════════
 * Servo PWM — TIM2 CH2 on PA1
 * ─────────────────────────────────────────────────────────────
 * TIM2 clock = 72MHz  (APB1 timer clock = PCLK1×2 = 36×2)
 * PSC  = 71     → tick = 1 µs
 * ARR  = 19999  → period = 20 ms (50 Hz)
 * CCR2 = 500–2500 → pulse 0.5–2.5 ms → angle 0°–180°
 * ══════════════════════════════════════════════════════════════ */

static void servo_init(void)
{
    RCC_APB2ENR |= RCC_APB2ENR_IOPAEN;
    RCC_APB1ENR |= RCC_APB1ENR_TIM2EN;

    /* PA1 → AF push-pull, 50MHz (CRL bits [7:4]) */
    GPIOA_CRL &= ~(0xFU << 4);
    GPIOA_CRL |=  (0xBU << 4);   /* CNF=10 (AF PP), MODE=11 */

    TIM2_PSC  = 71;              /* 1 MHz */
    TIM2_ARR  = 19999;           /* 20 ms  */

    /* CH2 → PWM mode 1 (CCMR1 bits [15:8]):
     *   OC2M[2:0] = 110, OC2PE = 1 */
    TIM2_CCMR1 &= ~(0xFFU << 8);
    TIM2_CCMR1 |=  (0x68U << 8);

    /* Enable CH2 output (CCER bit 4) */
    TIM2_CCER  |=  (1U << 4);

    /* Start at mid position (90°) */
    TIM2_CCR2  = 1500;

    /* Generate update + enable counter */
    TIM2_EGR   |=  (1U << 0);
    TIM2_CR1   |=  (1U << 0);
}

static void servo_set_angle(uint8_t angle)
{
    if (angle > 180) angle = 180;
    /* 0°→500µs, 180°→2500µs */
    uint32_t pulse = 500 + (uint32_t)angle * 2000 / 180;
    TIM2_CCR2 = pulse;
}

/* ═══════════════════════════════════════════════════════════════
 * Main
 * ══════════════════════════════════════════════════════════════ */

int main(void)
{
    clock_init();
    delay_init();
    i2c_init();

    servo_init();
    encoder_init();
    oled_init();
    oled_clear();

    delay_ms(50);

    while (1) {
        /* ── Determine servo angle ──────────────────────── */
        uint32_t raw = remote_angle;
        uint8_t  angle;
        int      pc_mode = 0;   /* 1 = PC remote, 0 = encoder */

        if (raw <= 180) {
            /* PC remote angle is valid */
            angle   = (uint8_t)raw;
            pc_mode = 1;
            /* Reset encoder accumulator when PC takes over */
            enc_count = 0;
        } else {
            /* Encoder mode — accumulate delta */
            static int32_t num = 0;
            int32_t delta = enc_count;
            enc_count = 0;
            num += delta;
            int32_t a = num + 90;
            if (a < 0)   a = 0;
            if (a > 180) a = 180;
            angle = (uint8_t)a;
        }

        servo_set_angle(angle);

        /* ── OLED display ───────────────────────────────── */
        /* Line 1 (page 1): source mode */
        oled_str(pc_mode ? "PC REMOTE" : "ENCODER ", 1, 0);

        /* Line 2 (page 3): "ANGLE: xxx DEG" */
        oled_str("ANGLE:", 3, 0);
        {
            char buf[12];
            int i = 0;
            uint32_t v = (uint32_t)angle;
            if (v == 0) {
                oled_glyph('0', 3, 56);
            } else {
                while (v > 0) { buf[i++] = '0' + (v % 10); v /= 10; }
                uint8_t c = 56;
                while (i > 0) { oled_glyph(buf[--i], 3, c); c += 8; }
            }
        }
        oled_str("DEG", 3, 80);

        /* Line 3 (page 5): progress bar */
        {
            uint8_t bar_w = (uint8_t)((uint32_t)angle * 112 / 180);
            uint8_t bar[112];
            for (int i = 0; i < 112; i++)
                bar[i] = (i < bar_w) ? 0x7F : 0x00;
            oled_set_cursor(5, 8);
            oled_data_burst(bar, 112);
        }

        delay_ms(20);   /* 50Hz update rate */
    }

    return 0;
}
