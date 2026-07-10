/* Linker script for STM32F103C8T6
 * Flash: 64KB, SRAM: 20KB
 */

MEMORY
{
    FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 64K
    SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 20K
}

_estack = ORIGIN(SRAM) + LENGTH(SRAM);  /* stack grows down from end of SRAM */

SECTIONS
{
    /* Vector table and code in Flash */
    .text :
    {
        KEEP(*(.vectors))     /* vector table must be first */
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
        _etext = .;
    } > FLASH

    /* Remote control section — fixed at SRAM start (0x20000000)
     * NOT initialized: survives watchdog resets, writable by OpenOCD */
    .remote 0x20000000 (NOLOAD) :
    {
        KEEP(*(.remote))
        . = ALIGN(4);
    } > SRAM

    /* Initialized data: copied from Flash to SRAM at startup */
    _sidata = LOADADDR(.data);
    .data : AT (_etext)
    {
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > SRAM

    /* Uninitialized (zero-filled) data */
    .bss :
    {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > SRAM
}
