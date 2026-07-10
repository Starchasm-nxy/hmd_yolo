# Makefile for STM32F103C8T6 LED Blink

# Toolchain
CC      = arm-none-eabi-gcc
OBJCOPY = arm-none-eabi-objcopy
SIZE    = arm-none-eabi-size

# Files
TARGET  = firmware
SRCS    = main.c startup_stm32f103c8t6.s
OBJS    = main.o startup_stm32f103c8t6.o
LDSCRIPT = stm32f103c8t6.ld

# Flags
CFLAGS   = -mcpu=cortex-m3 -mthumb -nostdlib -O2 -g -Wall
LDFLAGS  = -mcpu=cortex-m3 -mthumb -T$(LDSCRIPT) -nostdlib -Wl,-Map=$(TARGET).map

# ── Targets ──────────────────────────────────────────────

.PHONY: all flash clean

all: $(TARGET).elf $(TARGET).bin
	@echo "=== Build complete ==="
	@$(SIZE) $(TARGET).elf

$(TARGET).elf: $(OBJS) $(LDSCRIPT)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)

$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.s
	$(CC) $(CFLAGS) -c -o $@ $<

flash: $(TARGET).bin
	st-flash write $(TARGET).bin 0x08000000

clean:
	rm -f *.o *.elf *.bin *.map
