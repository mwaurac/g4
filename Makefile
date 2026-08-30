CC ?= gcc
TARGET ?= g4
BUILD_DIR ?= build

SRCS = main.c


# flags
CFLAGS_BASE = -Wall -Wextra -Wpedantic -std=c11
CFLAGS_RELEASE = -O3 -DNDEBUG
CFLAGS_DEBUG   = -g -O0 -DDEBUG

# mode selection
MODE ?= release
ifneq ($(filter debug,$(MAKECMDGOALS)),)
    MODE = debug
endif

CFLAGS = $(CFLAGS_BASE)

ifeq ($(MODE),release)
    CFLAGS += $(CFLAGS_RELEASE)
else ifeq ($(MODE),debug)
    CFLAGS += $(CFLAGS_DEBUG)
else
    $(error Unknown MODE '$(MODE)'. Valid modes: release, debug)
endif

# build rules
BIN_DIR = $(BUILD_DIR)/$(MODE)
BIN = $(BIN_DIR)/$(TARGET)

OBJS = $(patsubst %.c,$(BIN_DIR)/%.o,$(SRCS))

.PHONY: all debug run clean help

all: $(BIN)

debug: $(BIN)

$(BIN): $(OBJS)
	@echo "Linking $(BIN) [$(MODE)]..."
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(BIN_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

run: $(BIN)
	$(BIN)

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Available targets:"
	@echo "  make            : Build release (default)"
	@echo "  make debug      : Build with debug symbols"
	@echo "  make run        : Build and run the current mode"
	@echo "  make clean      : Remove all build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  MODE=release|debug (or pass as a target)"
	@echo "  CC=... TARGET=... BUILD_DIR=..."
