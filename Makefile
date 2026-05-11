# pso-cuda — Makefile
# Toolchain: nvcc. Target: Quadro RTX 6000 (Turing, sm_75).

# ---- toolchain ---------------------------------------------------------------
NVCC      ?= nvcc
SM        ?= 75
STD       ?= c++17

# ---- layout ------------------------------------------------------------------
PSO_DIR    := pso
EVAL_DIR   := evals
BUILD_DIR  := build
BIN        := $(BUILD_DIR)/pso_cuda

PSO_SRCS   := $(wildcard $(PSO_DIR)/*.cu)
EVAL_SRCS  := $(wildcard $(EVAL_DIR)/*.cu)

ifeq ($(wildcard $(PSO_DIR)/main.cu),)
  $(error $(PSO_DIR)/main.cu not found in $(CURDIR) — sync it from the dev box)
endif

PSO_OBJS   := $(patsubst $(PSO_DIR)/%.cu,$(BUILD_DIR)/$(PSO_DIR)/%.o,$(PSO_SRCS))
EVAL_OBJS  := $(patsubst $(EVAL_DIR)/%.cu,$(BUILD_DIR)/$(EVAL_DIR)/%.o,$(EVAL_SRCS))
OBJS       := $(PSO_OBJS) $(EVAL_OBJS)

# ---- flags -------------------------------------------------------------------
INCLUDES  := -I$(PSO_DIR) -I$(EVAL_DIR)
ARCHFLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)

NVCCFLAGS := -std=$(STD) $(ARCHFLAGS) $(INCLUDES) \
             --expt-relaxed-constexpr \
             -Xcompiler -Wall,-Wextra

BUILD ?= release
ifeq ($(BUILD),debug)
  NVCCFLAGS += -O0 -g -G -lineinfo -DDEBUG
else
  NVCCFLAGS += -O3 -lineinfo
endif

LDLIBS := -lcurand

# ---- rules -------------------------------------------------------------------
.PHONY: all clean run debug release info

all: $(BIN)

$(BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(ARCHFLAGS) $(OBJS) -o $@ $(LDLIBS)

$(BUILD_DIR)/$(PSO_DIR)/%.o: $(PSO_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BUILD_DIR)/$(EVAL_DIR)/%.o: $(EVAL_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

run: $(BIN)
	./$(BIN)

debug:
	$(MAKE) BUILD=debug

release:
	$(MAKE) BUILD=release

info:
	@echo "PSO_SRCS  = $(PSO_SRCS)"
	@echo "EVAL_SRCS = $(EVAL_SRCS)"
	@echo "OBJS      = $(OBJS)"
	@echo "NVCCFLAGS = $(NVCCFLAGS)"

clean:
	rm -rf $(BUILD_DIR)
