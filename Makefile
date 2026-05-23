#pso-cuda — Makefile
#toolchain: nvcc. target: Quadro RTX 6000 (Turing, sm_75).

# ---- toolchain ---------------------------------------------------------------
NVCC      ?= nvcc
CXX       ?= g++
SM        ?= 75
STD       ?= c++17

# ---- layout ------------------------------------------------------------------
PSO_DIR   := pso
EVAL_DIR  := evals
SRC_DIR   := src
BUILD_DIR := build
BIN       := $(BUILD_DIR)/pso_cuda
CPU_BIN   := $(BUILD_DIR)/pso_cpu

#exclude main.cu from library objects — entry point lives in src/ now
PSO_SRCS  := $(filter-out $(PSO_DIR)/main.cu, $(wildcard $(PSO_DIR)/*.cu))
EVAL_SRCS := $(wildcard $(EVAL_DIR)/*.cu)
MAIN_SRC  := $(SRC_DIR)/main.cu

ifeq ($(wildcard $(MAIN_SRC)),)
  $(error $(MAIN_SRC) not found — did you move pso/main.cu to src/main.cu?)
endif

PSO_OBJS  := $(patsubst $(PSO_DIR)/%.cu,  $(BUILD_DIR)/$(PSO_DIR)/%.o,  $(PSO_SRCS))
EVAL_OBJS := $(patsubst $(EVAL_DIR)/%.cu, $(BUILD_DIR)/$(EVAL_DIR)/%.o, $(EVAL_SRCS))
MAIN_OBJ  := $(BUILD_DIR)/$(SRC_DIR)/main.o
OBJS      := $(PSO_OBJS) $(EVAL_OBJS) $(MAIN_OBJ)

# ---- flags -------------------------------------------------------------------
INCLUDES  := -I$(PSO_DIR) -I$(EVAL_DIR) -I$(SRC_DIR)
ARCHFLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)

NVCCFLAGS := -std=$(STD) $(ARCHFLAGS) $(INCLUDES) \
             --expt-relaxed-constexpr \
             -Xcompiler -Wall,-Wextra
CXXFLAGS  := -std=$(STD) -O3 -Wall -Wextra

BUILD ?= release
ifeq ($(BUILD),debug)
  NVCCFLAGS += -O0 -g -G -lineinfo -DDEBUG
else
  NVCCFLAGS += -O3 -lineinfo
endif

LDLIBS := -lcurand

# ---- rules -------------------------------------------------------------------
.PHONY: all clean run cpu bench-cpu debug release info

all: $(BIN)

$(BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(ARCHFLAGS) $(OBJS) -o $@ $(LDLIBS)

#pso library objects
$(BUILD_DIR)/$(PSO_DIR)/%.o: $(PSO_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

#eval objects
$(BUILD_DIR)/$(EVAL_DIR)/%.o: $(EVAL_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

#single-GPU entry point
$(MAIN_OBJ): $(MAIN_SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

#TODO(MPI): add pso_ring and pso_fc targets here that pull in mpi/ and link -lmpi

run: $(BIN)
	./$(BIN)

cpu: $(CPU_BIN)

$(CPU_BIN): bench/pso_cpu.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $< -o $@

bench-cpu: $(CPU_BIN)
	./$(CPU_BIN) 1024 30 100 1234

debug:
	$(MAKE) BUILD=debug

release:
	$(MAKE) BUILD=release

info:
	@echo "PSO_SRCS  = $(PSO_SRCS)"
	@echo "EVAL_SRCS = $(EVAL_SRCS)"
	@echo "MAIN_SRC  = $(MAIN_SRC)"
	@echo "OBJS      = $(OBJS)"
	@echo "NVCCFLAGS = $(NVCCFLAGS)"

clean:
	rm -rf $(BUILD_DIR)