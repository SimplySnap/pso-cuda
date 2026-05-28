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
MPI_DIR   := mpi
BUILD_DIR := build

BIN       := $(BUILD_DIR)/pso_cuda
RING_BIN  := $(BUILD_DIR)/pso_ring
FC_BIN    := $(BUILD_DIR)/pso_fc
CPU_BIN   := $(BUILD_DIR)/pso_cpu

#exclude main.cu from library objects — entry point lives in src/ now
PSO_SRCS  := $(filter-out $(PSO_DIR)/main.cu, $(wildcard $(PSO_DIR)/*.cu))
EVAL_SRCS := $(wildcard $(EVAL_DIR)/*.cu)
MAIN_SRC  := $(SRC_DIR)/main.cu
RING_SRC  := $(SRC_DIR)/main_ring.cu
FC_SRC    := $(SRC_DIR)/main_fc.cu
MPI_SRCS  := $(wildcard $(MPI_DIR)/*.cu)

ifeq ($(wildcard $(MAIN_SRC)),)
  $(error $(MAIN_SRC) not found — did you move pso/main.cu to src/main.cu?)
endif
ifeq ($(wildcard $(RING_SRC)),)
  $(error $(RING_SRC) not found)
endif
ifeq ($(wildcard $(FC_SRC)),)
  $(error $(FC_SRC) not found)
endif

PSO_OBJS  := $(patsubst $(PSO_DIR)/%.cu,  $(BUILD_DIR)/$(PSO_DIR)/%.o,  $(PSO_SRCS))
EVAL_OBJS := $(patsubst $(EVAL_DIR)/%.cu, $(BUILD_DIR)/$(EVAL_DIR)/%.o, $(EVAL_SRCS))
MAIN_OBJ  := $(BUILD_DIR)/$(SRC_DIR)/main.o
RING_OBJ  := $(BUILD_DIR)/$(SRC_DIR)/main_ring.o
FC_OBJ    := $(BUILD_DIR)/$(SRC_DIR)/main_fc.o
MPI_OBJS  := $(patsubst $(MPI_DIR)/%.cu,  $(BUILD_DIR)/$(MPI_DIR)/%.o,  $(MPI_SRCS))

#shared library objects used by all three binaries
LIB_OBJS  := $(PSO_OBJS) $(EVAL_OBJS)

# ---- flags -------------------------------------------------------------------
INCLUDES  := -I$(PSO_DIR) -I$(EVAL_DIR) -I$(SRC_DIR)
ARCHFLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)

#query MPI compile/link flags from the wrapper so we adapt to whichever
#MPI module is loaded (openmpi, hpcx, mpich, etc). Keep only the -I and -L
#options — nvcc handles those natively; the wrapper's -pthread / -Wl,-rpath
#noise is dropped for portability and rediscovered automatically by -lmpi.
MPI_INC := $(filter -I%,$(shell mpicxx --showme:compile 2>/dev/null))
MPI_LIB := $(filter -L%,$(shell mpicxx --showme:link    2>/dev/null)) -lmpi

NVCCFLAGS := -std=$(STD) $(ARCHFLAGS) $(INCLUDES) \
             --expt-relaxed-constexpr \
             -Xcompiler -Wall,-Wextra
NVCCFLAGS_MPI := $(NVCCFLAGS) $(MPI_INC)

CXXFLAGS  := -std=$(STD) -O3 -Wall -Wextra

BUILD ?= release
ifeq ($(BUILD),debug)
  NVCCFLAGS     += -O0 -g -G -lineinfo -DDEBUG
  NVCCFLAGS_MPI += -O0 -g -G -lineinfo -DDEBUG
else
  NVCCFLAGS     += -O3 -lineinfo
  NVCCFLAGS_MPI += -O3 -lineinfo
endif

LDLIBS     := -lcurand
LDLIBS_MPI := -lcurand $(MPI_LIB)

# ---- rules -------------------------------------------------------------------
.PHONY: all mpi clean run ring fc cpu bench-cpu debug release info

#default: single-GPU only (no MPI dependency)
all: $(BIN)

#build all three binaries
mpi: $(BIN) $(RING_BIN) $(FC_BIN)

# ---- single-GPU --------------------------------------------------------------
$(BIN): $(LIB_OBJS) $(MAIN_OBJ)
	@mkdir -p $(dir $@)
	$(NVCC) $(ARCHFLAGS) $^ -o $@ $(LDLIBS)

$(MAIN_OBJ): $(MAIN_SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# ---- ring topology -----------------------------------------------------------
$(RING_BIN): $(LIB_OBJS) $(MPI_OBJS) $(RING_OBJ)
	@mkdir -p $(dir $@)
	$(NVCC) $(ARCHFLAGS) $^ -o $@ $(LDLIBS_MPI)

$(RING_OBJ): $(RING_SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS_MPI) -dc $< -o $@

# ---- fully connected ---------------------------------------------------------
$(FC_BIN): $(LIB_OBJS) $(MPI_OBJS) $(FC_OBJ)
	@mkdir -p $(dir $@)
	$(NVCC) $(ARCHFLAGS) $^ -o $@ $(LDLIBS_MPI)

$(FC_OBJ): $(FC_SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS_MPI) -dc $< -o $@

# ---- shared pattern rules ----------------------------------------------------
$(BUILD_DIR)/$(PSO_DIR)/%.o: $(PSO_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BUILD_DIR)/$(EVAL_DIR)/%.o: $(EVAL_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BUILD_DIR)/$(MPI_DIR)/%.o: $(MPI_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS_MPI) -dc $< -o $@ 
# ---- convenience targets -----------------------------------------------------
run: $(BIN)
	./$(BIN)

ring: $(RING_BIN)

fc: $(FC_BIN)

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
	@echo "PSO_SRCS      = $(PSO_SRCS)"
	@echo "EVAL_SRCS     = $(EVAL_SRCS)"
	@echo "MPI_SRCS      = $(MPI_SRCS)"
	@echo "LIB_OBJS      = $(LIB_OBJS)"
	@echo "MPI_INC       = $(MPI_INC)"
	@echo "MPI_LIB       = $(MPI_LIB)"
	@echo "LDLIBS_MPI    = $(LDLIBS_MPI)"
	@echo "NVCCFLAGS     = $(NVCCFLAGS)"
	@echo "NVCCFLAGS_MPI = $(NVCCFLAGS_MPI)"

clean:
	rm -rf $(BUILD_DIR)