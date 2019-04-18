PYTHON=python3

# Halide utilities
HALIDE_DIR = gradient-halide
HALIDE_GEN = $(HALIDE_DIR)/tools/GenGen.cpp

# C++ compile options
CXX ?= g++
CXXFLAGS += -std=c++11 -fno-rtti
INCLUDE=-I$(HALIDE_DIR)/include/
LDFLAGS ?=
LDFLAGS += $(HALIDE_DIR)/lib/libHalide.a -lpthread -ldl -lcurses -lz

# Cuda options
NVCC = nvcc -std c++11 -O2 #-G  -pg
NVFLAGS = -x cu -Xcompiler -fPIC -I$(SRC_DIR) \
		  -gencode=arch=compute_30,code=\"sm_30,compute_30\" \
		  -expt-relaxed-constexpr -Wno-deprecated-gpu-targets \
		  -ftz=true --ptxas-options=-v -lineinfo
CUDA_LDFLAGS= -L/usr/local/cuda/lib64 -lcuda -lcudart

AUTO_SCHEDULE = false
TARGET_FEATURES ?=

BUILD_TYPE ?= release
ifeq ($(BUILD_TYPE), profile)
TARGET_FEATURES = -profile
endif

ifeq ($(BUILD_TYPE), debug)
CXXFLAGS += -g -rdynamic
NVFLAGS += -g -G -lineinfo
# TARGET_FEATURES += -debug
else
CXXFLAGS += -O3
endif

ifeq ($(UNAME), Darwin)
CXXFLAGS += -fvisibility=hidden
endif

ifeq ($(UNAME), Darwin)
DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH):$(HALIDE_DIR)/bin
else
endif

# Directories
SRC_DIR = src
BUILD_DIR = build
EXT_DIR = dist

# The makefile assumes that each op that has a backward also has a forward
OPS = scatter2gather_forward \
	  kernel_weighting_forward \
	  kernel_weighting_backward \
	  kernel_lookup_forward \
	  kernel_lookup_backward

OPS_LIBS = $(addsuffix .a, $(addprefix $(BUILD_DIR)/, $(OPS)))

# Check we have cuda
ifeq ($(shell which nvcc),)
HAS_CUDA=0
CUDA_OPS_LIBS = 
CUDA_SRC =
CUDA_OBJ =
else
HAS_CUDA=1
CUDA_OPS_LIBS = $(addsuffix _cuda.a, $(addprefix $(BUILD_DIR)/, $(OPS)))
# CUDA_SRC = bilateral_slice
# CUDA_OBJ = $(addsuffix _cuda.so, $(addprefix $(BUILD_DIR)/, $(CUDA_SRC)))
endif

# Assemble all the custom operators in a single shared library to be loaded by Python
_ext/operators/_operators.so: $(CUDA_OPS_LIBS) $(CUDA_OBJ) $(OPS_LIBS)
	@echo Installing python module
	@$(PYTHON) setup.py install #2>&1 > /dev/null

# Rule for pure cuda kernels
# $(BUILD_DIR)/%_cuda.so: $(SRC_DIR)/cuda_kernels/%.cu.cc $(SRC_DIR)/cuda_kernels/%.h $(BUILD_DIR)
# 		$(NVCC) -c  $< -o $@ $(NVFLAGS) $(TORCH_INC)
#
# TODO(mgharbi): auto generate float/double variants

$(BUILD_DIR)/%.a: $(BUILD_DIR)/%
	@echo Building Op $(subst $(BUILD_DIR)/,, $@)
	@cd $(BUILD_DIR); \
	DYLD_LIBRARY_PATH=$(HALIDE_DIR)/bin LD_LIBRARY_PATH=$(HALIDE_DIR)/bin FLUSH_MEMOIZE_CACHE=0 \
										./$(basename $(@F)) -g $(basename $(@F)) \
										-e static_library,h,pytorch_wrapper\
										-o . target=host$(TARGET_FEATURES)\
										auto_schedule=$(AUTO_SCHEDULE)

$(BUILD_DIR)/%_cuda.a: $(BUILD_DIR)/%
	@$(eval name = $(subst _cuda,,$(basename $(@F))))
	@echo "Building Op (CUDA) $(name)"
	@cd $(BUILD_DIR); \
	DYLD_LIBRARY_PATH=$(HALIDE_DIR)/bin LD_LIBRARY_PATH=$(HALIDE_DIR)/bin FLUSH_MEMOIZE_CACHE=0 \
										./$(name) -g $(name) -f $(name)_cuda \
										-e static_library,h,pytorch_wrapper\
										-o . target=host-cuda-cuda_capability_61-user_context$(TARGET_FEATURES) \
										auto_schedule=$(AUTO_SCHEDULE)


# Generators, prevent auto-deletion of intermediate files by make
GENERATORS = $(addprefix $(BUILD_DIR)/, $(OPS))
.SECONDARY: $(GENERATORS)

$(BUILD_DIR)/%: $(SRC_DIR)/%.hl.cxx $(HALIDE_GEN) gradient-halide/lib/libHalide.a
	@echo Building Generator $(subst $(BUILD_DIR)/,, $@)
	@mkdir -p $(BUILD_DIR)
	@$(CXX) $(HALIDE_GEN) $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) -MMD -MP -o $(basename $@)

$(BUILD_DIR):
	@echo Making $@ dir
	@mkdir -p $@

clean:
	$(RM) -r $(BUILD_DIR) $(EXT_DIR)

submodules:
	@echo "Updating submodules"
	@git submodule init
	@git submodule update --recursive

gradient-halide/lib/libHalide.a: submodules
	@echo "Building gradient halide"
	@$(MAKE) -j distrib -C gradient-halide

.PHONY: submodules

DEPS = $(wildcard $(BUILD_DIR)/*.d)
-include $(DEPS)
