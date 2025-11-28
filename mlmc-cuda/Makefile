###########################################################
# Project settings
###########################################################

EXECUTABLE := test-suite

CU_FILES   := mlmc-cuda.cu
CU_DEPS    :=
CC_FILES   := test-suite.cpp

OBJDIR     = objs
LOGS       := logs

###########################################################
# Compiler setup
###########################################################

CXX        = g++-11 -m64
CXXFLAGS   = -O3 -Wall -g

NVCC       = nvcc
NVCCFLAGS  = -O3 -m64 --gpu-architecture compute_61 --allow-unsupported-compiler \
             -ccbin /usr/bin/g++-11

# CUDA libs
LDFLAGS    = -L/usr/local/cuda-11.7/lib64/ -lcudart

# Extra libraries (none for now, but keep the format)
LIBS       :=
FRAMEWORKS :=

LDLIBS        := $(addprefix -l, $(LIBS))
LDFRAMEWORKS  := $(addprefix -framework , $(FRAMEWORKS))

###########################################################
# Objects
###########################################################

OBJS = $(OBJDIR)/test-suite.o $(OBJDIR)/mlmc-cuda.o

###########################################################
# Targets
###########################################################

.PHONY: all dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS)

###########################################################
# Linking
###########################################################

$(EXECUTABLE): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

###########################################################
# Compilation rules
###########################################################

$(OBJDIR)/%.o: %.cpp mlmc-cuda.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.cu mlmc-cuda.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@