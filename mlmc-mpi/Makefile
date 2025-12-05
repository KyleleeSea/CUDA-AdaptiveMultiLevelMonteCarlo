###########################################################
# Project settings
###########################################################

EXECUTABLE      := test-suite
CLI_EXECUTABLE  := mlmc-cli

CU_FILES   := mlmc-cuda.cu
CU_DEPS    :=
CC_FILES   := test-suite.cpp mlmc-cli.cpp

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

# Extra libraries (none for now)
LIBS       :=
FRAMEWORKS :=

LDLIBS        := $(addprefix -l, $(LIBS))
LDFRAMEWORKS  := $(addprefix -framework , $(FRAMEWORKS))

###########################################################
# Objects
###########################################################

OBJS        = $(OBJDIR)/test-suite.o $(OBJDIR)/mlmc-cuda.o
CLI_OBJS    = $(OBJDIR)/mlmc-cli.o  $(OBJDIR)/mlmc-cuda.o

###########################################################
# Targets
###########################################################

.PHONY: all dirs clean

all: $(EXECUTABLE) $(CLI_EXECUTABLE)

default: all

dirs:
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(CLI_EXECUTABLE) $(LOGS)

###########################################################
# Linking
###########################################################

$(EXECUTABLE): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(CLI_EXECUTABLE): dirs $(CLI_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(CLI_OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

###########################################################
# Compilation rules
###########################################################

$(OBJDIR)/%.o: %.cpp mlmc-cuda.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.cu mlmc-cuda.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
