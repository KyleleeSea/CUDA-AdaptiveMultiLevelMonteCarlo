# Makefile for MLMC Sequential Test Suite

# Compiler and flags
CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra

# Executable name
TARGET = test-suite

# Source files
SRCS = mlmc-sequential.cpp test-suite.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target builds the test suite
all: $(TARGET)

# Build test-suite executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f *.o $(TARGET)
