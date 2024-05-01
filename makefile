# Compiler settings - Can be customized.
CXX = g++-mp-12
CXXFLAGS = -O3
LDFLAGS = 

# Source file and executable names
SRC = source/hiddenfoot.cpp
EXEC_WITH_OMP = hiddenfoot_omp
EXEC_WITHOUT_OMP = hiddenfoot
DIR_BIN = binary/

# OpenMP specific settings
OMP_FLAGS = -Xpreprocessor -fopenmp
OMP_LIBS = -L/opt/local/lib/libomp/ -lomp
OMP_INCS = -I/opt/local/include/libomp/
OMP_DEF = -DUSE_OMP

# Default target
all: $(EXEC_WITH_OMP) $(EXEC_WITHOUT_OMP)

# Target to compile with OpenMP
$(EXEC_WITH_OMP): $(SRC)
	$(CXX) $(CXXFLAGS) $(OMP_DEF) $(OMP_FLAGS) $(OMP_INCS) $(SRC) -o $(DIR_BIN)$@ $(OMP_LIBS) $(LDFLAGS)

# Target to compile without OpenMP
$(EXEC_WITHOUT_OMP): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(DIR_BIN)$@ $(LDFLAGS)

# Clean objects and executables
clean:
	rm -f $(EXEC_WITH_OMP) $(EXEC_WITHOUT_OMP)
