mpi = mpiicpx
C++ = g++

CFLAGS = -fopenmp -std=c++17 -DMKL_ILP64  -m64  -I"${MKLROOT}/include" -I../utils/headers -O3
CFLAGSDISTR = -qopenmp -std=c++17 -DMKL_ILP64  -m64  -I"${MKLROOT}/include" -I../utils/headers -O3
DEBUG = -g

TARGETDISTR = exp-distr
TARGETSHARED = exp-shared
TARGETTEST = test

LINK = -m64  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

GREEN = \033[0;32m
RED = \033[0;31m
YELLOW = \033[0;33m
NORMAL = \033[0m

OBJDIR = obj
OBJDIR_DISTR = obj_distr

# Source files
SRCS_SHARED = ../utils/csr_matrix.cpp ../utils/dense_vector.cpp ../utils/dense_matrix.cpp \
              ../utils/arnoldi_iteration_shared.cpp ../utils/io_ops.cpp ../utils/scaling_and_squaring.cpp \
              ../utils/mtx_ops_mkl.cpp exponential_mtx_shared.cpp

SRCS_DISTR = ../utils/csr_matrix.cpp ../utils/dense_vector.cpp ../utils/dense_matrix.cpp \
             ../utils/arnoldi_iteration.cpp ../utils/distr_mtx_ops.cpp ../utils/help_process.cpp \
             ../utils/io_ops.cpp ../utils/scaling_and_squaring.cpp ../utils/mtx_ops_mkl.cpp \
             exponential_mtx_distr.cpp

# Object files
OBJS_SHARED = $(patsubst %.cpp, $(OBJDIR)/%.o, $(notdir $(SRCS_SHARED)))
OBJS_DISTR = $(patsubst %.cpp, $(OBJDIR_DISTR)/%.o, $(notdir $(SRCS_DISTR)))

all: distr shared

# Create OBJDIR and OBJDIR_DISTR if they don't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR_DISTR):
	mkdir -p $(OBJDIR_DISTR)

# Pattern rules for object files
$(OBJDIR)/%.o: ../utils/%.cpp | $(OBJDIR)
	$(C++) $(CFLAGS) -c $< -o $@

$(OBJDIR_DISTR)/%.o: ../utils/%.cpp | $(OBJDIR_DISTR)
	$(mpi) $(CFLAGSDISTR) -c $< -o $@

$(OBJDIR)/exponential_mtx_shared.o: exponential_mtx_shared.cpp | $(OBJDIR)
	$(C++) $(CFLAGS) -c $< -o $@

$(OBJDIR_DISTR)/exponential_mtx_distr.o: exponential_mtx_distr.cpp | $(OBJDIR_DISTR)
	$(mpi) $(CFLAGSDISTR) -c $< -o $@

# Targets

distr: $(OBJDIR_DISTR) $(OBJS_DISTR)
	$(mpi) $(CFLAGSDISTR) $(LINK) $(OBJS_DISTR) -o $(TARGETDISTR)
	@echo "$(GREEN)Compiled exponential computation distributed memory version: $(TARGETDISTR) $(NORMAL)"


shared: $(OBJDIR) $(OBJS_SHARED)
	$(C++) $(CFLAGS) $(LINK) $(OBJS_SHARED) -o $(TARGETSHARED)
	@echo "$(GREEN)Compiled exponential computation shared memory version: $(TARGETSHARED) $(NORMAL)"


clean:
	@echo "$(RED)Warning: deleting executables and binary files!$(NORMAL)"
	rm -rf $(OBJDIR) $(OBJDIR_DISTR) $(TARGETDISTR) $(TARGETSHARED) $(TARGETTEST)

.PHONY: clean
