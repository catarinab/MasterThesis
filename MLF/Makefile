mpi = mpiicpx
C++ = g++

CFLAGS= -fopenmp -std=c++17 -DMKL_ILP64  -m64  -I"${MKLROOT}/include" -march=native -O3
CFLAGSDISTR= -qopenmp -std=c++17 -DMKL_ILP64  -m64  -I"${MKLROOT}/include" -O3
CFLAGSEigen= -I ../eigen
CFLAGSAutoDiff= -I ../autodiff
CFLAGSCubature= -I ../cubature
DEBUG = -g

TARGET=mlf-distr-v1.out
TARGETFULL=mlf-distr-v1-full.out
TARGETV4=mlf-distr-v4.out
TARGETV0=mlf-distr-v0.out
TARGETSHARED=mlf-shared.out
TARGETSHAREDCBLAS=mlf-shared-cblas.out
TARGETFRACSHARED=frac.out
TARGETTESTEigen=testEigen.out

LINK= -m64  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

GREEN=\033[0;32m
RED=\033[0;31m
YELLOW=\033[0;33m
NORMAL=\033[0m

OBJDIR = obj
OBJDIR_DISTR = obj_distr

# Source files for each build
SRCS_SHARED = ../utils/arnoldi_iteration_shared.cpp ../utils/csr_matrix.cpp ../utils/dense_matrix.cpp \
              ../utils/dense_vector.cpp ../utils/io_ops.cpp ../utils/schur_blocking.cpp ../utils/mtx_ops_mkl.cpp \
              ../utils/calculate_MLF.cpp ../utils/mittag_leffler_scalar.cpp mlf_shared.cpp

SRCS_SHARED_CBLAS = ../utils/arnoldi_iteration_shared_cblas.cpp ../utils/csr_matrix.cpp ../utils/dense_matrix.cpp \
                    ../utils/dense_vector.cpp ../utils/io_ops.cpp ../utils/schur_blocking.cpp ../utils/mtx_ops_mkl.cpp \
                    ../utils/calculate_MLF.cpp ../utils/mittag_leffler_scalar.cpp mlf_shared.cpp

SRCS_DISTR = ../utils/arnoldi_iteration-v1.cpp ../utils/csr_matrix.cpp ../utils/dense_matrix.cpp ../utils/dense_vector.cpp \
             ../utils/distr_mtx_ops.cpp ../utils/help_process.cpp ../utils/io_ops.cpp ../utils/schur_blocking.cpp \
             ../utils/mtx_ops_mkl.cpp ../utils/calculate_MLF.cpp ../utils/mittag_leffler_scalar.cpp mlf_distr.cpp

SRCS_DISTRV4 = ../utils/arnoldi_iteration-v4-paper.cpp ../utils/csr_matrix.cpp ../utils/dense_matrix.cpp ../utils/dense_vector.cpp \
             ../utils/distr_mtx_ops.cpp ../utils/help_process.cpp ../utils/io_ops.cpp ../utils/schur_blocking.cpp \
             ../utils/mtx_ops_mkl.cpp ../utils/calculate_MLF.cpp ../utils/mittag_leffler_scalar.cpp mlf_distr.cpp

SRCS_DISTRV0 = ../utils/arnoldi_iteration-v0.cpp ../utils/csr_matrix.cpp ../utils/dense_matrix.cpp ../utils/dense_vector.cpp \
             ../utils/distr_mtx_ops.cpp ../utils/help_process.cpp ../utils/io_ops.cpp ../utils/schur_blocking.cpp \
             ../utils/mtx_ops_mkl.cpp ../utils/calculate_MLF.cpp ../utils/mittag_leffler_scalar.cpp mlf_distr.cpp

SRCS_DISTR_FULL = ../utils/arnoldi_iteration-v1-full.cpp ../utils/csr_matrix.cpp ../utils/dense_matrix.cpp ../utils/dense_vector.cpp \
             ../utils/distr_mtx_ops.cpp ../utils/help_process.cpp ../utils/io_ops.cpp ../utils/schur_blocking.cpp \
             ../utils/mtx_ops_mkl.cpp ../utils/calculate_MLF.cpp ../utils/mittag_leffler_scalar.cpp mlf_distr-fullmtx.cpp

SRCS_FRAC_SHARED = ../utils/arnoldi_iteration_shared_nu.cpp ../utils/csr_matrix.cpp ../utils/dense_matrix.cpp \
                   ../utils/dense_vector.cpp ../utils/io_ops.cpp ../utils/schur_blocking.cpp ../utils/mtx_ops_mkl.cpp \
                   ../utils/calculate_MLF.cpp ../utils/mittag_leffler_scalar.cpp ../utils/scaling_and_squaring.cpp \
                   ../utils/fractional_Krylov.cpp frac-krylov.cpp

SRCS_TEST_EIGEN = ../utils/csr_matrix.cpp ../utils/dense_matrix.cpp ../utils/dense_vector.cpp ../utils/io_ops.cpp \
                  ../utils/schur_blocking.cpp ../utils/mtx_ops_mkl.cpp ../utils/mittag_leffler_scalar.cpp \
                  ../utils/calculate_MLF_Eigen.cpp mlf_shared.cpp

# Convert source files to object files in the object directory
OBJS_SHARED = $(patsubst %.cpp, $(OBJDIR)/%.o, $(notdir $(SRCS_SHARED)))
OBJS_SHARED_CBLAS = $(patsubst %.cpp, $(OBJDIR)/%.o, $(notdir $(SRCS_SHARED_CBLAS)))
OBJS_DISTR = $(patsubst %.cpp, $(OBJDIR_DISTR)/%.o, $(notdir $(SRCS_DISTR)))
OBJS_DISTR_V0 = $(patsubst %.cpp, $(OBJDIR_DISTR)/%.o, $(notdir $(SRCS_DISTRV0)))
OBJS_DISTR_V4 = $(patsubst %.cpp, $(OBJDIR_DISTR)/%.o, $(notdir $(SRCS_DISTRV4)))
OBJS_DISTR_FULL = $(patsubst %.cpp, $(OBJDIR_DISTR)/%.o, $(notdir $(SRCS_DISTR_FULL)))
OBJS_FRAC_SHARED = $(patsubst %.cpp, $(OBJDIR)/%.o, $(notdir $(SRCS_FRAC_SHARED)))
OBJS_TEST_EIGEN = $(patsubst %.cpp, $(OBJDIR)/%.o, $(notdir $(SRCS_TEST_EIGEN)))

all: mlf-shared mlf-distr

distr: mlf-distr mlf-distr-v4

shared: mlf-shared mlf-shared-cblas

frac-test: mlf-shared-test frac

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

$(OBJDIR)/fractional_Krylov.o: ../utils/fractional_Krylov.cpp | $(OBJDIR)
	$(C++) $(CFLAGS) $(CFLAGSCubature) $(CFLAGSAutoDiff) $(CFLAGSEigen) -c $< -o $@

$(OBJDIR)/mlf_shared.o: mlf_shared.cpp | $(OBJDIR)
	$(C++) $(CFLAGS) -c $< -o $@

$(OBJDIR_DISTR)/mlf_distr.o: mlf_distr.cpp | $(OBJDIR_DISTR)
	$(mpi) $(CFLAGSDISTR) -c $< -o $@

$(OBJDIR_DISTR)/mlf_distr-fullmtx.o: mlf_distr-fullmtx.cpp | $(OBJDIR_DISTR)
	$(mpi) $(CFLAGSDISTR) -c $< -o $@

$(OBJDIR)/frac-krylov.o: frac-krylov.cpp | $(OBJDIR)
	$(C++) $(CFLAGS) $(CFLAGSCubature) $(CFLAGSAutoDiff) $(CFLAGSEigen) -c $< -o $@

# Targets

mlf-shared: $(OBJS_SHARED)
	$(C++) $(CFLAGS) $(LINK) $^ -o $(TARGETSHARED)
	@echo "$(GREEN)Compiled Mittag-Leffler computation shared version: $(TARGETSHARED) $(NORMAL)"


mlf-shared-cblas: $(OBJS_SHARED_CBLAS)
	$(C++) $(CFLAGS) $(LINK) $^ -o $(TARGETSHAREDCBLAS)
	@echo "$(GREEN)Compiled Mittag-Leffler computation shared CBLAS version: $(TARGETSHAREDCBLAS) $(NORMAL)"


mlf-distr: $(OBJS_DISTR)
	$(mpi) $(CFLAGSDISTR) $(LINK) $^ -o $(TARGET)
	@echo "$(GREEN)Compiled Mittag-Leffler computation distributed version: $(TARGET) $(NORMAL)"

mlf-distr-v4: $(OBJS_DISTR_V4)
	$(mpi) $(CFLAGSDISTR) $(LINK) $^ -o $(TARGETV4)
	@echo "$(GREEN)Compiled Mittag-Leffler computation distributed version: $(TARGETV4) $(NORMAL)"

mlf-distr-v0: $(OBJS_DISTR_V0)
	$(mpi) $(CFLAGSDISTR) $(LINK) $^ -o $(TARGETV0)
	@echo "$(GREEN)Compiled Mittag-Leffler computation distributed version: $(TARGETV0) $(NORMAL)"

mlf-distr-full: $(OBJS_DISTR_FULL)
	$(mpi) $(CFLAGSDISTR) $(LINK) $^ -o $(TARGETFULL)
	@echo "$(GREEN)Compiled Mittag-Leffler computation distributed version: $(TARGETFULL) $(NORMAL)"

mlf-shared-test: $(OBJS_SHARED)
	$(C++) $(CFLAGS) $(LINK) $^ -o $(TARGETSHARED)
	@echo "$(GREEN)Compiled Mittag-Leffler computation frac shared test version: $(TARGETSHARED) $(NORMAL)"


frac: $(OBJS_FRAC_SHARED)
	$(C++) $(CFLAGSCubature) $(CFLAGS) $(LINK) $^ ../cubature/hcubature.o -o $(TARGETFRACSHARED)
	@echo "$(GREEN)Compiled fractional Krylov shared version: $(TARGETFRACSHARED) $(NORMAL)"


testEigen: $(OBJS_TEST_EIGEN)
	$(C++) $(CFLAGS) $(LINK) $^ -o $(TARGETTESTEigen)
	@echo "$(YELLOW)Compiled test eigen version: $(TARGETTESTEigen) $(NORMAL)"


clean:
	@echo "$(RED)Warning: deleting executables and binary files!$(NORMAL)"
	rm -rf $(OBJDIR) $(OBJDIR_DISTR) $(TARGET) $(TARGETSHARED) $(TARGETTESTEigen) $(TARGETSHAREDCBLAS) $(TARGETFRACSHARED)

clean-target:
	@echo "$(RED)Warning: deleting executables!$(NORMAL)"
	rm -rf $(TARGET) $(TARGETSHARED) $(TARGETTESTEigen) $(TARGETSHAREDCBLAS) $(TARGETFRACSHARED)

#even if there is a file name clean, the Makefile will still execute that command instead of saying its up to date
.PHONY: clean
