SRCS += $(wildcard src/*.cpp)

OBJS :=	build/main.o \
				build/LLC_Keccak-compact64.o \
				build/LLC_KeccakDuplex.o \
				build/LLC_KeccakHash.o \
				build/LLC_KeccakSponge.o \
				build/LLC_skein.o \
				build/LLC_skein_block.o \
				build/LLC_bignum.o \
				build/LLC_base_uint.o \
				build/LLC_global.o \
				build/LLC_prime.o \
				build/LLC_origins.o \
				build/LLC_prime2.o \
				build/LLC_cuda_prime.o \
				build/LLC_cpu_primetest.o \
				build/LLC_cpu_primesieve.o \
				build/LLC_cuda_hash.o \
				build/LLC_cpu_hash.o \
				build/LLP_base_address.o \
				build/LLP_base_connection.o \
				build/LLP_connection.o \
				build/LLP_ddos.o \
				build/LLP_hosts.o \
				build/LLP_network.o \
				build/LLP_socket.o \
				build/LLP_outbound.o \
				build/LLP_miner.o \
				build/LLP_worker.o \
				build/TAO_Ledger_block.o \
				build/TAO_Ledger_difficulty.o \
  				build/Util_debug.o \
				build/Util_args.o \
				build/Util_config.o \
				build/Util_memory.o \
				build/Util_version.o \
				build/Util_filesystem.o \
				build/Util_signals.o \
				build/Util_ini_parser.o \
				build/Util_prime_config.o

CUDA_SRCS := $(wildcard src/CUDA/*.cu)
CUDA_OBJS := build/CUDA_prime_sieve.o \
			 build/CUDA_prime_combo_sieve.o \
	         build/CUDA_prime_test.o \
			 build/CUDA_hash_sk1024.o \
	         build/CUDA_util.o \
			 build/CUDA_streams_events.o \
			 build/CUDA_constants.o

CUDA_LINK_OBJ := build/CUDA_cuLink.o

CUDA_PATH := /usr/local/cuda-10.1
NVCC := $(CUDA_PATH)/bin/nvcc
CXX := g++
CC := gcc


LIB := -lpthread -lgmp -lcrypto -lgomp

CUDA_LIB := -L$(CUDA_PATH)/lib64 -lcudadevrt -lcudart -lcuda
CUDA_INC += -I$(CUDA_PATH)/include
CUDA_INC += $(addprefix -I, $(CURDIR) $(CURDIR)/build $(CURDIR)/src )

CFLAGS   :=
CXXFLAGS := -std=c++11 -msse2 -O2 -fopenmp
INCLUDES += $(addprefix -I, $(CURDIR) $(CURDIR)/build $(CURDIR)/src )

GPU_CARD := -gencode arch=compute_60,code=sm_60 \
            -gencode arch=compute_70,code=sm_70

NVCC_FLAGS += -std=c++11 -rdc=true -O2 -D_FORCE_INLINES -Xptxas "-v" --ptxas-options=-v
CUDA_LINK_FLAGS := -dlink

ifdef ENABLE_DEBUG
	NVCC_FLAGS+=-g -G
	CXXFLAGS+= -g
endif

ifdef RACE_CHECK
	NVCC_FLAGS+= -lineinfo -Xcompiler -rdynamic
endif

EXEC          := nexusminer

all:	$(EXEC)
$(EXEC):	$(CUDA_OBJS)	$(OBJS)
					$(NVCC) $(GPU_CARD) $(CUDA_LINK_FLAGS) -o $(CUDA_LINK_OBJ) $(CUDA_OBJS)
					$(CXX) -o $@ $(OBJS) $(CUDA_OBJS) $(CUDA_LINK_OBJ) $(CUDA_LIB) $(LIB)

build/%.o:	src/%.cpp
	$(CXX) -o $@ -c $(CXXFLAGS) $< $(INCLUDES)

build/LLC_%.o:	src/LLC/%.cpp
	$(CXX) -o $@ -c $(CXXFLAGS) $< $(INCLUDES)

build/LLC_%.o:	src/LLC/hash/SK/%.cpp
	$(CXX) -o $@ -c $(CXXFLAGS) $< $(INCLUDES)

build/LLP_%.o:	src/LLP/%.cpp
	$(CXX) -o $@ -c $(CXXFLAGS) $< $(INCLUDES)

build/TAO_Ledger_%.o:	src/TAO/Ledger/%.cpp
	$(CXX) -o $@ -c $(CXXFLAGS) $< $(INCLUDES)

build/Util_%.o:	src/Util/%.cpp
	$(CXX) -o $@ -c $(CXXFLAGS) $< $(INCLUDES)

build/CUDA_prime_sieve.o:	src/CUDA/prime/sieve.cu
	$(NVCC) $(GPU_CARD) $(NVCC_FLAGS) --maxrregcount=72 -o $@ -c $< $(CUDA_INC)

build/CUDA_prime_combo_sieve.o:	src/CUDA/prime/combo_sieve.cu
	$(NVCC) $(GPU_CARD) $(NVCC_FLAGS) --maxrregcount=72 -o $@ -c $< $(CUDA_INC)

build/CUDA_prime_test.o:	src/CUDA/prime/test.cu
	$(NVCC) $(GPU_CARD) $(NVCC_FLAGS) --maxrregcount=128 -o $@ -c $< $(CUDA_INC)

build/CUDA_hash_sk1024.o:	src/CUDA/hash/sk1024.cu
	$(NVCC) $(GPU_CARD) $(NVCC_FLAGS) --maxrregcount=72 -o $@ -c $< $(CUDA_INC)

build/CUDA_%.o:	src/CUDA/%.cu
	$(NVCC) $(GPU_CARD) $(NVCC_FLAGS) -o $@ -c $< $(CUDA_INC)

clean:
	rm -f $(EXEC)
	rm -f build/*.o
