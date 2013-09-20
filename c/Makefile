CC      = /usr/bin/cc
CCFLAGS = -c -Wall -msse4.1 -DUSE_SSE2 -g 
LDFLAGS = -lm

SRCS    = ao.c ao_simd_soa.c ao_simd_aos.c ao_simd_soa_inlined.c
OBJS    = $(SRCS:.c=.o)
EXES    = $(SRCS:.c=)
OBJSd   = $(SRCS:.c=_dbg.o)
EXESd   = $(SRCS:.c=_dbg)

INCS    = 
LIBS    = 

all: $(EXES) $(EXESd)

$(OBJS): %.o: %.c
	$(CC) $(CCFLAGS) $(INCS) $< -o $@

$(EXES): %: %.o
	$(CC) $< $(LDFLAGS) -o $@

$(OBJSd): %_dbg.o: %.c
	$(CC) $(CCFLAGS) $(INCS) -DDBG $< -o $@

$(EXESd): %_dbg: %_dbg.o
	$(CC) $< $(LDFLAGS) -o $@

run:
	./ao; ./ao_simd_aos; ./ao_simd_soa; ./ao_simd_soa_inlined; \
	./ao_dbg 2> seq.txt; ./ao_simd_soa_dbg 2> soa.txt; diff seq.txt soa.txt \

clean:
	rm -rf $(OBJS) $(EXES) $(OBJSd) $(EXESd) ao.ppm seq.txt soa.txt
