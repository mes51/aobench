CC      = /usr/bin/cc
CCFLAGS = -c -Wall -msse4.1 -DUSE_SSE2 
LDFLAGS =

SRCS    = ao.c ao_simd_soa.c ao_simd_aos.c ao_simd_soa_inlined.c
OBJS    = $(SRCS:.c=.o)
EXES    = $(SRCS:.c=)

INCS    = 
LIBS    = 

all: $(EXES)

%.o: %.c
	$(CC) $(CCFLAGS) $(INCS) $< -o $@

$(EXES): %: %.o
	$(CC) $(LDFLAGS) $< -o $@

clean:
	rm -rf $(OBJS) $(EXES) ao.ppm
