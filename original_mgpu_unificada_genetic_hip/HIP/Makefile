include ../config.mk
include flags.mk

all:
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c hip_driver.hip
	$(HIPCC) $(HIPCFLAGS) $(COMMON_FLAGS) -c hip_stuff.hip
	$(HIPCC) $(HIPCFLAGS) $(COMMON_FLAGS) -c hip_propagate.hip
	$(HIPCC) $(HIPCFLAGS) $(COMMON_FLAGS) -c hip_insertsource.hip
#	$(HIPCC) $(HIPCFLAGS) $(COMMON_FLAGS) -c hip_allocate.cpp

clean:
	rm -f *.o *.a
