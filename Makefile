NVCC := nvcc
NVFLAGs := -arch=sm_86


# source files
SRC := simple_kernel.cu

# executable
EXE := sk

# compile
$(EXE):
	$(NVCC) $(NVFLAGs) $(SRC) -o $(EXE)


# run
run:
	./$(EXE)

clean:
	rm -f $(EXE) *.o