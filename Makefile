APPS=Activations

all: ${APPS}

%: %.cu
	# nvcc -O2 -lcublas -DCUBLASXt -DCUBLAS -DSIMPLECPU -DSIMPLECUDA -o $@ $<
	nvcc -O2 -o $@ $<

clean:
	rm -f ${APPS} *.dat

run:	
	./multmat_solucion

submit:	clean all
	qsub jobGPU; watch qstat



#define SIMPLECPU
#define SIMPLECUDA
#define CUBLAS
#define CUBLASXt
#file="multmat_solucion.cu"
#nvcc $file -lcublas -DCUBLASXt -o cublasxt
#nvcc $file -lcublas -DCUBLASN -o cublas
#nvcc $file -lcublas -DCPU -o naivecpu
#nvcc $file -lcublas -DSIMPLECUDA -o naivecuda

