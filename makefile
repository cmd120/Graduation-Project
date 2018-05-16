objects = entrance.o MNIST_Read.o covtype_read.o DenseMat.o \
			SparseMat.o LogisticError.o LogisticGradient.o IAG_logistic.o \
			IAGA_logistic.o SAG_logistic.o SAGA_logistic.o SGD_logistic.o \
			SIG_logistic.o SVRG_logistic.o covtype_binary_read.o
headers = include/comm.h include/MNIST_Read.h include/covtype.h include/IAG.h \
		 	include/IAGA.h include/SAG.h include/SAGA.h include/SGD.h \
		 	include/SIG.h include/SVRG.h include/DenseMat.h include/SparseMat.h

run: $(objects)
	g++ -O3 -std=c++11 -o run $(objects)
	#g++ -O3 -std=c++11 -o run $(objects)  ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl	
entrance.o:entrance.cpp $(headers)
	g++ -O3 -std=c++11 -c entrance.cpp
DenseMat.o:DenseMat.cpp include/DenseMat.h
	g++ -O3 -std=c++11 -c DenseMat.cpp
SparseMat.o:SparseMat.cpp include/SparseMat.h
	g++ -O3 -std=c++11 -c SparseMat.cpp
covtype_read.o:covtype_read.cpp include/covtype.h
	g++ -O3 -std=c++11 -c covtype_read.cpp
covtype_binary_read.o: covtype_binary_read.cpp include/covtype.h
	g++ -O3 -std=c++11 -c covtype_binary_read.cpp
MNIST_Read.o:MNIST_Read.cpp include/MNIST_Read.h
	g++ -O3 -std=c++11 -c MNIST_Read.cpp
LogisticError.o:LogisticError.cpp include/LogisticError.h
	g++ -O3 -std=c++11 -c LogisticError.cpp
LogisticGradient.o:LogisticGradient.cpp include/LogisticGradient.h
	g++ -O3 -std=c++11 -c LogisticGradient.cpp
IAG_logistic.o:IAG_logistic.cpp include/IAG.h
	g++ -O3 -std=c++11 -c IAG_logistic.cpp
IAGA_logistic.o:IAGA_logistic.cpp include/IAGA.h
	g++ -O3 -std=c++11 -c IAGA_logistic.cpp
SAG_logistic.o:SAG_logistic.cpp include/SAG.h
	g++ -O3 -std=c++11 -c SAG_logistic.cpp
SAGA_logistic.o:SAGA_logistic.cpp include/SAGA.h
	g++ -O3 -std=c++11 -c SAGA_logistic.cpp
SGD_logistic.o:SGD_logistic.cpp include/SGD.h
	g++ -O3 -std=c++11 -c SGD_logistic.cpp
SIG_logistic.o:SIG_logistic.cpp include/SIG.h
	g++ -O3 -std=c++11 -c SIG_logistic.cpp
SVRG_logistic.o:SVRG_logistic.cpp include/SVRG.h
	g++ -O3 -std=c++11 -c SVRG_logistic.cpp
.PHONY : clean
clean:
	rm run $(objects)