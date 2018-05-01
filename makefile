run: entrance.o MNIST_Read.o DenseMat.o LogisticError.o IAG_logistic.o IAGA_logistic.o SAG_logistic.o SAGA_logistic.o SGD_logistic.o SIG_logistic.o SVRG_logistic.o
	g++ -o run entrance.o MNIST_Read.o DenseMat.o LogisticError.o IAG_logistic.o IAGA_logistic.o SAG_logistic.o SAGA_logistic.o SGD_logistic.o SIG_logistic.o SVRG_logistic.o
entrance.o:entrance.cpp include/comm.h include/MNIST_Read.h include/IAG.h include/IAGA.h include/SAG.h include/SAGA.h include/SGD.h include/SIG.h include/SVRG.h
	g++ -c entrance.cpp
DenseMat.o:DenseMat.cpp include/DenseMat.h
	g++ -c DenseMat.cpp
MNIST_Read.o:MNIST_Read.cpp include/MNIST_Read.h
	g++ -c MNIST_Read.cpp
LogisticError.o:LogisticError.cpp include/LogisticError.h
	g++ -c LogisticError.cpp
IAG_logistic.o:IAG_logistic.cpp include/IAG.h
	g++ -c IAG_logistic.cpp
IAGA_logistic.o:IAGA_logistic.cpp include/IAGA.h
	g++ -c IAGA_logistic.cpp
SAG_logistic.o:SAG_logistic.cpp include/SAG.h
	g++ -c SAG_logistic.cpp
SAGA_logistic.o:SAGA_logistic.cpp include/SAGA.h
	g++ -c SAGA_logistic.cpp
SGD_logistic.o:SGD_logistic.cpp include/SGD.h
	g++ -c SGD_logistic.cpp
SIG_logistic.o:SIG_logistic.cpp include/SIG.h
	g++ -c SIG_logistic.cpp
SVRG_logistic.o:SVRG_logistic.cpp include/SVRG.h
	g++ -c SVRG_logistic.cpp
clean:
	rm *.o output