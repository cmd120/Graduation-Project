#include "include/MNIST_Read.h"
#include "include/covtype.h"
#include "include/IAG.h"
#include "include/IAGA.h"
#include "include/SAG.h"
#include "include/SAGA.h"
#include "include/SGD.h"
#include "include/SIG.h"
#include "include/SVRG.h"


int epochCounter;
FILE *fp;
std::chrono::high_resolution_clock::time_point startTime;
int SPARSE;

int LogisticEntrance(int algorithmType){
	MatrixXd Xt,XtTest;
	VectorXd y,yTest;
    VectorXd w,sumIG(Xt.rows()),gradients(Xt.cols()), wtilde,G;
    double lambda , eta, a, b, gamma;
	int maxIter, batchSize, passes, maxRunTime;
	SPARSE = 0;

	if(!algorithmType){
		cout << "Bye." << endl;
		return 1;
	}
	cout << "Your choice of algorithm: " << algorithmType << endl;
	cout << "Available datasets: " << endl << "1. MNIST " << "2. COVTYPE"<< endl;
	int datasetNum;
	cin >> datasetNum;
	cout << "Your choice of dataset: " << datasetNum << endl;;
	//default dataset is MNIST
	
	//MNIST dataset
	switch(datasetNum){
		case 1:
			mnist_read(Xt, y, XtTest, yTest);break;
		case 2:
			covtype_read(Xt, y, XtTest, yTest);break;
	}
    w = MatrixXd::Zero(Xt.rows(),1);
    wtilde = w;
    G = w;
    gradients = (1+(-Xt.adjoint()*w).array().exp()).inverse() - y.array();
    sumIG = Xt*gradients;
    string filename;
    int nVars, nSamples, flag;
    epochCounter = 0;
    nVars = Xt.rows();
    nSamples = Xt.cols();
    switch(algorithmType){
		if(!SPARSE){
	    	case 1:
	    		IAG_init(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter, batchSize, passes, maxRunTime, filename);
	    		for(int pass=0;pass<passes;++pass){
		    		if(batchSize>=2?IAG_LogisticInnerLoopBatchDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize):\
		    					IAG_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime)
	    			)break;
	    		}
	    		break;
	    	case 2:
	    		IAGA_init(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter, batchSize, passes, maxRunTime, filename);
	    		for(int pass=0;pass<passes;++pass){
		    		if(batchSize>=2?IAGA_LogisticInnerLoopBatchDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize):\
		    					IAGA_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime)
	    			)break;
	    		}
	    		break;
	    	case 3:
	    		SAG_init(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter, batchSize, passes, maxRunTime, filename);
	    		for(int pass=0;pass<passes;++pass){
		    		if(batchSize>=2?SAG_LogisticInnerLoopBatchDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize):\
		    					SAG_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime)
	    			)break;
	    		}
	    		break;
	    	case 4:
	    		SAGA_init(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter, batchSize, passes, maxRunTime, filename);
	    		for(int pass=0;pass<passes;++pass){
		    		if(batchSize>=2?SAGA_LogisticInnerLoopBatchDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize):\
		    					SAGA_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime)
	    			)break;
	    		}
	    		break;
	    	case 5:
		    	SGD_init(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter, batchSize, passes, maxRunTime, filename);
	    		for(int pass=0;pass<passes;++pass){
		    		if(batchSize>=2?SGD_LogisticInnerLoopBatchDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize):\
		    					SGD_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime)
	    			)break;
	    		}
	    		break;
	    	case 6:
	    		SIG_init(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter, batchSize, passes, maxRunTime, filename);
	    		for(int pass=0;pass<passes;++pass){
	    			LogisticGradient(wtilde, G, Xt, y);
		    		if(batchSize>=2?SIG_LogisticInnerLoopBatchDense(w, Xt, y, XtTest, yTest, wtilde, G, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize):\
		    					SIG_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, wtilde, G, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime)
	    			)break;
	    		}
	    		break;
	    	case 7:
	    		SVRG_init(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter, batchSize, passes, maxRunTime, filename);
	    		for(int pass=0;pass<passes;++pass){
	    			LogisticGradient(wtilde, G, Xt, y);
		    		if(batchSize>=2?SVRG_LogisticInnerLoopBatchDense(w, Xt, y, XtTest, yTest, wtilde, G, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize):\
		    					SVRG_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, wtilde, G, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime)
					)break;
				}
				break;
			default:
				;
		}
		else{
			;
		}
    }
	return 0;
}

int main(int argc, char* argv[]){
	int algorithmType, dataset=1;
	while(1){
		cout << "Available algorithms:" << endl << "1. IAG  " << "2. IAGA" << endl << "3. SAG  " << "4. SAGA" << endl << "5. SGD  " << "6. SIG" << endl << "7. SVRG " << "0. Quit" << endl;
		cout << "Enter your choice of algorithm: " << endl;
		if(cin >> algorithmType){
			if(LogisticEntrance(algorithmType))break;
		}
		else{
			cout << "Invalid Input! Please intput a numerical value." << endl;
			cin.clear();
			while(cin.get()!='\n');
		}
	}
	return 0;
}