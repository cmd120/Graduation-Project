#include "include/MNIST_Read.h"
#include "include/covtype.h"
#include "include/IAG.h"
#include "include/IAGA.h"
#include "include/SAG.h"
#include "include/SAGA.h"
#include "include/SGD.h"
#include "include/SIG.h"
#include "include/SVRG.h"

const int trainSetSize = 2000;
const int testSetSize = 100;
int epochCounter;
FILE *fp;
std::chrono::high_resolution_clock::time_point startTime;
int SPARSE;
int LogisticEntrance(int algorithmType, int datasetNum, SparseMatrix<double> &XtS, VectorXd & y, SparseMatrix<double> &XtTestS, VectorXd & yTest){
    VectorXd w, wtilde, G, sumIG, gradients;
    double lambda , eta, a, b, gamma;
	int maxIter, batchSize, passes, maxRunTime;
	SPARSE = 0;
	if(!algorithmType){
		cout << "Bye." << endl;
		return 1;
	}
	cout << "Your choice of algorithm: " << algorithmType << endl;
	int nVars, nSamples, flag;
	string filename;
	int *innerIndices,*outerStarts;
	innerIndices = XtS.innerIndexPtr();
	outerStarts = new int[XtS.cols()];
	if(!outerStarts){
		cout << "run out of space!" << endl;
	}
	InitOuterStarts(XtS,outerStarts);
	cout << "1 pass" << endl;
	    w = MatrixXd::Zero(XtS.rows(),1);
	    cout << "2 pass" << endl;
    wtilde = w;
    cout << "wtilde size: " << wtilde.size() << endl;
    G = w;
    cout << "G size: "<< G.size() << endl;
    gradients = (1+(-XtS.adjoint()*w).array().exp()).inverse() - y.array();
    cout << "3 pass" << endl;
    sumIG = XtS*gradients;
    epochCounter = 0;
    cout << "4 pass" << endl;
    nVars = XtS.rows();
    nSamples = XtS.cols();
    cout << "init pass" << endl;
    switch(algorithmType){
    	case 1:
			IAG_init(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter, batchSize, passes, maxRunTime, filename, datasetNum);
			for(int pass;pass<passes;++pass){
				if(batchSize>=2?IAG_LogisticInnerLoopBatch(w,XtS,y,innerIndices,outerStarts,XtTestS,yTest,lambda,sumIG,gradients,maxIter,nSamples,nVars,pass,a,b,gamma,maxRunTime,batchSize):\
							IAG_LogisticInnerLoopSingle(w,XtS,y,innerIndices,outerStarts,XtTestS,yTest,lambda,sumIG,gradients,maxIter,nSamples,nVars,pass,a,b,gamma,maxRunTime)
				)break;
			}
    		break;
		default:
			;
    }
    return 0;
}
int LogisticEntrance(int algorithmType, int datasetNum, MatrixXd &Xt, VectorXd & y, MatrixXd &XtTest, VectorXd & yTest){
    VectorXd w, wtilde, G, sumIG, gradients;
    double lambda , eta, a, b, gamma;
	int maxIter, batchSize, passes, maxRunTime;
	SPARSE = 0;
	if(!algorithmType){
		cout << "Bye." << endl;
		return 1;
	}
	cout << "Your choice of algorithm: " << algorithmType << endl;
	int nVars, nSamples, flag;
	string filename;
    w = MatrixXd::Zero(Xt.rows(),1);
    wtilde = w;
    G = w;
    gradients = (1+(-Xt.adjoint()*w).array().exp()).inverse() - y.array();
    sumIG = Xt*gradients;
    epochCounter = 0;
    nVars = Xt.rows();
    nSamples = Xt.cols();
    switch(algorithmType){
    	case 1:
    		IAG_init(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter, batchSize, passes, maxRunTime, filename, datasetNum);
    		for(int pass=0;pass<passes;++pass){
	    		if(batchSize>=2?IAG_LogisticInnerLoopBatch(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime, datasetNum):\
	    					IAG_LogisticInnerLoopSingle(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime)
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
	return 0;
}

int main(int argc, char* argv[]){
	MatrixXd Xt,XtTest;
	SparseMatrix<double> XtS(54,trainSetSize), XtTestS(54,testSetSize);
	VectorXd y,yTest;
	int algorithmType, dataset;
	cout << "Available datasets: " << endl << "1. MNIST " << "2. COVTYPE"<< endl;
	int datasetNum;
	cin >> datasetNum;
	cout << "Your choice of dataset: " << datasetNum << endl;
	switch(datasetNum){
		case 1:
			mnist_read(Xt, y, XtTest, yTest);
			break;
		case 2:
			// covtype_read(Xt, y, XtTest, yTest);break;
			covtype_binary_read(XtS,y,trainSetSize);
			covtype_binary_read(XtTestS,yTest,testSetSize);
			break;
		case 3:
			covtype_read(Xt, y, XtTest, yTest);
	}
	cout << "dataset loaded." << endl;
	if(SPARSE){
		cout << "dataset is sparse" << endl;
	}
	else{
		cout << "dataset is dense" <<endl;
	}
	while(1){
		cout << "Available algorithms:" << endl << "1. IAG  " << "2. IAGA" << endl << "3. SAG  " << "4. SAGA" << endl << "5. SGD  " << "6. SIG" << endl << "7. SVRG " << "0. Quit" << endl;
		cout << "Enter your choice of algorithm: " << endl;
		if(cin >> algorithmType){
			int ret;
			ret = SPARSE?LogisticEntrance(algorithmType,datasetNum,XtS,y,XtTestS,yTest):LogisticEntrance(algorithmType,datasetNum,Xt,y,XtTest,yTest);
			if(ret)break;
		}
		else{
			cout << "Invalid Input! Please intput a numerical value." << endl;
			cin.clear();
			while(cin.get()!='\n');
		}
	}
	return 0;
}