#include "include/MNIST_Read.h"
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
	int i;
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
	cout << "Available datasets: " << endl << "1. MNIST" << endl;
	//default dataset is MNIST
	int dataset = 1;
	//MNIST dataset
	if(dataset==1){
	    string train_image_path = "train-images";
	    string train_label_path = "train-labels";
	    string test_image_path = "test-images";
	    string test_label_path = "test-labels";
	    vector<BYTE> train_image_dataset = read_mnist_images(train_image_path);
	    vector<BYTE> train_label_dataset = read_mnist_labels(train_label_path);
	 	vector<BYTE> test_image_dataset = read_mnist_images(test_image_path);
	    vector<BYTE> test_label_dataset = read_mnist_labels(test_label_path);
	    
	    vector<double> Xt_train(train_image_dataset.begin(),train_image_dataset.end());
	    vector<double> y_train(train_label_dataset.begin(),train_label_dataset.end());
	    vector<double> Xt_test(test_image_dataset.begin(),test_image_dataset.end());
	    vector<double> y_test(test_label_dataset.begin(),test_label_dataset.end());  
	    //classification
	    vector<double> Xt_train_classify,y_train_classify,Xt_test_classify,y_test_classify;
	    for(i=0;i<y_train.size();++i){
	    	if(y_train[i]<=1){
	    		Xt_train_classify.insert(Xt_train_classify.end(),Xt_train.begin()+i*784,Xt_train.begin()+(i+1)*784);
	    		y_train_classify.push_back(y_train[i]);
	    	}
	    }
	    for(i=0;i<y_test.size();++i){
	    	if(y_test[i]<=1){
	    		Xt_test_classify.insert(Xt_test_classify.end(),Xt_test.begin()+i*784,Xt_test.begin()+(i+1)*784);
	    		y_test_classify.push_back(y_test[i]);
	    	}
	    }
	    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> Xtt(Xt_train_classify.data(), 784, Xt_train_classify.size()/784);
	    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> yy(y_train_classify.data(), y_train_classify.size(), 1);
	    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> XttTest(Xt_test_classify.data(), 784, Xt_test_classify.size()/784);
	    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> yyTest(y_test_classify.data(), y_test_classify.size(), 1);
		//normalization
	    for(i=0;i<Xtt.cols();++i){
	    	Xtt.col(i) = Xtt.col(i)/Xtt.col(i).norm();
	    }
	    for(i=0;i<XttTest.cols();++i){
	    	XttTest.col(i) = XttTest.col(i)/XttTest.col(i).norm();
	    }
	    // VectorXd yy = y;
	    // VectorXd yyTest = yTest;
	    w = MatrixXd::Zero(Xtt.rows(),1);
	    wtilde = w;
	    G = w;
	    gradients = (1+(-Xtt.adjoint()*w).array().exp()).inverse() - yy.array();
	    sumIG = Xtt*gradients;
	    //DEBUG
	    cout << "Xt rows: " << Xtt.rows() << endl << "Xt cols: " << Xtt.cols() << endl;
	    cout << "XtTest rows: " << XttTest.rows() << endl << "XtTest cols: " << XttTest.cols() << endl;
	    cout << "y cols: " << yy.size() << endl;
	    cout << "yTest cols: " << yyTest.size() << endl;
	    Xt = Xtt; y = yy; XtTest = XttTest; yTest = yyTest;
	}
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