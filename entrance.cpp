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

void LogisticEntrance(int algorithmType, int dataset){
	int i;
	MatrixXd Xt,XtTest;
	VectorXd y,yTest;
    VectorXd w,sumIG(Xt.rows()),gradients(Xt.cols());
    double lambda = 0.1, eta = 0.1, a = 1.0, b = 1.0, gamma = 1.0;
	int maxIter = 60, batchSize = 1, pass = 20, maxRunTime=20;
	SPARSE = 0;
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
	    for(i=0;i<train_label_dataset.size();++i){
	    	if(train_label_dataset[i]<=1){
	    		Xt_train_classify.insert(Xt_train_classify.end(),Xt_train.begin()+i*784,Xt_train.begin()+(i+1)*784);
	    		y_train_classify.push_back(y_train[i]);
	    	}
	    }
	    for(i=0;i<test_label_dataset.size();++i){
	    	if(train_label_dataset[i]<=1){
	    		Xt_test_classify.insert(Xt_test_classify.end(),Xt_test.begin()+i*784,Xt_test.begin()+(i+1)*784);
	    		y_test_classify.push_back(y_test[i]);
	    	}
	    }
	    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> Xt(Xt_train_classify.data(), 784, Xt_train_classify.size()/784);
	    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> y(y_train_classify.data(), y_train_classify.size(), 1);
	    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> XtTest(Xt_test_classify.data(), 784, Xt_test_classify.size()/784);
	    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> yTest(y_test_classify.data(), y_test_classify.size(), 1);
		//normalization
	    for(i=0;i<Xt.cols();++i){
	    	Xt.col(i) = Xt.col(i)/Xt.col(i).norm();
	    }
	    for(i=0;i<XtTest.cols();++i){
	    	XtTest.col(i) = XtTest.col(i)/XtTest.col(i).norm();
	    }
	    // VectorXd yy = y;
	    // VectorXd yyTest = yTest;
	    w = MatrixXd::Zero(Xt.rows(),1);
	    w = VectorXd(w);
	    gradients = (1+(-Xt.adjoint()*w).array().exp()).inverse() - y.array();
	    sumIG = Xt*gradients;
	    //DEBUG
	    cout << "Xt rows: " << Xt.rows() << endl << "Xt cols: " << Xt.cols() << endl;
	    cout << "XtTest rows: " << XtTest.rows() << endl << "XtTest cols: " << XtTest.cols() << endl;
	    cout << "y cols: " << y.size() << endl;
	    cout << "yTest cols: " << yTest.size() << endl;
	}
    string filename = "output";
	startTime = Clock::now();
    int nVars, nSamples, flag;
    epochCounter = 0;
    nVars = Xt.rows();
    nSamples = Xt.cols();
    fp = fopen(filename.c_str(), "a");
    if (fp == NULL) {
        cout << "Cannot write results to file: " << filename << endl;
    }
    epochCounter = 0;
    LogisticError(w, XtTest, yTest, 0, 0, fp);
    epochCounter = (epochCounter + 1) % PRINT_FREQ;
    switch(algorithmType){
    	if(batchSize==1){
    		if(!SPARSE){
		    	case 1:
		    		IAG_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime);break;
		    	case 2:
		    		IAGA_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime);break;
		    	case 3:
		    		SAG_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime);break;
		    	case 4:
		    		SAGA_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime);break;
		    	case 5:
		    		SGD_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime);break;
		    	case 6:
		    		SIG_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime);break;
		    	case 7:
		    		SVRG_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime);break;
    		}
    		else{
    			;
    		}
    	}
    	else if(batchSize>=2){
    		;
    	}
    }
	return;
}

int main(int argc, char* argv[]){
	int algorithmType, dataset=1;
	cout << "Available algorithms:" << endl << "1. IAG  " << "2. IAGA" << endl << "3. SAG  " << "4. SAGA" << endl << "5. SGD  " << "6. SID" << endl << "7. SVRG" << endl;
	cout << "Enter your choice of algorithm: " << endl;
	while(1){
		if(cin >> algorithmType){
			cout << "Your choice of algorithm: " << algorithmType << endl;
			cout << "Available datasets: " << endl << "1. MNIST" << endl;
			//default dataset is MNIST
			LogisticEntrance(algorithmType, dataset);
		}
		else{
			cout << "Invalid Input! Please intput a numerical value." << endl;
			cin.clear();
			while(cin.get()!='\n');
		}
	}
	return 0;
}