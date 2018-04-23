#ifndef IAGA_H
#define IAGA_H
#include "comm.h"
#include "LogisticError.h"
#include "RidgeError.h"
void IAGA_logistic(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, \
	 VectorXd &yTest, VectorXd d, VectorXd g, string filename, double lambda=0.1, double eta=0.1, \
	int maxIter=60, int batchSize=1, int pass=20, int a=1, int b=1, int gamma=1,  int maxRunTime=20);

// void IAGA_logistic(const VectorXd &w, const SparseMatrix<double> &Xt, int* innerIndices, int* outerStarts, const VectorXd &y, double lambda, double eta, VectorXd d, double* g, \
// 	int maxIter, int batchSize, int pass, int a, int b, int gamma, const MatrixXd &XtTest, \
// 	const VectorXd &yTest, int maxRunTime, string filename);

int InnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime);
// int InnerLoopSingleSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);
int InnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int batchSize,int pass, double a, double b, double gamma, int maxRunTime);
// int InnerLoopBatchSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize,int pass, double a, double b, double gamma);

// void IAGA_logistic(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, \
// 	const VectorXd &yTest, VectorXd d, VectorXd g, string filename, double lambda, double eta, \
// 	int maxIter, int batchSize, int pass, int a, int b, int gamma,  int maxRunTime) {
	
// 	int nVars, nSamples, flag;
// 	int epochCounter = 0;
// 	nVars = Xt.rows();
// 	nSamples = Xt.cols();
// 	FILE *fp = fopen(filename.c_str(), "a");
// 	if (fp == NULL) {
// 		cout << "Cannot write results to file: " << filename << endl;
// 	}
// 	LogisticError(w, XtTest, yTest, 0, 0, fp);
// 	epochCounter = (epochCounter + 1) % PRINT_FREQ;
// 	//为什么ret会在循环内部不断更新
// 	for (int i = 0; i < pass; i++) {
// 		flag = batchSize?InnerLoopBatchDense(w, Xt, y, lambda, d, g, maxIter, nSamples, nVars, pass, a, b, gamma, batchSize):\
// 							InnerLoopSingleDense(w, Xt, y, lambda, d, g, maxIter, nSamples, nVars, pass, a, b, gamma);
// 		if (flag) {
// 			break;
// 		}
// 	}
// 	fclose(fp);
// }

// int InnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, const VectorXd &y, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma){
// 	;
// }
// int InnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, const VectorXd &y, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int batchSize,int pass, double a, double b, double gamma){
// 	;
// }
// int main(){
// 	return 0;
// }
#endif