#ifndef SAGA_H
#define SAGA_H
#include "comm.h"
#include "LogisticError.h"
#include "RidgeError.h"
using namespace Eigen;

// void SAGA_logistic(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, \
// 	 VectorXd &yTest, VectorXd &d, VectorXd &g, string filename, double lambda=0.1, double eta=0.1, \
// 	int maxIter=60, int batchSize=1, int pass=20, double a=1, double b=1, int gamma=1,  int maxRunTime=20);

// void SAGA_logistic(VectorXd &w, const SparseMatrix<double> &Xt, int* innerIndices, int* outerStarts, const VectorXd &y, double lambda, double eta, VectorXd d, VectorXd g, \
// 	int maxIter, int batchSize, int pass, int a, int b, int gamma, const MatrixXd &XtTest, \
// 	const VectorXd &yTest, int maxRunTime, string filename);

void SAGA_init(MatrixXd &Xt, VectorXd &w, MatrixXd &XtTest, VectorXd &yTest, double &lambda, double &eta, double &a, double &b, double &gamma,\
	int &maxIter, int &batchSize, int &passes, int &maxRunTime, string &filename);
int SAGA_LogisticInnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime);
// int SAGA_LogsiticInnerLoopSingleSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);
int SAGA_LogisticInnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime, int batchSize); 
// int SAGA_LogsiticInnerLoopBatchSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize,int pass, double a, double b, double gamma);
int SAGA_RidgeSAGA_InnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime);
// int SAGA_RidgeInnerLoopSingleSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);
int SAGA_RidgeSAGA_InnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime, int batchSize); 
// int SAGA_RidgeInnerLoopBatchSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize,int pass, double a, double b, double gamma);




#endif 