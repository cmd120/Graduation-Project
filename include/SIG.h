#ifndef SIG_H
#define SIG_H
#include "comm.h"
#include "LogisticError.h"

using namespace Eigen;

void SIG_logistic(VectorXd &w, const MatrixXd &Xt, VectorXd y, const MatrixXd &XtTest, \
	const VectorXd &yTest, VectorXd wtilde, VectorXd G, string filename, double lambda=0.1, double eta=0.1, \
	int maxIter=60, int batchSize=1, int pass=20, int a=1, int b=1, int gamma=1,  int maxRunTime=20);

void SIG_logistic(VectorXd &w, const SparseMatrix<double> &Xt, int* innerIndices, int* outerStarts, const VectorXd &y, double lambda, double eta, VectorXd wtilde, VectorXd G, \
	int maxIter, int batchSize, int pass, int a, int b, int gamma, const MatrixXd &XtTest, \
	const VectorXd &yTest, int maxRunTime, string filename);


int InnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd y, VectorXd &wtilde, VectorXd &G, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma);
int InnerLoopSingleSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd wtilde, double *G, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);
int InnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd y, VectorXd &wtilde, VectorXd &G, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int batchSize);
int InnerLoopBatchSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd wtilde, double *G, long maxIter, int nSamples, int nVars, int batchSize,int pass, double a, double b, double gamma);




#endif 