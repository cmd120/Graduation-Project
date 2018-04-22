#include "include/IAG.h"
/*
IAG_logistic(w,Xt,y,lambda,eta,d,g);
% w(p,1) - updated in place
% Xt(p,n) - real, can be sparse
% y(n,1) - {0,1}
% lambda - scalar regularization param
% d(p,1) - initial approximation of average gradient (should be sum of previous gradients)
% g(n,1) - previous derivatives of loss
% maxIter - scalar maximal iterations
% batchSize - mini-batch size m
% pass - used to determine #iteration
% a - used to determine step sizes
% b - used to determine step sizes
% gamma - used to determine step sizes
% XtTest
% ytest
% maxRunTime
% filename - saving results
*/
void IAG_logistic(VectorXd &w, const MatrixXd &Xt, VectorXd y, const MatrixXd &XtTest, \
	const VectorXd &yTest, VectorXd d, VectorXd g, string filename, double lambda, double eta, \
	int maxIter, int batchSize, int pass, int a, int b, int gamma,  int maxRunTime) {
	
	int nVars, nSamples, flag;
	int epochCounter = 0;
	nVars = Xt.rows();
	nSamples = Xt.cols();
	FILE *fp = fopen(filename.c_str(), "a");
	if (fp == NULL) {
		cout << "Cannot write results to file: " << filename << endl;
	}
	LogisticError(w, XtTest, yTest, 0, 0, fp);
	epochCounter = (epochCounter + 1) % PRINT_FREQ;
	//为什么ret会在循环内部不断更新
	for (int i = 0; i < pass; i++) {
		flag = batchSize?InnerLoopBatchDense(w, Xt, y, d, g, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, batchSize):\
							InnerLoopSingleDense(w, Xt, y, d, g, lambda, maxIter, nSamples, nVars, pass, a, b, gamma);
		if (flag) {
			break;
		}
	}
	fclose(fp);
}
// void IAG_logistic(VectorXd &w, const SparseMatrix<double> &Xt, int* innerIndices, int* outerStarts, const VectorXd &y, double lambda, double eta, VectorXd d, VectorXd g, \
// 	int maxIter, int batchSize, int pass, int a, int b, int gamma, const MatrixXd &XtTest, \
// 	const VectorXd &yTest, int maxRunTime, string filename){

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
// 		flag = batchSize?InnerLoopBatchSparse(w, Xt, innerIndices, outerStarts, y, lambda, d, g, maxIter, nSamples, nVars, i+1, a, b, gamma):\
// 							InnerLoopSingleSparse(w, Xt, innerIndices, outerStarts, y, lambda, d, g, maxIter, nSamples, nVars, i+1, a, b, gamma);
// 		if (flag) {
// 			break;
// 		}
// 	}
// 	fclose(fp);
// }


int InnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd y, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma)
{
	long i, idx, j;
	double innerProd = 0 , tmpDelta, eta;
	Noise noise(0.0, sqrt(eta * 2 / nSamples));
	for (i = 0; i < maxIter; i++) {
		eta = a * pow(b + i + 1, -gamma);
		idx = i % nSamples;
		innerProd = Xt.col(idx).dot(w);
		tmpDelta = LogisticPartialGradient(innerProd, y(idx));
		w = -eta/nSamples*d+(1-eta*lambda)*w;
		w = NOISY?w.array()+noise.gen():w;
		w = w + (-eta) * (tmpDelta - g(idx)) / nSamples * Xt.col(idx);
		d = d + (tmpDelta - g(idx)) * Xt.col(idx);
		g(idx) = tmpDelta;

		// //compute error
		// if ((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ) {
		// 	LogisticError(w, XtTest, yTest, pass + (i + 1)*1.0 / maxIter, telapsed, fp);
		// 	epochCounter = (epochCounter + 1) % PRINT_FREQ;
		// 	if (telapsed >= maxRunTime) {
		// 		return 1;
		// 	}
		// }
	}
	return 0;
}

int InnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd y, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int batchSize)
{
	long i, idx, j, k;
	double innerProd, eta;

	VectorXd gradBuffer(batchSize);
	int* sampleBuffer = new int[batchSize];
	Noise noise(0.0, sqrt(eta * 2 / nSamples));
	for (i = 0; i < maxIter;i++) {
		eta = a * pow(b + i + 1, -gamma);
		for (k = 0; k < batchSize; k++) {
			idx = (i*batchSize + k) % nSamples;
			sampleBuffer[k] = idx;
			innerProd = Xt.col(idx).dot(w);
			gradBuffer(k) = LogisticPartialGradient(innerProd, y(idx));
		}

		w = -eta / nSamples * d + (1 - eta * lambda)*w;
		
		w = NOISY? w.array() + noise.gen():w;

		for (k = 0; k < batchSize; k++) {
			idx = sampleBuffer[k];
			w += (-eta * (gradBuffer(k) - g(idx)) / nSamples)*Xt.col(idx);
		}
		for (k = 0; k < batchSize; k++) {
			idx = sampleBuffer[k];
			d += (gradBuffer(k) - g(idx))*Xt.col(idx);
			g(idx) = gradBuffer(k);
		}
		// if ((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ) {
		// 	...
		// 		LogisticError(w, XtTest, yTest, pass + (i + 1)*1.0 / maxIter, telapsed, fp);
		// 	epochCounter = (epochCounter + 1) % PRINT_FREQ;
		// 	if (telapsed >= maxRunTime) {
		// 		return 1;
		// 	}
		// }
	}
	delete[] sampleBuffer;
	return 0;
}


// int InnerLoopSingleSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma)
// {
// 	long i, j, idx;
// 	double innerProd, tmpGrad, tmpFactor, eta;
// 	double c = 1;

// 	int *lastVisited = new int[nVars];
//     double *cumSum = new double[maxIter], *cumNoise = new double[maxIter];
//     // int *innerIndices = mat.innerIndexPtr();
//     // int *outerStarts = new int[mat.cols()];
//     // InitOuterStarts(mat, outerStarts);
// 	for (i = 0; i < maxIter; i++) {
// 		eta = a * pow(b + i + 1, -gamma);
// 		idx = i % nSamples;
// 		Noise noise(0.0, sqrt(eta * 2 / nSamples));
// 		if (i) {
// 			for (j = outerStarts[idx]; j < (long)outerStarts[idx + 1]; j++) {
// 				if (lastVisited[innerIndices[j]] == 0)
// 					w[innerIndices[j]] += -d[innerIndices[j]] * cumSum[i - 1] + cumNoise[i - 1];
// 				else
// 					w[innerIndices[j]] += -d[innerIndices[j]] * (cumSum[i - 1] - cumSum[lastVisited[innerIndices[j]] - 1]) + cumNoise[i - 1] - cumNoise[lastVisited[innerIndices[j]] - 1];
// 				lastVisited[innerIndices[j]] = i;
// 			}
// 		}

// 		innerProd = 0;
// 		for (j = outerStarts[idx]; j < (long)outerStarts[idx + 1]; j++) {
// 			innerProd += w[innerIndices[j]] * Xt[j];
// 		}
// 		innerProd *= c;  // rescale
// 		tmpGrad = LogisticPartialGradient(innerProd, y[idx]);

// 		// update cumSum
// 		c *= 1 - eta * lambda;
// 		tmpFactor = eta / c / nSamples;

// 		if (i == 0)
// 		{
// 			cumSum[0] = tmpFactor;
// 			cumNoise[0] = NOISY?noise.gen():0 / c;
// 		}
// 		else
// 		{
// 			cumSum[i] = cumSum[i - 1] + tmpFactor;
// 			cumNoise[i] = cumNoise[i - 1] + NOISY?noise.gen():0 / c;
// 		}

// 		/* Step 3: approximate w_{i+1} */
// 		tmpFactor = eta / c / nSamples * (tmpGrad - g(idx));  // @NOTE biased estimator
// 		//该如何转换成Eigen的表达，可以考虑提问
// 		cblas_daxpyi(outerStarts[idx + 1] - outerStarts[idx], -tmpFactor, Xt + outerStarts[idx], (int *)(innerIndices + outerStarts[idx]), w);
// 		// @NOTE (int *) here is 64bit because mwIndex is 64bit, and we have to link libmkl_intel_ilp64.a for 64bit integer

// 		 Step 4: update d and g(idx) 
// 		for (j = outerStarts[idx]; j < (long)outerStarts[idx + 1]; j++)
// 			d[innerIndices[j]] += Xt[j] * (tmpGrad - g(idx));
// 		g(idx) = tmpGrad;

// 		// Re-normalize the parameter vector if it has gone numerically crazy
// 		if (((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ) || c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) || (c < 0 && c > -1e-100))
// 		{
// 			for (j = 0; j < nVars; j++)
// 			{
// 				if (lastVisited[j] == 0)
// 					w[j] += -d[j] * cumSum[i] + cumNoise[i];
// 				else
// 					w[j] += -d[j] * (cumSum[i] - cumSum[lastVisited[j] - 1]) + cumNoise[i] - cumNoise[lastVisited[j] - 1];
// 				lastVisited[j] = i + 1;
// 			}
// 			cumSum[i] = 0;
// 			cumNoise[i] = 0;
//             w = c * w;
// 			// cblas_dscal(nVars, c, w, 1);
// 			c = 1;

// 			// @NOTE compute error
// 			if ((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ)// print test error
// 			{
// 				clock_gettime(CLOCK_MONOTONIC_RAW, &requestEnd);
// 				telapsed = (requestEnd.tv_sec - requestStart.tv_sec) + (requestEnd.tv_nsec - requestStart.tv_nsec) / BILLION;

// 				LogisticError(w, XtTest, yTest, pass + (i + 1)*1.0 / maxIter, telapsed, fp);
// 				epochCounter = (epochCounter + 1) % PRINT_FREQ;
// 				if (telapsed >= maxRunTime)
// 				{
//                     delete[] lastVisited;
//                     delete[] cumSum;
//                     delete[] cumNoise;
//                     delete[] outerStarts;
// 					return 1;
// 				}
// 			}
// 		}
// 	}

// 	// at last, correct the iterate once more
// 	for (j = 0; j < nVars; j++)
// 	{
// 		if (lastVisited[j] == 0)
// 			w[j] += -d[j] * cumSum[maxIter - 1] + cumNoise[maxIter - 1];
// 		else
// 			w[j] += -d[j] * (cumSum[maxIter - 1] - cumSum[lastVisited[j] - 1]) + cumNoise[maxIter - 1] - cumNoise[lastVisited[j] - 1];
// 	}
// 	w = c * w;
//     delete[] lastVisited;
//     delete[] cumSum;
//     delete[] cumNoise;
//     delete[] outerStarts;
// 	return 0;
// }
// int InnerLoopBatchSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize,int pass, double a, double b, double gamma)
// {

// }