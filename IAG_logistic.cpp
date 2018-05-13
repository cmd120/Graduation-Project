#include "include/IAG.h"
/*
IAG_logistic(w,Xt,y,lambda,eta,d,g);
% w(p,1) - updated in place
% Xt(p,n) - real, can be sparse
% y(n,1) - {0,1}
% lambda - scalar regularization param
% d(p,1) - initial approximation of average gradient (should be sum of previous
gradients)
% g(n,1) - previous derivatives of loss % maxIter - scalar maximal
iterations
% batchSize - mini-batch size m % pass - used to determine #iteration
% a - used to determine step sizes
% b - used to determine step sizes
% gamma - used to determine step sizes
% XtTest
% yTest
% maxRunTime
% filename - saving results
*/
int IAG_LogisticInnerLoopSingle(VectorXd &w, const MatrixXd &Xt, VectorXd &y,
                                const MatrixXd &XtTest, VectorXd &yTest,
                                VectorXd &d, VectorXd &g, double lambda,
                                long maxIter, int nSamples, int nVars, int pass,
                                double a, double b, double gamma,
                                int maxRunTime) {
  long i, idx, j;
  double innerProd = 0, tmpDelta, eta, telapsed;
  auto endTime = Clock::now();
  for (i = 0; i < maxIter; i++) {
    eta = a * pow(b + i + 1, -gamma);
    Noise noise(0.0, sqrt(eta * 2 / nSamples));
    idx = i % nSamples;
    innerProd = Xt.col(idx).dot(w);
    tmpDelta = LogisticPartialGradient(innerProd, y(idx));
    w = -eta / nSamples * d + (1 - eta * lambda) * w;
    w = NOISY ? w.array() + noise.gen() : w;
    w = w + (-eta) * (tmpDelta - g(idx)) / nSamples * Xt.col(idx);
    d = d + (tmpDelta - g(idx)) * Xt.col(idx);
    g(idx) = tmpDelta;
    // compute error
    if ((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ) {
      endTime = Clock::now();
      telapsed = chrono::duration_cast<chrono::nanoseconds>(endTime - startTime)
                     .count() /
                 BILLION;
      LogisticError(w, XtTest, yTest, pass + (i + 1) * 1.0 / maxIter, telapsed,
                    fp);
      epochCounter = (epochCounter + 1) % PRINT_FREQ;
      if (telapsed >= maxRunTime) {
        cout << "total telapsed: " << telapsed << endl;
        return 1;
      }
    }
  }
  return 0;
}

int IAG_LogisticInnerLoopBatch(VectorXd &w, const MatrixXd &Xt, VectorXd &y,
                               const MatrixXd &XtTest, VectorXd &yTest,
                               VectorXd &d, VectorXd &g, double lambda,
                               long maxIter, int nSamples, int nVars, int pass,
                               double a, double b, double gamma, int maxRunTime,
                               int batchSize) {
  long i, idx, j, k;
  double innerProd, eta, telapsed;
  auto endTime = Clock::now();
  VectorXd gradBuffer(batchSize);
  int *sampleBuffer = new int[batchSize];
  for (i = 0; i < maxIter; i++) {
    eta = a * pow(b + i + 1, -gamma);
    Noise noise(0.0, sqrt(eta * 2 / nSamples));
    for (k = 0; k < batchSize; k++) {
      idx = (i * batchSize + k) % nSamples;
      sampleBuffer[k] = idx;
      innerProd = Xt.col(idx).dot(w);
      gradBuffer(k) = LogisticPartialGradient(innerProd, y(idx));
    }
    w = -eta / nSamples * d + (1 - eta * lambda) * w;
    w = NOISY ? w.array() + noise.gen() : w;
    for (k = 0; k < batchSize; k++) {
      idx = sampleBuffer[k];
      w += (-eta * (gradBuffer(k) - g(idx)) / nSamples) * Xt.col(idx);
    }
    for (k = 0; k < batchSize; k++) {
      idx = sampleBuffer[k];
      d += (gradBuffer(k) - g(idx)) * Xt.col(idx);
      g(idx) = gradBuffer(k);
    }
    // compute error
    if ((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ) {
      endTime = Clock::now();
      telapsed = chrono::duration_cast<chrono::nanoseconds>(endTime - startTime)
                     .count() /
                 BILLION;
      LogisticError(w, XtTest, yTest, pass + (i + 1) * 1.0 / maxIter, telapsed,
                    fp);
      epochCounter = (epochCounter + 1) % PRINT_FREQ;
      if (telapsed >= maxRunTime) {
        return 1;
      }
    }
  }
  delete[] sampleBuffer;
  return 0;
}

int IAG_LogisticInnerLoopSingle(VectorXd &w, SparseMatrix<double> &Xt,
                                VectorXd &y, int *innerIndices,
                                int *outerStarts, SparseMatrix<double> &XtTest,
                                VectorXd &yTest, double lambda, VectorXd &d,
                                VectorXd &g, long maxIter, int nSamples,
                                int nVars, int pass, double a, double b,
                                double gamma, int maxRunTime) {
  long i, j, idx;
  double innerProd, tmpGrad, tmpFactor, eta, telapsed;
  double c = 1;
  int *lastVisited = new int[nVars];
  double *cumSum = new double[maxIter], *cumNoise = new double[maxIter];
  auto endTime = Clock::now();

  for (i = 0; i < nVars; ++i) {
    lastVisited[i] = 0;
  }
  for (i = 0; i < maxIter; ++i) {
    cumSum[i] = 0;
    cumNoise[i] = 0;
  }
  if (!Xt.isCompressed()) Xt.makeCompressed();
  if (!XtTest.isCompressed()) XtTest.makeCompressed();
  for (i = 0; i < maxIter; i++) {
    eta = a * pow(b + i + 1, -gamma);
    idx = i % nSamples;
    Noise noise(0.0, sqrt(eta * 2 / nSamples));
    if (i) {
      for (j = outerStarts[idx]; j < (long)outerStarts[idx + 1]; j++) {
        if (lastVisited[innerIndices[j]] == 0)
          w[innerIndices[j]] +=
              -d[innerIndices[j]] * cumSum[i - 1] + cumNoise[i - 1];
        else
          w[innerIndices[j]] +=
              -d[innerIndices[j]] *
                  (cumSum[i - 1] - cumSum[lastVisited[innerIndices[j]] - 1]) +
              cumNoise[i - 1] - cumNoise[lastVisited[innerIndices[j]] - 1];
        lastVisited[innerIndices[j]] = i;
      }
    }
    innerProd = 0;
    j = outerStarts[idx];
    for (SparseMatrix<double>::InnerIterator it(Xt, idx); it; ++it, ++j) {
      innerProd += w[innerIndices[j]] * it.value();
    }
    innerProd *= c;  // rescale
    tmpGrad = LogisticPartialGradient(innerProd, y[idx]);
    // update cumSum
    c *= 1 - eta * lambda;
    tmpFactor = eta / c / nSamples;
    if (i == 0) {
      cumSum[0] = tmpFactor;
      cumNoise[0] = NOISY ? noise.gen() : 0 / c;
    } else {
      cumSum[i] = cumSum[i - 1] + tmpFactor;
      cumNoise[i] = cumNoise[i - 1] + (NOISY ? noise.gen() : 0) / c;
    }

    /* Step 3: approximate w_{i+1} */
    tmpFactor =
        eta / c / nSamples * (tmpGrad - g[idx]);  // @NOTE biased estimator
    w += -tmpFactor * Xt.col(idx);
    // cblas_daxpyi(outerStarts[idx + 1] - outerStarts[idx], -tmpFactor, Xt +
    // outerStarts[idx], (int *)(innerIndices + outerStarts[idx]), w);
    // @NOTE (int *) here is 64bit because mwIndex is 64bit, and we have to link
    // libmkl_intel_ilp64.a for 64bit integer cout << "pass 5" << endl; Step 4:
    // update d and g(idx)
    j = outerStarts[idx];
    for (SparseMatrix<double>::InnerIterator it(Xt, idx); it; ++it, ++j) {
      d[innerIndices[j]] += it.value() * (tmpGrad - g[idx]);
    }
    g[idx] = tmpGrad;
    // Re-normalize the parameter vector if it has gone numerically crazy
    if (((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ) ||
        c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) ||
        (c < 0 && c > -1e-100)) {
      for (j = 0; j < nVars; j++) {
        if (lastVisited[j] == 0)
          w[j] += -d[j] * cumSum[i] + cumNoise[i];
        else
          w[j] += -d[j] * (cumSum[i] - cumSum[lastVisited[j] - 1]) +
                  cumNoise[i] - cumNoise[lastVisited[j] - 1];
        lastVisited[j] = i + 1;
      }
      cumSum[i] = 0;
      cumNoise[i] = 0;
      w = c * w;
      c = 1;

      // @NOTE compute error
      if ((i + 1) % maxIter ==
          maxIter * epochCounter / PRINT_FREQ)  // print test error
      {
        endTime = Clock::now();
        telapsed =
            chrono::duration_cast<chrono::nanoseconds>(endTime - startTime)
                .count() /
            BILLION;
        LogisticError(w, XtTest, yTest, pass + (i + 1) * 1.0 / maxIter,
                      telapsed, fp);
        epochCounter = (epochCounter + 1) % PRINT_FREQ;
        if (telapsed >= maxRunTime) {
          delete[] lastVisited;
          delete[] cumSum;
          delete[] cumNoise;
          delete[] outerStarts;
          return 1;
        }
      }
    }
  }
  // at last, correct the iterate once more
  for (j = 0; j < nVars; j++) {
    if (lastVisited[j] == 0) {
      w[j] += -d[j] * cumSum[maxIter - 1] + cumNoise[maxIter - 1];
    } else {
      w[j] += -d[j] * (cumSum[maxIter - 1] - cumSum[lastVisited[j] - 1]) +
              cumNoise[maxIter - 1] - cumNoise[lastVisited[j] - 1];
    }
  }
  w = c * w;
  delete[] lastVisited;
  delete[] cumSum;
  delete[] cumNoise;
  return 0;
}
int IAG_LogisticInnerLoopBatch(VectorXd &w, SparseMatrix<double> &Xt,
                               VectorXd &y, int *innerIndices, int *outerStarts,
                               SparseMatrix<double> &XtTest, VectorXd &yTest,
                               double lambda, VectorXd &d, VectorXd &g,
                               long maxIter, int nSamples, int nVars, int pass,
                               double a, double b, double gamma, int maxRunTime,
                               int batchSize) {
  int k;
  long i, idx, j;
  double innerProd, c = 1, tmpFactor, eta, telapsed;
  int *lastVisited = new int[nVars], *sampleBuffer = new int[batchSize];
  double *cumSum = new double[maxIter], *cumNoise = new double[maxIter],
         *gradBuffer = new double[batchSize];
  auto endTime = Clock::now();
  // initialization
  for (i = 0; i < nVars; ++i) {
    lastVisited[i] = 0;
  }
  for (i = 0; i < maxIter; ++i) {
    cumSum[i] = 0;
    cumNoise[i] = 0;
  }
  if (!Xt.isCompressed()) Xt.makeCompressed();
  if (!XtTest.isCompressed()) XtTest.makeCompressed();
  for (i = 0; i < maxIter; i++) {
    eta = a * pow(b + i + 1, -gamma);
    Noise noise(0.0, sqrt(eta * 2 / nSamples));
    for (k = 0; k < batchSize; k++)
      sampleBuffer[k] = (i * batchSize + k) % nSamples;
    /* Step 1: Compute current values of needed parameters w_{i} */
    if (i) {
      for (k = 0; k < batchSize; k++) {
        idx = sampleBuffer[k];
        for (j = outerStarts[idx]; j < (long)outerStarts[idx + 1]; j++) {
          if (lastVisited[innerIndices[j]] ==
              0)  // or we can let lastVisited[-1] = 0
            w[innerIndices[j]] +=
                -d[innerIndices[j]] * cumSum[i - 1] + cumNoise[i - 1];
          else if (lastVisited[innerIndices[j]] !=
                   i)  // if lastVisited[innerIndices[j]] > 0 && != i
            w[innerIndices[j]] +=
                -d[innerIndices[j]] *
                    (cumSum[i - 1] - cumSum[lastVisited[innerIndices[j]] - 1]) +
                cumNoise[i - 1] - cumNoise[lastVisited[innerIndices[j]] - 1];
          lastVisited[innerIndices[j]] = i;
        }
      }
    }
    /* Step 2:  Compute derivative of loss \nabla f(w_{i}) */
    for (k = 0; k < batchSize; k++) {
      idx = sampleBuffer[k];
      innerProd = 0;
      j = outerStarts[idx];
      for (SparseMatrix<double>::InnerIterator it(Xt, idx); it; ++it, ++j) {
        innerProd += w[innerIndices[j]] * it.value();
      }
      innerProd *= c;  // rescale
      gradBuffer[k] = LogisticPartialGradient(innerProd, y[idx]);
    }
    // update cumSum
    c *= 1 - eta * lambda;
    tmpFactor = eta / c / nSamples;
    if (i == 0) {
      cumSum[0] = tmpFactor;
      cumNoise[0] = noise.gen() / c;
    } else {
      cumSum[i] = cumSum[i - 1] + tmpFactor;
      cumNoise[i] = cumNoise[i - 1] + (NOISY ? noise.gen() : 0) / c;
    }
    /* Step 3: approximate w_{i+1} */
    for (k = 0; k < batchSize; k++) {
      idx = sampleBuffer[k];
      tmpFactor = eta / c / nSamples *
                  (gradBuffer[k] - g[idx]);  // @NOTE biased estimator
      w += -tmpFactor * Xt.col(idx);
      // cblas_daxpyi(outerStarts[idx + 1] - outerStarts[idx], -tmpFactor, Xt +
      // outerStarts[idx], (int *)(innerIndices + outerStarts[idx]), w);
      // @NOTE (int *) here is 64bit because mwIndex is 64bit, and we have to
      // link libmkl_intel_ilp64.a for 64bit integer
    }
    /* Step 4: update d and g[idx] */
    for (k = 0; k < batchSize; k++) {
      idx = sampleBuffer[k];
      j = outerStarts[idx];
      for (SparseMatrix<double>::InnerIterator it(Xt, idx); it; ++it, ++j) {
        d[innerIndices[j]] += it.value() * (gradBuffer[k] - g[idx]);
      }
      g[idx] = gradBuffer[k];
    }
    // Re-normalize the parameter vector if it has gone numerically crazy
    if (((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ) ||
        c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) ||
        (c < 0 && c > -1e-100)) {
      for (j = 0; j < nVars; j++) {
        if (lastVisited[j] == 0)
          w[j] += -d[j] * cumSum[i] + cumNoise[i];
        else
          w[j] += -d[j] * (cumSum[i] - cumSum[lastVisited[j] - 1]) +
                  cumNoise[i] - cumNoise[lastVisited[j] - 1];
        lastVisited[j] = i + 1;
      }
      cumSum[i] = 0;
      cumNoise[i] = 0;
      w = c * w;
      c = 1;

      // @NOTE compute error
      if ((i + 1) % maxIter ==
          maxIter * epochCounter / PRINT_FREQ)  // print test error
      {
        endTime = Clock::now();
        telapsed =
            chrono::duration_cast<chrono::nanoseconds>(endTime - startTime)
                .count() /
            BILLION;
        LogisticError(w, XtTest, yTest, pass + (i + 1) * 1.0 / maxIter,
                      telapsed, fp);
        epochCounter = (epochCounter + 1) % PRINT_FREQ;
        if (telapsed >= maxRunTime) {
          delete[] lastVisited;
          delete[] sampleBuffer;
          delete[] gradBuffer;
          delete[] cumSum;
          delete[] cumNoise;
          return 1;
        }
      }
    }
  }
  // at last, correct the iterate once more
  for (j = 0; j < nVars; j++) {
    if (lastVisited[j] == 0)
      w[j] += -d[j] * cumSum[maxIter - 1] + cumNoise[maxIter - 1];
    else
      w[j] += -d[j] * (cumSum[maxIter - 1] - cumSum[lastVisited[j] - 1]) +
              cumNoise[maxIter - 1] - cumNoise[lastVisited[j] - 1];
  }
  w = c * w;
  delete[] lastVisited;
  delete[] sampleBuffer;
  delete[] gradBuffer;
  delete[] cumSum;
  delete[] cumNoise;
  return 0;
}