#include "include/IAG.h"
#include "include/IAGA.h"
#include "include/MNIST_Read.h"
#include "include/SAG.h"
#include "include/SAGA.h"
#include "include/SGD.h"
#include "include/SIG.h"
#include "include/SVRG.h"
#include "include/covtype.h"

int epochCounter;
FILE *fp;
std::chrono::high_resolution_clock::time_point startTime;
int SPARSE;
LR objFuncLR;
RR objFuncRR;
int LogisticEntrance(int algorithmType, int datasetNum,
                     SparseMatrix<double> &XtS, VectorXd &y,
                     SparseMatrix<double> &XtTestS, VectorXd &yTest) {
  VectorXd w, wtilde, G, sumIG, gradients;
  double lambda, eta, a, b, gamma;
  int maxIter, batchSize, passes, maxRunTime;
  SPARSE = 0;
  int nVars, nSamples, flag;
  string filename;
  int *innerIndices, *outerStarts;
  innerIndices = XtS.innerIndexPtr();
  outerStarts = new int[XtS.cols()];
  if (!outerStarts) {
    cout << "run out of space!" << endl;
  }
  InitOuterStarts(XtS, outerStarts);
  w = MatrixXd::Zero(XtS.rows(), 1);
  wtilde = w;
  G = w;
  gradients = (1 + (-XtS.adjoint() * w).array().exp()).inverse() - y.array();
  sumIG = XtS * gradients;
  epochCounter = 0;
  nVars = XtS.rows();
  nSamples = XtS.cols();
  switch (algorithmType) {
    case 1:
      filename = "IAG";
      algorithmInit(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        if (batchSize >= 2
                ? IAG_LogisticInnerLoopBatch(w, XtS, y, XtTestS, yTest, sumIG,
                                             gradients, lambda, maxIter,
                                             nSamples, nVars, pass, a, b, gamma,
                                             maxRunTime, batchSize)
                : IAG_LogisticInnerLoopSingle(
                      w, XtS, y, XtTestS, yTest, sumIG, gradients, lambda,
                      maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    case 2:
      filename = "IAGA";
      algorithmInit(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        if (batchSize >= 2
                ? IAGA_LogisticInnerLoopBatch(w, XtS, y, XtTestS, yTest, sumIG,
                                              gradients, lambda, maxIter,
                                              nSamples, nVars, pass, a, b,
                                              gamma, maxRunTime, batchSize)
                : IAGA_LogisticInnerLoopSingle(
                      w, XtS, y, XtTestS, yTest, sumIG, gradients, lambda,
                      maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    case 3:
      filename = "SAG";
      algorithmInit(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        if (batchSize >= 2
                ? SAG_LogisticInnerLoopBatch(w, XtS, y, XtTestS, yTest, sumIG,
                                             gradients, lambda, maxIter,
                                             nSamples, nVars, pass, a, b, gamma,
                                             maxRunTime, batchSize)
                : SAG_LogisticInnerLoopSingle(
                      w, XtS, y, XtTestS, yTest, sumIG, gradients, lambda,
                      maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    case 4:
      filename = "SAGA";
      algorithmInit(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        if (batchSize >= 2
                ? SAGA_LogisticInnerLoopBatch(w, XtS, y, XtTestS, yTest, sumIG,
                                              gradients, lambda, maxIter,
                                              nSamples, nVars, pass, a, b,
                                              gamma, maxRunTime, batchSize)
                : SAGA_LogisticInnerLoopSingle(
                      w, XtS, y, XtTestS, yTest, sumIG, gradients, lambda,
                      maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    case 5:
      filename = "SGD";
      algorithmInit(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        if (batchSize >= 2
                ? SGD_LogisticInnerLoopBatch(w, XtS, y, XtTestS, yTest, sumIG,
                                             gradients, lambda, maxIter,
                                             nSamples, nVars, pass, a, b, gamma,
                                             maxRunTime, batchSize)
                : SGD_LogisticInnerLoopSingle(
                      w, XtS, y, XtTestS, yTest, sumIG, gradients, lambda,
                      maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    case 6:
      filename = "SIG";
      algorithmInit(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        LogisticGradient(wtilde, G, XtS, y);
        if (batchSize >= 2
                ? SIG_LogisticInnerLoopBatch(
                      w, XtS, y, XtTestS, yTest, wtilde, G, lambda, maxIter,
                      nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
                : SIG_LogisticInnerLoopSingle(
                      w, XtS, y, XtTestS, yTest, wtilde, G, lambda, maxIter,
                      nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    case 7:
      filename = "SVRG";
      algorithmInit(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        LogisticGradient(wtilde, G, XtS, y);
        if (batchSize >= 2
                ? SVRG_LogisticInnerLoopBatch(
                      w, XtS, y, XtTestS, yTest, wtilde, G, lambda, maxIter,
                      nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
                : SVRG_LogisticInnerLoopSingle(
                      w, XtS, y, XtTestS, yTest, wtilde, G, lambda, maxIter,
                      nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    default:;
  }
  MatrixXd tmpXt = MatrixXd(XtS), tmpXtTest = MatrixXd(XtTestS);
  printf("training accuracy: %f\n", objFuncLR.score(w, tmpXt, y));
  printf("test accuracy: %f\n", objFuncLR.score(w, tmpXtTest, yTest));
  // fprintf('time elapsed: %f\n', telapsed);
  return 0;
}
int LogisticEntrance(int algorithmType, int datasetNum, MatrixXd &Xt,
                     VectorXd &y, MatrixXd &XtTest, VectorXd &yTest) {
  VectorXd w, wtilde, G, sumIG, gradients;
  double lambda, eta, a, b, gamma;
  int maxIter, batchSize, passes, maxRunTime;
  SPARSE = 0;
  int nVars, nSamples, flag;
  string filename;
  w = MatrixXd::Zero(Xt.rows(), 1);
  wtilde = w;
  G = w;
  gradients = (1 + (-Xt.adjoint() * w).array().exp()).inverse() - y.array();
  sumIG = Xt * gradients;
  epochCounter = 0;
  nVars = Xt.rows();
  nSamples = Xt.cols();
  switch (algorithmType) {
    case 1:
      filename = "IAG";
      algorithmInit(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        if (batchSize >= 2
                ? IAG_LogisticInnerLoopBatch(w, Xt, y, XtTest, yTest, sumIG,
                                             gradients, lambda, maxIter,
                                             nSamples, nVars, pass, a, b, gamma,
                                             maxRunTime, batchSize)
                : IAG_LogisticInnerLoopSingle(
                      w, Xt, y, XtTest, yTest, sumIG, gradients, lambda,
                      maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    case 2:
      filename = "IAGA";
      algorithmInit(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        if (batchSize >= 2
                ? IAGA_LogisticInnerLoopBatch(w, Xt, y, XtTest, yTest, sumIG,
                                              gradients, lambda, maxIter,
                                              nSamples, nVars, pass, a, b,
                                              gamma, maxRunTime, batchSize)
                : IAGA_LogisticInnerLoopSingle(
                      w, Xt, y, XtTest, yTest, sumIG, gradients, lambda,
                      maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    case 3:
      filename = "SAG";
      algorithmInit(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        if (batchSize >= 2
                ? SAG_LogisticInnerLoopBatch(w, Xt, y, XtTest, yTest, sumIG,
                                             gradients, lambda, maxIter,
                                             nSamples, nVars, pass, a, b, gamma,
                                             maxRunTime, batchSize)
                : SAG_LogisticInnerLoopSingle(
                      w, Xt, y, XtTest, yTest, sumIG, gradients, lambda,
                      maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    case 4:
      filename = "SAGA";
      algorithmInit(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        if (batchSize >= 2
                ? SAGA_LogisticInnerLoopBatch(w, Xt, y, XtTest, yTest, sumIG,
                                              gradients, lambda, maxIter,
                                              nSamples, nVars, pass, a, b,
                                              gamma, maxRunTime, batchSize)
                : SAGA_LogisticInnerLoopSingle(
                      w, Xt, y, XtTest, yTest, sumIG, gradients, lambda,
                      maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    case 5:
      filename = "SGD";
      algorithmInit(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        if (batchSize >= 2
                ? SGD_LogisticInnerLoopBatch(w, Xt, y, XtTest, yTest, sumIG,
                                             gradients, lambda, maxIter,
                                             nSamples, nVars, pass, a, b, gamma,
                                             maxRunTime, batchSize)
                : SGD_LogisticInnerLoopSingle(
                      w, Xt, y, XtTest, yTest, sumIG, gradients, lambda,
                      maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    case 6:
      filename = "SIG";
      algorithmInit(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        LogisticGradient(wtilde, G, Xt, y);
        if (batchSize >= 2
                ? SIG_LogisticInnerLoopBatch(
                      w, Xt, y, XtTest, yTest, wtilde, G, lambda, maxIter,
                      nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
                : SIG_LogisticInnerLoopSingle(
                      w, Xt, y, XtTest, yTest, wtilde, G, lambda, maxIter,
                      nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    case 7:
      filename = "SVRG";
      algorithmInit(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        LogisticGradient(wtilde, G, Xt, y);
        if (batchSize >= 2
                ? SVRG_LogisticInnerLoopBatch(
                      w, Xt, y, XtTest, yTest, wtilde, G, lambda, maxIter,
                      nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
                : SVRG_LogisticInnerLoopSingle(
                      w, Xt, y, XtTest, yTest, wtilde, G, lambda, maxIter,
                      nSamples, nVars, pass, a, b, gamma, maxRunTime))
          break;
      }
      break;
    default:;
  }
  printf("training accuracy: %f\n", objFuncLR.score(w, Xt, y));
  printf("test accuracy: %f\n", objFuncLR.score(w, XtTest, yTest));
  // fprintf('time elapsed: %f\n', telapsed);
  return 0;
}

void datasetOption(int &datasetNum) {
  const int NUMBEROFDATASET = 2;
  const string datasets[NUMBEROFDATASET] = {"MNIST", "COVTYPE"};
  cout << "Available datasets to choose from:" << endl;
  for (int i = 0; i < NUMBEROFDATASET; ++i) {
    cout << i + 1 << "." << datasets[i] << endl;
  }
  cout << "Enter your choice of dataset: " << endl;
  cin >> datasetNum;
  cout << "Your choice of dataset: " << datasets[datasetNum - 1] << endl;
  return;
}
void algorithmOption(int &algorithmType) {
  cout << "123" << endl;
  const int NUMBEROFAlGORITHM = 7;
  const string algorithms[NUMBEROFAlGORITHM] = {"IAG", "IAGA", "SAG", "SAGA",
                                                "SGD", "SIG",  "SVRG"};
  cout << "Enter your choice of algorithm: (0 to quit)" << endl;
  for (int i = 0; i < NUMBEROFAlGORITHM; ++i) {
    cout << i + 1 << "." << algorithms[i] << endl;
  }
  while (1) {
    if (cin >> algorithmType) {
      if (algorithmType)
        cout << "Your choice of algorithm: " << algorithms[algorithmType - 1]
             << endl;
      else
        cout << "Bye" << endl;
      break;
    } else {
      cout << "Invalid Input! Please intput a numerical value." << endl;
      cin.clear();
      while (cin.get() != '\n')
        ;
    }
  }
  return;
}

int main(int argc, char *argv[]) {
  MatrixXd Xt, XtTest;
  SparseMatrix<double> XtS, XtTestS;
  VectorXd y, yTest;
  int algorithmType = 0, datasetNum;

  datasetOption(datasetNum);
  switch (datasetNum) {
    case 1:
      mnist_read(Xt, y, XtTest, yTest);
      break;
    case 2:
      covtype_read(Xt, y, XtTest, yTest);
      break;
  }
  cout << "dataset loaded." << endl;
  if (SPARSE) {
    XtS = Xt.sparseView();
    XtTestS = XtTest.sparseView();
    cout << "dataset is sparse" << endl;
  } else {
    cout << "dataset is dense" << endl;
  }
  while (1) {
    algorithmOption(algorithmType);
    if (algorithmType) {
      int ret;
      if (SPARSE) {
        ret =
            LogisticEntrance(algorithmType, datasetNum, XtS, y, XtTestS, yTest);
      } else {
        ret = LogisticEntrance(algorithmType, datasetNum, Xt, y, XtTest, yTest);
      }
      if (ret) break;
    } else {
      break;
    }
  }
  return 0;
}