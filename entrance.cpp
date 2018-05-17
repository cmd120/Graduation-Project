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
                     Eigen::SparseMatrix<double> &XtS, Eigen::VectorXd &y,
                     Eigen::SparseMatrix<double> &XtTestS,
                     Eigen::VectorXd &yTest) {
  Eigen::VectorXd w, wtilde, G, sumIG, gradients;
  double lambda, eta, a, b, gamma;
  int maxIter, batchSize, passes, maxRunTime;
  SPARSE = 0;
  int nVars, nSamples, flag;
  std::string filename;
  int *innerIndices, *outerStarts;
  innerIndices = XtS.innerIndexPtr();
  outerStarts = new int[XtS.cols()];
  if (!outerStarts) {
    std::cout << "run out of space!" << std::endl;
  }
  InitOuterStarts(XtS, outerStarts);
  w = Eigen::MatrixXd::Zero(XtS.rows(), 1);
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
        batchSize >= 2
            ? IAG_LogisticInnerLoopBatch(
                  w, XtS, y, XtTestS, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
            : IAG_LogisticInnerLoopSingle(w, XtS, y, XtTestS, yTest, sumIG,
                                          gradients, lambda, maxIter, nSamples,
                                          nVars, pass, a, b, gamma, maxRunTime);
      }
      break;
    case 2:
      filename = "IAGA";
      algorithmInit(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        batchSize >= 2
            ? IAGA_LogisticInnerLoopBatch(
                  w, XtS, y, XtTestS, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
            : IAGA_LogisticInnerLoopSingle(
                  w, XtS, y, XtTestS, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime);
      }
      break;
    case 3:
      filename = "SAG";
      algorithmInit(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        batchSize >= 2
            ? SAG_LogisticInnerLoopBatch(
                  w, XtS, y, XtTestS, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
            : SAG_LogisticInnerLoopSingle(w, XtS, y, XtTestS, yTest, sumIG,
                                          gradients, lambda, maxIter, nSamples,
                                          nVars, pass, a, b, gamma, maxRunTime);
      }
      break;
    case 4:
      filename = "SAGA";
      algorithmInit(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        batchSize >= 2
            ? SAGA_LogisticInnerLoopBatch(
                  w, XtS, y, XtTestS, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
            : SAGA_LogisticInnerLoopSingle(
                  w, XtS, y, XtTestS, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime);
      }
      break;
    case 5:
      filename = "SGD";
      algorithmInit(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        batchSize >= 2
            ? SGD_LogisticInnerLoopBatch(
                  w, XtS, y, XtTestS, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
            : SGD_LogisticInnerLoopSingle(w, XtS, y, XtTestS, yTest, sumIG,
                                          gradients, lambda, maxIter, nSamples,
                                          nVars, pass, a, b, gamma, maxRunTime);
      }
      break;
    case 6:
      filename = "SIG";
      algorithmInit(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        LogisticGradient(wtilde, G, innerIndices, outerStarts, XtS, y);
        batchSize >= 2
            ? SIG_LogisticInnerLoopBatch(w, XtS, y, innerIndices, outerStarts,
                                         XtTestS, yTest, wtilde, G, lambda,
                                         maxIter, nSamples, nVars, pass, a, b,
                                         gamma, maxRunTime, batchSize)
            : SIG_LogisticInnerLoopSingle(w, XtS, y, innerIndices, outerStarts,
                                          XtTestS, yTest, wtilde, G, lambda,
                                          maxIter, nSamples, nVars, pass, a, b,
                                          gamma, maxRunTime);
      }
      wtilde = w;
      break;
    case 7:
      filename = "SVRG";
      algorithmInit(XtS, w, XtTestS, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        LogisticGradient(wtilde, G, innerIndices, outerStarts, XtS, y);
        batchSize >= 2
            ? SVRG_LogisticInnerLoopBatch(w, XtS, y, innerIndices, outerStarts,
                                          XtTestS, yTest, wtilde, G, lambda,
                                          maxIter, nSamples, nVars, pass, a, b,
                                          gamma, maxRunTime, batchSize)
            : SVRG_LogisticInnerLoopSingle(w, XtS, y, innerIndices, outerStarts,
                                           XtTestS, yTest, wtilde, G, lambda,
                                           maxIter, nSamples, nVars, pass, a, b,
                                           gamma, maxRunTime);
      }
      wtilde = w;
      break;
    default:;
  }
  if (DEBUG) {
    std::cout << "hypothesis: "
              << 1 / (1 + (-XtTestS.adjoint() * w).array().exp()) << std::endl;
  }
  Eigen::MatrixXd tmpXt = Eigen::MatrixXd(XtS),
                  tmpXtTest = Eigen::MatrixXd(XtTestS);
  auto endTime = Clock::now();
  printf("telapsed %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(
                              endTime - startTime)
                                  .count() /
                              BILLION);
  printf("training accuracy: %f\n", objFuncLR.score(w, tmpXt, y));
  printf("test accuracy: %f\n", objFuncLR.score(w, tmpXtTest, yTest));
  // fprintf('time elapsed: %f\n', telapsed);
  return 0;
}
int LogisticEntrance(int algorithmType, int datasetNum, Eigen::MatrixXd &Xt,
                     Eigen::VectorXd &y, Eigen::MatrixXd &XtTest,
                     Eigen::VectorXd &yTest) {
  Eigen::VectorXd w, wtilde, G, sumIG, gradients;
  double lambda, eta, a, b, gamma;
  int maxIter, batchSize, passes, maxRunTime;
  SPARSE = 0;
  int nVars, nSamples, flag;
  std::string filename;
  w = Eigen::MatrixXd::Zero(Xt.rows(), 1);
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
        batchSize >= 2
            ? IAG_LogisticInnerLoopBatch(
                  w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
            : IAG_LogisticInnerLoopSingle(w, Xt, y, XtTest, yTest, sumIG,
                                          gradients, lambda, maxIter, nSamples,
                                          nVars, pass, a, b, gamma, maxRunTime);
      }
      break;
    case 2:
      filename = "IAGA";
      algorithmInit(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        batchSize >= 2
            ? IAGA_LogisticInnerLoopBatch(
                  w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
            : IAGA_LogisticInnerLoopSingle(
                  w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime);
      }
      break;
    case 3:
      filename = "SAG";
      algorithmInit(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        batchSize >= 2
            ? SAG_LogisticInnerLoopBatch(
                  w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
            : SAG_LogisticInnerLoopSingle(w, Xt, y, XtTest, yTest, sumIG,
                                          gradients, lambda, maxIter, nSamples,
                                          nVars, pass, a, b, gamma, maxRunTime);
      }
      break;
    case 4:
      filename = "SAGA";
      algorithmInit(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        batchSize >= 2
            ? SAGA_LogisticInnerLoopBatch(
                  w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
            : SAGA_LogisticInnerLoopSingle(
                  w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime);
      }
      break;
    case 5:
      filename = "SGD";
      algorithmInit(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        batchSize >= 2
            ? SGD_LogisticInnerLoopBatch(
                  w, Xt, y, XtTest, yTest, sumIG, gradients, lambda, maxIter,
                  nSamples, nVars, pass, a, b, gamma, maxRunTime, batchSize)
            : SGD_LogisticInnerLoopSingle(w, Xt, y, XtTest, yTest, sumIG,
                                          gradients, lambda, maxIter, nSamples,
                                          nVars, pass, a, b, gamma, maxRunTime);
      }
      break;
    case 6:
      filename = "SIG";
      algorithmInit(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        LogisticGradient(wtilde, G, Xt, y);
        batchSize >= 2
            ? SIG_LogisticInnerLoopBatch(w, Xt, y, XtTest, yTest, wtilde, G,
                                         lambda, maxIter, nSamples, nVars, pass,
                                         a, b, gamma, maxRunTime, batchSize)
            : SIG_LogisticInnerLoopSingle(w, Xt, y, XtTest, yTest, wtilde, G,
                                          lambda, maxIter, nSamples, nVars,
                                          pass, a, b, gamma, maxRunTime);
      }
      break;
    case 7:
      filename = "SVRG";
      algorithmInit(Xt, w, XtTest, yTest, lambda, eta, a, b, gamma, maxIter,
                    batchSize, passes, maxRunTime, filename, datasetNum);
      for (int pass = 0; pass < passes; ++pass) {
        LogisticGradient(wtilde, G, Xt, y);
        batchSize >= 2
            ? SVRG_LogisticInnerLoopBatch(
                  w, Xt, y, XtTest, yTest, wtilde, G, lambda, maxIter, nSamples,
                  nVars, pass, a, b, gamma, maxRunTime, batchSize)
            : SVRG_LogisticInnerLoopSingle(w, Xt, y, XtTest, yTest, wtilde, G,
                                           lambda, maxIter, nSamples, nVars,
                                           pass, a, b, gamma, maxRunTime);
      }
      break;
    default:;
  }
  if (DEBUG) {
    // std::cout << "Xt'*w: " << XtTest.adjoint() * w << std::endl;
    // std::cout << "yTest: " << yTest << std::endl;
    // std::cout << "w:" << w << std::endl;
    std::cout << "hypothesis: "
              << 1 / (1 + (-XtTest.adjoint() * w).array().exp()) << std::endl;
  }
  auto endTime = Clock::now();
  printf("telapsed %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(
                              endTime - startTime)
                                  .count() /
                              BILLION);
  printf("training accuracy: %f\n", objFuncLR.score(w, Xt, y));
  printf("test accuracy: %f\n", objFuncLR.score(w, XtTest, yTest));
  // fprintf('time elapsed: %f\n', telapsed);
  return 0;
}

void datasetOption(int &datasetNum) {
  const int NUMBEROFDATASET = 2;
  const std::string datasets[NUMBEROFDATASET] = {"MNIST", "COVTYPE"};
  std::cout << "Available datasets to choose from:" << std::endl;
  for (int i = 0; i < NUMBEROFDATASET; ++i) {
    std::cout << i + 1 << "." << datasets[i] << std::endl;
  }
  std::cout << "Enter your choice of dataset: " << std::endl;
  std::cin >> datasetNum;
  std::cout << "Your choice of dataset: " << datasets[datasetNum - 1]
            << std::endl;
  return;
}
void algorithmOption(int &algorithmType) {
  const int NUMBEROFAlGORITHM = 7;
  const std::string algorithms[NUMBEROFAlGORITHM] = {"IAG", "IAGA", "SAG", "SAGA",
                                                "SGD", "SIG",  "SVRG"};
  std::cout << "Enter your choice of algorithm: (0 to quit)" << std::endl;
  for (int i = 0; i < NUMBEROFAlGORITHM; ++i) {
    std::cout << i + 1 << "." << algorithms[i] << std::endl;
  }
  while (1) {
    if (std::cin >> algorithmType) {
      if (algorithmType)
        std::cout << "Your choice of algorithm: "
                  << algorithms[algorithmType - 1] << std::endl;
      else
        std::cout << "Bye" << std::endl;
      break;
    } else {
      std::cout << "Invalid Input! Please intput a numerical value."
                << std::endl;
      std::cin.clear();
      while (std::cin.get() != '\n')
        ;
    }
  }
  return;
}

int main(int argc, char *argv[]) {
  Eigen::MatrixXd Xt, XtTest;
  Eigen::SparseMatrix<double> XtS, XtTestS;
  Eigen::VectorXd y, yTest;
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
  std::cout << "dataset loaded." << std::endl;
  if (SPARSE) {
    XtS = Xt.sparseView();
    XtTestS = XtTest.sparseView();
    std::cout << "dataset is sparse" << std::endl;
  } else {
    std::cout << "dataset is dense" << std::endl;
  }
  while (1) {
    algorithmOption(algorithmType);
    if (algorithmType) {
      int ret;
      ret = SPARSE ? LogisticEntrance(algorithmType, datasetNum, XtS, y,
                                      XtTestS, yTest)
                   : LogisticEntrance(algorithmType, datasetNum, Xt, y, XtTest,
                                      yTest);
      if (ret) break;
    } else {
      break;
    }
  }
  return 0;
}
