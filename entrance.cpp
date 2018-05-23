#include "include/Data.h"
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
LR objFuncLR;
RR objFuncRR;
// available datasets
const std::vector<std::string> datasetNameList = {
    "MNIST",   "COVTYPE",      "A1A",     "AUSTRALIAN",
    "COD-RNA", "COLON-CANCER", "DIABETS", "BREAST-CANCER"};
// supported algorithms
const std::vector<std::string> algorithmNameList = {
    "IAG", "IAGA", "SAG", "SAGA", "SGD", "SIG", "SVRG"};
int LogisticEntrance(int algorithmType, int datasetNum,
                     Eigen::SparseMatrix<double> &XtS, Eigen::VectorXd &y,
                     Eigen::SparseMatrix<double> &XtTestS,
                     Eigen::VectorXd &yTest) {
  Eigen::VectorXd w, wtilde, G, sumIG, gradients;
  double lambda, eta, a, b, gamma;
  int maxIter, batchSize, passes, maxRunTime;
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
  std::cout << "Available datasets to choose from:" << std::endl;
  for (int i = 0; i < datasetNameList.size(); ++i) {
    std::cout << i << "." << datasetNameList[i] << std::endl;
  }
  std::cout << "Enter your choice of dataset: " << std::endl;
  std::cin >> datasetNum;
  std::cout << "Your choice of dataset: " << datasetNameList[datasetNum]
            << std::endl;
  return;
}
void algorithmOption(int &algorithmType) {
  std::cout << "Enter your choice of algorithm: (0 to quit)" << std::endl;
  for (int i = 0; i < algorithmNameList.size(); ++i) {
    std::cout << i + 1 << "." << algorithmNameList[i] << std::endl;
  }
  while (1) {
    if (std::cin >> algorithmType) {
      if (algorithmType)
        std::cout << "Your choice of algorithm: "
                  << algorithmNameList[algorithmType - 1] << std::endl;
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
  int algorithmType = 0, datasetNum, features, trainSize = 0, testSize = 0,
      sparseFormat = 1;
  std::string trainfile, testfile;
  datasetOption(datasetNum);
  DATASET data;
  switch (datasetNum) {
    case 0:
      /*Custom dataset example*/
      mnist_read(Xt, y, XtTest, yTest);
      break;
    case 1:
      /*If dataset file is dense format, use DenseFormatRead to read*/
      /*If dataset file is sparse format, use SparseFormatRead to read*/
      // Covetype
      // Source: UCI / Covertype
      // # of classes: 2
      // # of data: 581,012
      // # of features: 54
      data = DATASET(datasetNameList[datasetNum], 54, "covtype.libsvm.binary",
                     "covtype.libsvm.binary", 2000, 100);
      break;
    case 2:
      // a1a
      // Source: UCI / Adult
      // # of classes: 2
      // # of data: 1,605 / 30,956 (testing)
      // # of features: 123 / 123 (testing)
      data = DATASET(datasetNameList[datasetNum], 123, "a1a", "a1a.t", 1605,
                     30956);
      break;
    case 3:
      // australian
      // Source: Statlog / Australian
      // # of classes: 2
      // # of data: 690
      // # of features: 14
      data = DATASET(datasetNameList[datasetNum], 14, "australian",
                     "australian", 690, 690);
      break;
    case 4:
      // cod-rna
      // Source: [AVU06a]
      // # of classes: 2
      // # of data: 59,535 / 271617 (validation) / 157413 (unused/remaining)
      // # of features: 8
      data = DATASET(datasetNameList[datasetNum], 8, "cod-rna", "cod-rna.t",
                     59535, 271617);
      break;
    case 5:
      // colon-cancer
      // Source: [AU99a]
      // # of classes: 2
      // # of data: 62
      // # of features: 2,000
      data = DATASET(datasetNameList[datasetNum], 2000, "colon-cancer",
                     "colon-cancer", 62, 62);
      break;
    case 6:
      // diabets
      // UCI / Pima Indians Diabetes
      // # of classes: 2
      // # of data: 768
      // # of features: 8
      data = DATASET(datasetNameList[datasetNum], 8, "diabetes", "diabetes",
                     768, 768);
      break;
    case 7:
      // duke breast-cancer
      // Source: [MW01a]
      // # of classes: 2
      // # of data: 44
      // # of features: 7,129
      data = DATASET(datasetNameList[datasetNum], 7129, "duke.tr", "duke.val",
                     38, 4);
      break;
    default:
      std::cout << "Input Invalid." << std::endl;
  }
  // datasetNum = 0 is mnist dataset
  if (datasetNum && sparseFormat) {
    SparseFormatRead(XtS, y, data.getfeatures(), data.gettrainsize(),
                     data.gettrainfilename());
    SparseFormatRead(XtTestS, yTest, data.getfeatures(), data.gettestsize(),
                     data.gettestfilename());
  } else if (datasetNum && !sparseFormat) {
    DenseFormatRead(Xt, y, data.getfeatures(), data.gettrainsize(),
                    data.gettrainfilename());
    DenseFormatRead(Xt, y, data.getfeatures(), data.gettestsize(),
                    data.gettestfilename());
    if (SPARSE) {
      XtS = Xt.sparseView();
      XtTestS = Xt.sparseView();
    }
  }
  SPARSE ? printf("XtS rows:%d XtS cols:%d XtTestS rows:%d XtTestS cols:%d\n",
                  XtS.rows(), XtS.cols(), XtTestS.rows(), XtTestS.rows())
         : printf("Xt rows:%d Xt cols:%d XtTest rows:%d XtTest cols:%d\n",
                  Xt.rows(), Xt.cols(), XtTest.rows(), XtTest.rows());
  printf("y size:%d yTest size:%d", y.size(), yTest.size());
  std::cout << "dataset loaded." << std::endl;
  if (SPARSE) {
    std::cout << "dataset is sparse" << std::endl;
  } else {
    std::cout << "dataset is dense" << std::endl;
  }
  while (1) {
    algorithmOption(algorithmType);
    if (algorithmType) {
      int ret;
      // std::cout << "SPARSE(in while):" << SPARSE << std::endl;
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
