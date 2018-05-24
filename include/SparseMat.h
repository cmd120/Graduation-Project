#ifndef SPARSE_H
#define SPARSE_H

#include "Eigen/Sparse"
#include "LogisticError.h"
#include "comm.h"
///
/// Check if data in vector is sparse.
/// \return 1 if data is sparse else 0
///
int issparse(std::vector<double> &mat);
/// Initialize the OuterStarts array of given sparse matrix.
/// OuterStarts array stores for each column(resp. row) the index of the
/// non-zero in the values array and InnerIndices(stores the row(resp.
/// column) indices of the non-zeros) array. For Matlab programmer, OutStarts
/// array's is the counterpart of jr array.
///
void InitOuterStarts(const Eigen::SparseMatrix<double> &mat, int *outerStarts);
///
/// Set necessary environment before executing algorithm part, such as output
/// file, step size, batch size.
///
void algorithmInit(Eigen::SparseMatrix<double> &Xt, Eigen::VectorXd &w,
                   Eigen::SparseMatrix<double> &XtTest, Eigen::VectorXd &yTest,
                   double &lambda, double &eta, double &a, double &b,
                   double &gamma, int &maxIter, int &batchSize, int &passes,
                   int &maxRunTime, std::string &filename, int &datasetNum);
#endif