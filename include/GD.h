#include "comm.h"
#define DEBUG 1

void GD_logistic(Eigen::VectorXd &w, Eigen::MatrixXd &Xt, Eigen::VectorXd &y,
                 double lambda, double eta);