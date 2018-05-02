#include "include/LogisticGradient.h"

void LogisticGradient(VectorXd &wtilde, VectorXd &G, const MatrixXd &Xt, VectorXd &y){
    long i,j;
    int nVars,nSamples;
    double innerProd;
    nVars = Xt.rows();
    nSamples = Xt.cols();
    VectorXd tmpRes(nSamples);
    //clear G
    G = MatrixXd::Zero(nVars,1);
    if(!SPARSE){
        tmpRes = Xt.adjoint()*wtilde;
    }
    else{
        ;
    }
    for(i=0;i<nVars;++i){
        tmpRes(i) = (1.0/(1+exp(-tmpRes(i)))-y(i))/nSamples;
    }
    if(!SPARSE){
        G = Xt * tmpRes;
    }
    else{
        ;
    }
    return;
}
