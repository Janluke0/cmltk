#ifndef _LINEAR_SVM_H
#define _LINEAR_SVM_H_
#include "matrix.h"
#include "logreg.h"

/* like logistic regression the model is composed 
 * by a vector of weights and a bias */
typedef LOGREG_model_t LSVM_model_t;

LSVM_model_t* LSVM_new(size_t features){
    return LOGREG_new(features);
}

//with L2 regularization
int     LSVM_iteration    (LSVM_model_t *m,
                            float lr, 
                            float lambda,   //lambda factor for L2 regularization
                            matrix_t *X,    //observation matrix (m,n) 
                            matrix_t *y);   //labels vector (m,1)

matrix_t*   LSVM_inference    (LSVM_model_t *m, 
                                matrix_t *X);

//return number of perfomed iteration. If equal to max_it model didn't converge
//learning rate should be adaptative. TODO: find a good strategy for this
//convergence is reached when loss improvement is minor than loss_tol
int     LSVM_train    (LSVM_model_t *model, // the result of training
                        float lambda,   //lambda factor for L2 regularization
                        matrix_t *X,    //observation matrix (m,n) 
                        matrix_t *y,    //labels vector (m,1)
                        int max_it, 
                        double loss_tol);
#endif
