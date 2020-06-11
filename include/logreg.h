#ifndef _LOGREG_H_
#define _LOGREG_H_
#include "matrix.h"

#define EPS 1e-5
typedef struct LOGREG_model_t{
        int m;//features count equal to weights->nrows
        matrix_t *weights;
        m_element_t bias;
} LOGREG_model_t;

LOGREG_model_t*     LOGREG_new  (size_t features);

//with L2 regularization
int     LOGREG_iteration    (LOGREG_model_t *m,
                            float lr, 
                            float lambda,   //lambda factor for L2 regularization
                            matrix_t *X,    //observation matrix (m,n) 
                            matrix_t *y);   //labels vector (m,1)

matrix_t*   LOGREG_inference    (LOGREG_model_t *m, 
                                matrix_t *X);

//return number of perfomed iteration. If equal to max_it model didn't converge
//learning rate should be adaptative. TODO: find a good strategy for this
//convergence is reached when loss improvement is minor than loss_tol
int     LOGREG_train    (LOGREG_model_t *model, // the result of training
                        float lambda,   //lambda factor for L2 regularization
                        matrix_t *X,    //observation matrix (m,n) 
                        matrix_t *y,    //labels vector (m,1)
                        int max_it, 
                        double loss_tol);

m_element_t     cross_entropy   (matrix_t *P, //predictions 
                                matrix_t *Y); //exptected

m_element_t     accurancy   (LOGREG_model_t *model, matrix_t *X, matrix_t *y);
#endif
