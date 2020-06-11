#include "../include/logreg.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#undef M_ASSERTS 

m_element_t _array_mean(matrix_t *arr);



m_element_t _array_mean(matrix_t *arr){
    m_element_t v = 0;
    size_t i, len = arr->rows;
    for(i=0;i<len;i++)
        v += arr->data[i]; 
    return v/len;
}

int LOGREG_iteration(LOGREG_model_t *m, float lr, float lambda, matrix_t *X, matrix_t *Y){
    matrix_t *P = LOGREG_inference(m,X),
             *P_less_y = M_sub(P,Y); 
    
    m_element_t grad_b = _array_mean(P_less_y);
    
    //grad_w = (X.T @ (P - Y)) / m
    matrix_t *m_prod = M_dot_T(X,P_less_y),//X.T@(P-Y)
             *grad_w_tmp = M_mul_scalar(m_prod, 1./((m_element_t)X->rows)),// /m
             *t1 = M_mul_scalar(m->weights, 2*lambda),//L2 regularization
             *grad_w = M_sum(grad_w_tmp,t1);
    
    m->bias -= lr*grad_b;
    // this is more coerent but a lot slower i think, with inplace results would be better
    // t1 = M_mul_scalar(grad_w,lr)
    // t2  = M_sub(m->weights,t1)
    // M_free(t1); M_free(m->weights);
    // m->weights = t2
    for(size_t i=0;i<m->m;i++)
        m->weights->data[i] -= grad_w->data[i]*lr;

    M_free(P_less_y);
    M_free(m_prod);   
    M_free(P);
    M_free(grad_w_tmp);
    M_free(grad_w);
    M_free(t1);
    return 0;
}

matrix_t *LOGREG_inference(LOGREG_model_t *m, matrix_t *X){
    matrix_t *Z1 = M_dot(X, m->weights),
             *Z  = M_sum_scalar(Z1, m->bias);
    M_free(Z1);
    size_t len = Z->cols*Z->rows;
    //sigmoid
    #pragma omp parallel for
    for(size_t i=0;i<len;i++)
        Z->data[i] = 1.0/(1.0 + exp(-Z->data[i]));
    return Z;
}
#define T 100.
int  LOGREG_train(LOGREG_model_t *model, float lambda, matrix_t *X, 
                            matrix_t *y, int max_it, double loss_tol){
    float lr = 1., loss=0, last_loss=0;
    int it = 0;
    matrix_t *P; 
    do{
        for(int i=0;i<T;i++)
            LOGREG_iteration(model,lr,lambda,X,y);
        it += T;       
        //very stupid test with real dataset anyway. Not so bad but T must be tuned anyway
        lr =  lr/(1. + (it/T)) ;
        last_loss = loss;
        P = LOGREG_inference(model, X);
        loss = cross_entropy(P,y);
        M_free(P);
        printf("IT:%d\tLOSS:%.10f\tLR:%.10f\tIMPROVMENT:%.10f\n",it,loss,lr,fabs(last_loss-loss)/lr);
    }while(it < max_it && ( fabs(last_loss-loss)/lr > loss_tol ));
    
    return it;
}


LOGREG_model_t *LOGREG_new(size_t features){
    LOGREG_model_t *m = malloc(sizeof(LOGREG_model_t));
    m->m = features;
    m->weights = M_zeros(features,1); 
    return m;
}

m_element_t cross_entropy(matrix_t *P, matrix_t *Y){
    m_element_t res = 0.0;
    #pragma omp parallel for reduction(+:res)
    for(size_t i=0;i<P->rows;i++){
        m_element_t p = M_get(P,i,0);
        if(p< EPS)
            p = EPS;
        else if(p> 1-EPS)
            p = 1 - EPS;

        res += M_get(Y,i,0)*log(p) + (1-M_get(Y,i,0))*log(1-p);
    }
    return -res/P->rows;
}

m_element_t accurancy(LOGREG_model_t *model, matrix_t *X, matrix_t *y){
    matrix_t *P = LOGREG_inference(model,X);
    m_element_t mean = 0;
    size_t len = P->rows;
    for(size_t i=0;i<len;i++){
        mean += (M_get(P,i,1)>0.5) == (M_get(y,i,1)>0.5);
    }
    M_free(P);
    return mean/len;
}
