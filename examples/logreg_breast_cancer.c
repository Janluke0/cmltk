#include <stdlib.h>
#include <stdio.h>
#include "../include/logreg.h"

#undef M_ASSERTS 

int main(){
    matrix_t *X = M_load("datasets/breast_cancer.X.train.dat");
    matrix_t *Y = M_load("datasets/breast_cancer.y.train.dat");
    matrix_t *X_test = M_load("datasets/breast_cancer.X.test.dat");
    matrix_t *Y_test = M_load("datasets/breast_cancer.y.test.dat");
    
    printf("Breast cancer %ld features\n",X->cols);


    LOGREG_model_t *model= LOGREG_new(X->cols);
    LOGREG_train(model,0.001,X,Y,50000, 1e-4);
    
    float train_acc = 100*accurancy(model, X, Y);
    printf("Train accurancy:%.2f%%\n", train_acc);

    float test_acc = 100*accurancy(model, X_test, Y_test);
    printf("Test  accurancy:%.2f%%\n", test_acc);


    return 0;
}
