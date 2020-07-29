#include "../include/matrix.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef M_ASSERTS
#include <assert.h>
#endif
/*
 * TODO:MAKE UNIT TEST
void main(){
    matrix_t *A = M_rand(10,20,-2,2), 
             *B = M_rand(20,10,-3,2),
             *C,*D;
    C = M_dot(A,B);
    M_print(C);
    M_store(C,"test.dat");
    D = M_load("test.dat");
    M_print(D);
    M_print(A);
    M_print(C);

    C = M_mul(B,3.3);
    puts("mult by constant");
    M_free(B);
    B = C;
    for(int i=0;i<10;i++){
        C = M_dot(A,B);
        M_free(C);
    }
    puts("dot product x10");
    D = M_mul(M_dot(C,C),1./2.);
    puts("mult by constat and dot product");
    D = M_sum(D,D);
    puts("sum");
    for(int i=0;i<1000;i++){
        C = M_transpose(A);
        M_free(C);
    }
    puts("transposex10");
    //M_print(D);
}
*/
void M_free(matrix_t *m) {
        free(m->data);
        free(m);
}

matrix_t *M_load(char *f_path){
    FILE *infile;
    matrix_t *res = M_empty();
    infile = fopen(f_path, "r");
    if(infile == NULL){
        puts("Error opening the file");
        return NULL;
    }
    if(!fread(res,sizeof(size_t),2,infile)){
        puts("Error reading the file");
        fclose(infile);
        return NULL;
    }
    res->data = malloc(sizeof(m_element_t)*res->cols*res->rows);
    if(!fread(res->data,sizeof(m_element_t),res->cols*res->rows,infile)){
        puts("Error reading the file");
        fclose(infile);
        return NULL;
    }
    fclose(infile);
    return res;
}

int M_store(matrix_t *m,char *f_path){
    FILE *outfile;
    outfile = fopen(f_path, "w");
    if(outfile == NULL){
        puts("Error opening the file");
        return -1;
    }
    if(!fwrite(m,sizeof(size_t),2,outfile)){
        puts("Error writing the file");
        fclose(outfile);
        return -1;
    }    
    if(!fwrite(m->data,sizeof(m_element_t),m->cols*m->rows,outfile)){
        puts("Error writing the file");
        fclose(outfile);
        return -1;
    }

    fclose(outfile);

    return 0;
}

matrix_t *M_dot_T(matrix_t *a, matrix_t *b){
    //assert(a->rows==b->rows);
    matrix_t *res = M_new(a->cols, b->cols);
    size_t i,j,k;
    m_element_t sum = 0.0;
    #pragma omp parallel for private(i,j,k,sum) 
    for(i=0;i<res->rows;i++)
        for(j=0;j<res->cols;j++){
            for(k=0;k<b->rows;k++) 
                sum +=  M_get(a,k,i)* M_get(b,k,j);                
               
            M_get(res,i,j) = sum;
            sum = 0.;
        }
    return res;

}
matrix_t *M_dot(matrix_t *a, matrix_t *b){
#ifdef M_ASSERTS
    assert(a->cols==b->rows);
#endif
    matrix_t *res = M_new(a->rows, b->cols);
    size_t i,j,k;
    m_element_t sum = 0.0;
    #pragma omp parallel for private(i,j,k,sum) 
    for(i=0;i<res->rows;i++)
        for(j=0;j<res->cols;j++){
            for(k=0;k<b->rows;k++) 
                sum +=  M_get(a,i,k)* M_get(b,k,j);                
               
            M_get(res,i,j) = sum;
            sum = 0.;
        }
    return res;

}

matrix_t *M_new(size_t rows, size_t cols){
    matrix_t *out = M_empty();
    out->data = malloc(rows*cols*sizeof(m_element_t*));
    out->rows = rows;
    out->cols = cols;
    return out;
}

matrix_t *M_zeros(size_t rows, size_t cols){
    matrix_t *out = M_empty();
    out->data = calloc(rows*cols,sizeof(m_element_t));
    out->rows = rows;
    out->cols = cols;
    return out;
}
matrix_t *M_ones(size_t rows, size_t cols){
    matrix_t *out = M_new(rows,cols);
    for(size_t i=0;i<out->cols*out->rows;i++)
        out->data[i]=1;
    return out;
}


matrix_t *M_rand(size_t rows, size_t cols, m_element_t min, m_element_t max){
    matrix_t *res = M_new(rows, cols);
    for(size_t i=0;i< res->rows*res->cols; i++)
        res->data[i] = min + (m_element_t)rand()/(m_element_t)(RAND_MAX/max);
    return res;
}

matrix_t *M_copy(matrix_t *src){
    matrix_t *dst = M_new(src->rows,src->cols);
    memcpy(dst->data,src->data,src->cols*src->rows);
    return dst;
}

matrix_t *M_mul_scalar(matrix_t *a, m_element_t b){
    matrix_t *res = M_new(a->rows, a->cols);
    for(size_t i=0;i<a->rows*a->cols;i++)
        res->data[i] = a->data[i]*b;
    return res;
}

matrix_t *M_sum_scalar(matrix_t *a, m_element_t b){
    matrix_t *res = M_new(a->rows, a->cols);
    for(size_t i=0;i<a->rows*a->cols;i++)
        res->data[i] = a->data[i]+b;
    return res;
}

matrix_t *M_sum(matrix_t *a, matrix_t *b){
#ifdef M_ASSERTS
    assert(a->cols==b->cols);
    assert(a->rows==b->rows);
#endif
    matrix_t *res = M_new(a->rows, a->cols);
    for(size_t i=0;i<res->cols*res->rows;i++)
        res->data[i] = a->data[i] + b->data[i];
    return res;
}

matrix_t *M_sub(matrix_t *a, matrix_t *b){
#ifdef M_ASSERTS
    assert(a->cols==b->cols);
    assert(a->rows==b->rows);
#endif
    matrix_t *res = M_new(a->rows, a->cols);
    for(size_t i=0;i<res->cols*res->rows;i++)
        res->data[i] = a->data[i] - b->data[i];
    return res;
}

matrix_t *M_transpose(matrix_t *a){
    size_t i,j;
    matrix_t *res = M_new(a->cols,a->rows);
    for(i=0;i<res->rows;i++)
        for(j=0;j<res->cols;j++)
           M_get(res,i,j) = M_get(a,j,i);
    return res; 
}

matrix_t *M_identity(size_t n){
    matrix_t *res = M_zeros(n,n);
    for(size_t i=0;i<n;i++)
        M_get(res,i,i) = 1;
    return res;
}



void M_print(matrix_t *m){
    size_t i,j;
    for(i=0;i<m->rows;i++){
        for(j=0;j<m->cols;j++)
                printf("%f ",M_get(m,i,j));
        printf("\n");

    }
}
