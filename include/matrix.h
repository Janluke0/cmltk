#ifndef _MATRIX_H_
#define _MATRIX_H_
#include <stdlib.h>
#define M_get(matrix,i,j) matrix->data[(i*matrix->cols)+j]
#define M_empty() malloc(sizeof(matrix_t))
#define M_free(matrix) free(matrix->data);free(matrix)

#define M_ASSERTS 

typedef double m_element_t;
typedef m_element_t  (*m_fn)(m_element_t);
typedef struct matrix_t{
        size_t  rows;
        size_t  cols;
        m_element_t *data;
} matrix_t;

matrix_t*   M_load  (char *f_path);

int         M_store (matrix_t *m,
                    char *f_path);

matrix_t*   M_copy  (matrix_t *src);

matrix_t*   M_new   (size_t rows, 
                    size_t cols);


matrix_t*   M_ones  (size_t rows, 
                    size_t cols);

matrix_t*   M_zeros (size_t rows, 
                    size_t cols);

matrix_t*   M_identity  (size_t n);
matrix_t*   M_rand  (size_t rows, 
                    size_t cols, 
                    m_element_t min, 
                    m_element_t max);

//A*B
matrix_t*   M_dot   (matrix_t *a, matrix_t *b);
//A^T*B 
matrix_t*   M_dot_T (matrix_t *a, matrix_t *b);

matrix_t*   M_transpose   (matrix_t *a);

matrix_t*   M_invert  (matrix_t *a);


matrix_t*   M_sum   (matrix_t *a, matrix_t *b);

matrix_t*   M_sub   (matrix_t *a, matrix_t *b);

matrix_t*   M_mul_scalar  (matrix_t *a, m_element_t b);

matrix_t*   M_sum_scalar  (matrix_t *a, m_element_t b);


void    M_print (matrix_t *m);
#endif
