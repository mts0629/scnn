/**
 * @file layer.h
 * @brief basic layer structure and operations
 * 
 */
#ifndef LAYER_H
#define LAYER_H

#define N_DIM 4 //!< num of data dimensions, fixed to 4 for CNN

/**
 * @struct 
 * @brief basic layer structure
 * 
 */
typedef struct Layer {
    const float *x;     //!< layer input matrix
    int x_dim[N_DIM];   //!< dimension of x
    int x_size;         //!< num of elements of x

    float *y;           //!< layer output matrix
    int y_dim[N_DIM];   //!< dimension of y
    int y_size;         //!< num of elements of y

    float *w;           //!< layer weight
    int w_dim[N_DIM];   //!< dimension of w
    int w_size;         //!< num of elements of w

    float *b;           //!< layer bias
    int b_dim[N_DIM];   //!< dimension of b
    int b_size;         //!< num of elements of b

    float *dx;  //!< differential of x
    float *dw;  //!< differential of w
    float *db;  //!< differential of b

    struct Layer *prev; //!< pointer to previous layer
    struct Layer *next; //!< pointer to next layer

    void (*forward)(struct Layer *self, const float *x);     //!< forward propagation
    void (*backward)(struct Layer *self, const float *dy);   //!< backward propagation

    void (*update)(struct Layer *self, const float learning_rate);  //!< parameter updating
} Layer;

/**
 * @brief layer parameter structure
 * 
 */
typedef struct LayerParameter {
    int in;     //!< num of layer input
    int out;    //!< num of layer output
} LayerParameter;

/**
 * @brief allocate basic layer structure
 * 
 * @return Layer* poiner to layer structure
 */
Layer *layer_alloc(void);

/**
 * @brief deallocate layer structure
 * 
 * @param[in,out] layer layer structure to be deallocated
 */
void layer_free(Layer **layer);

#endif // LAYER_H
