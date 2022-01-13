/**
 * @file layer.h
 * @brief basic layer struct and operations
 * 
 */
#ifndef LAYER_H
#define LAYER_H

#define LAYER_NAME_MAX_LENGTH 64    //!< max length of layer name

/**
 * @struct 
 * @brief basic layer structure
 * 
 */
typedef struct Layer_tag
{
    char name[LAYER_NAME_MAX_LENGTH];   //!< layer name

    float *x;   //!< layer input matrix
    int in;     //!< num of layer input
    int in_h;   //!< input height
    int in_w;   //!< input width

    float *y;   //!< layer output matrix
    int out;    //!< num of layer output
    int out_h;  //!< output height
    int out_w;  //!< output width

    struct Layer_tag *prev; //!< pointer to previous layer
    struct Layer_tag *next; //!< pointer to next layer

    void (*forward)(struct Layer_tag*); //!< forward propagation
} Layer;

/**
 * @brief layer parameter structure
 * 
 */
typedef struct
{
    char name[LAYER_NAME_MAX_LENGTH];   //!< layer name

    int in;     //!< num of layer input
    int in_h;   //!< height of layer output
    int in_w;   //!< width of layer input

    int out;    //!< num of layer output
} LayerParameter;

/**
 * @brief allocate layer
 * 
 * @return Layer* 
 */
Layer *layer_alloc(const LayerParameter layer_param);

/**
 * @brief deallocate layer
 * 
 */
void layer_free(Layer **layer);

#endif // LAYER_H
