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
    float *y;   //!< layer output matrix

    struct Layer_tag *prev; //!< pointer to previous layer
    struct Layer_tag *next; //!< pointer to next layer

    void (*forward)(struct Layer_tag*); //!< forward propagation
} Layer;

/**
 * @brief allocate layer
 * 
 * @return Layer* 
 */
Layer *layer_alloc(const char*);

/**
 * @brief deallocate layer
 * 
 */
void layer_free(Layer**);

#endif // LAYER_H
