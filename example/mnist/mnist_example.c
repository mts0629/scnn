#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "util.h"
#include "layers.h"
#include "trainer.h"
#include "loss.h"
#include "random.h"
#include "data.h"

// the number of data
#define DATA_NUM 60000

// size (width/height) of data
#define DATA_SIZE 28

// the number of classes
#define CLASS_NUM 10

// load MNIST label data: "train-labels-idx1-ubyte"
void load_mnist_train_labels(const char *filename, float **labels)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "failed to open file: %s\n", filename);
        goto FCLOSE;
    }

    // check magic number (2049)
    int32_t magic_number = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        magic_number = (magic_number << 8) | byte;
    }
    if (magic_number != 2049) {
        fprintf(stderr, "magic number 2049 mismatch: %d\n", magic_number);
        goto FCLOSE;
    }

    // check the number of data
    int32_t num_data = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        num_data = (num_data << 8) | byte;
    }
    if (num_data != DATA_NUM) {
        fprintf(stderr, "the number of items 60000 mismatch: %d\n", num_data);
        goto FCLOSE;
    }

    // get labels
    for (int i = 0; i < num_data; i++) {
        uint8_t label;
        fread(&label, sizeof(uint8_t), 1, fp);
        // convert as one-hot vector
        for (int j = 0; j < CLASS_NUM; j++) {
            labels[i][j] = (j == label) ? 1.0f : 0.0f;
        }
    }

FCLOSE:
    fclose(fp);
}

// load MNIST image data: "train-images-idx3-ubyte"
void load_mnist_train_images(const char *filename, float **images)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "failed to open file: %s\n", filename);
        goto FCLOSE;
        return;
    }

    // check magic number (2051)
    int32_t magic_number = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        magic_number = (magic_number << 8) | byte;
    }
    if (magic_number != 2051) {
        fprintf(stderr, "magic number 2051 mismatch: %d\n", magic_number);
        goto FCLOSE;
    }

    // check the number of data
    int32_t num_data = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        num_data = (num_data << 8) | byte;
    }
    if (num_data != DATA_NUM) {
        fprintf(stderr, "the number of items 60000 mismatch: %d\n", num_data);
        goto FCLOSE;
    }

    // get rows and cols (28x28)
    int32_t num_rows = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        num_rows = (num_rows << 8) | byte;
    }
    if (num_rows != DATA_SIZE) {
        fprintf(stderr, "number of rows '28' mismatch: %d\n", num_rows);
        goto FCLOSE;
    }
    int32_t num_cols = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        num_cols = (num_cols << 8) | byte;
    }
    if (num_cols != DATA_SIZE) {
        fprintf(stderr, "number of columns '28' mismatch: %d\n", num_cols);
        goto FCLOSE;
    }

    // get pixels as vector
    for (int i = 0; i < num_data; i++) {
        for (int j = 0; j < (DATA_SIZE * DATA_SIZE); j++) {
            uint8_t pixel;
            fread(&pixel, sizeof(uint8_t), 1, fp);
            // normalize to [0,1]
            images[i][j] = (float)pixel / 255;
        }
    }

FCLOSE:
    fclose(fp);
}

int main(int argc, char *argv[])
{
    // check arguments
    // path to the MNIST data files, labels and training data
    if (argc < 3) {
        fprintf(stderr, "arguments are required: \n    path to MNIST label data (\"train-labels-idx1-ubyte\")\n    path to MNIST traing data (\"train-images-idx3-ubyte\")\n");
        exit(EXIT_FAILURE);
    }

    // allocate memory for dataset
    float **labels = malloc(sizeof(float*) * DATA_NUM);
    float **images = malloc(sizeof(float*) * DATA_NUM);
    for (int i = 0; i < DATA_NUM; i++) {
        labels[i] = malloc(sizeof(float) * CLASS_NUM);
        images[i] = malloc(sizeof(float) * DATA_SIZE * DATA_SIZE);
    }

    // load dataset
    load_mnist_train_labels(argv[1], labels);
    load_mnist_train_images(argv[2], images);

    // create network
    Net *net = net_create(
        5,
        (Layer*[]){
            fc_layer((LayerParameter){ .in=28*28, .out=100 }),
            sigmoid_layer((LayerParameter){ .in=100 }),
            fc_layer((LayerParameter){ .in=100, .out=10 }),
            sigmoid_layer((LayerParameter){ .in=10 }),
            softmax_layer((LayerParameter){ .in=10 })
        }
    );

    rand_seed(0);

    net_init_layer_params(net);

    // training
    printf("start training ...\n");

    train_sgd(
        net,
        images,
        labels,
        0.01,
        20,
        DATA_NUM,
        cross_entropy_error
    );

    printf("finished\n");

    net_free(&net);

    // free memory
    for (int i = 0; i < DATA_NUM; i++) {
        FREE_WITH_NULL(&labels[i]);
        FREE_WITH_NULL(&images[i]);
    }
    FREE_WITH_NULL(&labels);
    FREE_WITH_NULL(&images);

    return EXIT_SUCCESS;
}
