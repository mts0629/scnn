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
#define TRAIN_DATA_NUM 60000
#define TEST_DATA_NUM 10000

// size (width/height) of data
#define DATA_SIZE 28

// the number of classes
#define CLASS_NUM 10

// load MNIST label data: "***-labels-idx1-ubyte"
float **load_mnist_labels(const char *filename, float **labels, const int num)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "failed to open file: %s\n", filename);
    }

    // check magic number
    int32_t magic_number = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        magic_number = (magic_number << 8) | byte;
    }
    if (magic_number != 2049) {
        fprintf(stderr, "magic number %d mismatch to 2049d\n", magic_number);
        goto FCLOSE;
    }

    // check the number of data
    int32_t num_data = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        num_data = (num_data << 8) | byte;
    }
    if (num_data != num) {
        fprintf(stderr, "the number of items %d mismatch to %d\n", num_data, num);
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

    return labels;

FCLOSE:
    fclose(fp);

    return NULL;
}

// load MNIST image data: "***-images-idx3-ubyte"
float **load_mnist_images(const char *filename, float **images, const int num)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "failed to open file: %s\n", filename);
        goto FCLOSE;
    }

    // check magic number
    int32_t magic_number = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        magic_number = (magic_number << 8) | byte;
    }
    if (magic_number != 2051) {
        fprintf(stderr, "magic number %d mismatch to 2051\n", magic_number);
        goto FCLOSE;
    }

    // check the number of data
    int32_t num_data = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        num_data = (num_data << 8) | byte;
    }
    if (num_data != num) {
        fprintf(stderr, "the number of items %d mismatch to %d\n", num_data, num);
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
        fprintf(stderr, "number of rows %d mismatch to 28\n", num_rows);
        goto FCLOSE;
    }
    int32_t num_cols = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        num_cols = (num_cols << 8) | byte;
    }
    if (num_cols != DATA_SIZE) {
        fprintf(stderr, "number of columns %d mismatch to 28\n", num_cols);
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

    return images;

FCLOSE:
    fclose(fp);

    return NULL;
}

float **load_mnist_train_labels(const char *filename)
{
    // allocate memory for dataset
    float **train_labels = malloc(sizeof(float*) * TRAIN_DATA_NUM);
    if (train_labels == NULL) {
        return NULL;
    }
    for (int i = 0; i < TRAIN_DATA_NUM; i++) {
        train_labels[i] = malloc(sizeof(float) * CLASS_NUM);
        if (train_labels[i] == NULL) {
            return NULL;
        }
    }

    return load_mnist_labels(filename, train_labels, TRAIN_DATA_NUM);
}

float **load_mnist_train_images(const char *filename)
{
    // allocate memory for dataset
    float **train_images = malloc(sizeof(float*) * TRAIN_DATA_NUM);
    if (train_images == NULL) {
        return NULL;
    }
    for (int i = 0; i < TRAIN_DATA_NUM; i++) {
        train_images[i] = malloc(sizeof(float) * DATA_SIZE * DATA_SIZE);
        if (train_images[i] == NULL) {
            return NULL;
        }
    }

    return load_mnist_images(filename, train_images, TRAIN_DATA_NUM);
}

float **load_mnist_test_labels(const char *filename)
{
    // allocate memory for dataset
    float **test_labels  = malloc(sizeof(float*) * TEST_DATA_NUM);
    if (test_labels == NULL) {
        return NULL;
    }
    for (int i = 0; i < TEST_DATA_NUM; i++) {
        test_labels[i] = malloc(sizeof(float) * CLASS_NUM);
        if (test_labels[i] == NULL) {
            return NULL;
        }
    }

    return load_mnist_labels(filename, test_labels, TEST_DATA_NUM);
}

float **load_mnist_test_images(const char *filename)
{
    // allocate memory for dataset
    float **test_images  = malloc(sizeof(float*) * TEST_DATA_NUM);
    if (test_images == NULL) {
        return NULL;
    }

    for (int i = 0; i < TEST_DATA_NUM; i++) {
        test_images[i] = malloc(sizeof(float) * DATA_SIZE * DATA_SIZE);
        if (test_images[i] == NULL) {
            return NULL;
        }
    }

    return load_mnist_images(filename, test_images, TEST_DATA_NUM);
}

int main(int argc, char *argv[])
{
    // check arguments: path to the MNIST data files
    if (argc < 5) {
        fprintf(stderr, "arguments are required: \n    path to MNIST training label data\n    path to MNIST training image data\n    path to MNIST test label data\n    path to MNIST test image data");
        exit(EXIT_FAILURE);
    }

    // load dataset to allocated memory
    float **train_labels = load_mnist_train_labels(argv[1]);
    float **train_images = load_mnist_train_images(argv[2]);
    float **test_labels  = load_mnist_test_labels(argv[3]);
    float **test_images  = load_mnist_test_images(argv[4]);
    if ((train_labels == NULL) || (train_images == NULL) ||
        (test_labels == NULL)  || (test_images == NULL)) {
        goto FREE_MEMORY;
    }

    // create network
    Net *net = net_create(
        5,
        (Layer*[]){
            fc_layer(SET_PARAM( .in=28*28, .out=100 )),
            sigmoid_layer(SET_PARAM( .in=100 )),
            fc_layer(SET_PARAM( .in=100, .out=10 )),
            sigmoid_layer(SET_PARAM( .in=10 )),
            softmax_layer(SET_PARAM( .in=10 ))
        }
    );

    rand_seed(0);

    net_init_layer_params(net);

    // training
    printf("start training ...\n");

    train_sgd(
        net,
        train_images,
        train_labels,
        test_images,
        test_labels,
        0.1,
        20,
        TRAIN_DATA_NUM,
        TEST_DATA_NUM,
        cross_entropy_error
    );

    printf("finished\n");

    net_free(&net);

FREE_MEMORY:
    // free memory
    for (int i = 0; i < TRAIN_DATA_NUM; i++) {
        FREE_WITH_NULL(&train_labels[i]);
        FREE_WITH_NULL(&train_images[i]);
    }
    for (int i = 0; i < TEST_DATA_NUM; i++) {
        FREE_WITH_NULL(&test_labels[i]);
        FREE_WITH_NULL(&test_images[i]);
    }

    FREE_WITH_NULL(&train_labels);
    FREE_WITH_NULL(&train_images);
    FREE_WITH_NULL(&test_labels);
    FREE_WITH_NULL(&test_images);

    return EXIT_SUCCESS;
}
