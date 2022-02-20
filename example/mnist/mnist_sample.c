#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "util.h"

// load MNIST label data: "train-labels-idx1-ubyte"
void load_mnist_train_labels(const char *filename, uint8_t *labels)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "failed to open file: %s\n", filename);
        goto FCLOSE;
        return;
    }

    int32_t magic_number = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        magic_number = (magic_number << 8) | byte;
    }
    if (magic_number != 2049) {
        fprintf(stderr, "magic number '2049' mismatch: %d\n", magic_number);
        goto FCLOSE;
    }

    int32_t num_items = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        num_items = (num_items << 8) | byte;
    }
    if (num_items != 60000) {
        fprintf(stderr, "number of items '60000' mismatch: %d\n", num_items);
        goto FCLOSE;
    }

    for (int i = 0; i < num_items; i++) {
        uint8_t label;
        fread(&label, sizeof(uint8_t), 1, fp);
        labels[i] = label;
    }

FCLOSE:
    fclose(fp);
}

// load MNIST image data: "train-images-idx3-ubyte"
void load_mnist_train_images(const char *filename, uint8_t **images)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "failed to open file: %s\n", filename);
        goto FCLOSE;
        return;
    }

    int32_t magic_number = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        magic_number = (magic_number << 8) | byte;
    }
    if (magic_number != 2051) {
        fprintf(stderr, "magic number '2051' mismatch: %d\n", magic_number);
        goto FCLOSE;
    }

    int32_t num_items = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        num_items = (num_items << 8) | byte;
    }
    if (num_items != 60000) {
        fprintf(stderr, "number of items '60000' mismatch: %d\n", num_items);
        goto FCLOSE;
    }

    int32_t num_rows = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        num_rows = (num_rows << 8) | byte;
    }
    if (num_rows != 28) {
        fprintf(stderr, "number of rows '28' mismatch: %d\n", num_rows);
        goto FCLOSE;
    }

    int32_t num_cols = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t byte;
        fread(&byte, sizeof(uint8_t), 1, fp);
        num_cols = (num_cols << 8) | byte;
    }
    if (num_cols != 28) {
        fprintf(stderr, "number of columns '28' mismatch: %d\n", num_cols);
        goto FCLOSE;
    }

    for (int i = 0; i < num_items; i++) {
        for (int j = 0; j < (28 * 28); j++) {
            uint8_t pixel;
            fread(&pixel, sizeof(uint8_t), 1, fp);
            images[i][j] = pixel;
        }
    }

FCLOSE:
    fclose(fp);
}

int main(void)
{
    // allocate memory for dataset
    uint8_t *labels = malloc(sizeof(uint8_t) * 60000);
    uint8_t **images = malloc(sizeof(uint8_t*) * 28 * 28);
    for (int i = 0; i < 60000; i++) {
        images[i] = malloc(sizeof(uint8_t) * 28 * 28);
    }

    // load dataset
    load_mnist_train_labels("./data/train-labels-idx1-ubyte", labels);
    load_mnist_train_images("./data/train-images-idx3-ubyte", images);

    FREE_WITH_NULL(&images);
    FREE_WITH_NULL(&labels);

    return 0;
}
