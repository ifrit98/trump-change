# library(tensorflow)
# library(keras)
library(R2deepR)
library(reticulate)


source_python('misc/R-load-model-lr-range-test.py')

params <- lr_range_test(model, dataset)

filenm <- 'params.txt'
write(params, filenm)