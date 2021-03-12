library(tensorflow)
library(keras)
source('build_ResNet.R')
source('custom_residual_layer.R')
art <- reticulate::import("art")
tf$compat$v1$disable_eager_execution()
load('x_test.R')
load('y_test.R')

source_model <- build_ResNet(input_shape = dim(x_test)[2:5], num_classes = 2, 
                      optimizer = optimizer_nadam(lr=0.001), 
                      metrics = list("accuracy", tf$keras$metrics$AUC()))
source_model <- source_model %>% load_model_weights_hdf5('test_model_weights.h5')

model <- keras_model_sequential() %>%
  layer_conv_3d(filters = 64, kernel_size = c(7,7,7), strides = c(2,2,2), padding = 'same',
                use_bias = F, input_shape = dim(x_test)[2:5]) %>%
  layer_batch_normalization() %>%
  layer_activation_relu() %>%
  layer_max_pooling_3d(pool_size = c(3,3,3), strides = c(2,2,2), padding = 'same')

prev_filters <<- 64
filter_list <<- c(rep(64,3), rep(128, 8), rep(256,36), rep(512,3))
for(i in seq_along(filter_list)){
  i <<- i
  if(filter_list[i] == prev_filters){strides <<- 1} else {strides <<- 2}
  model %>% residual_layer(filters = filter_list[i], strides = strides)
  prev_filters <<- filter_list[i]
}

model <- model %>%
  layer_global_average_pooling_3d() %>%
  layer_flatten() %>%
  layer_dropout(0.4) %>%
  layer_dense(units = 2, activation = "softmax")

model <- model %>% compile(loss="binary_crossentropy",
                           optimizer = optimizer_adam(0.001),
                           metrics = "accuracy")

for(w in 1:(length(model$layers)-1)){
  source_weights <- source_model$layers[[w]]$get_weights()
  model$layers[[w]]$set_weights(source_weights)
}

classifier <- art$estimators$classification$KerasClassifier(model=model)
all_scores <- list()
for(i in 1:dim(x_test)[1]){
  x_single_sample <- x_test[i,,,,]
  dim(x_single_sample) <- c(dim(x_single_sample), 1)
  clever_score <- art$metrics$clever_u(classifier, x_single_sample, 
                                       nb_batches = 50L, batch_size = 6L, 
                                       radius = 10L, norm = 2)
  all_scores <- append(all_scores, clever_score)
  print(i)
}
