#' Builds ResNet 151
#'
#' @param input_shape
#'
#' @return ResNet model
#' @export
#'
#' @examples
#' resnet <- build_ResNet(input_shape, num_classes, loss, optimizer, metrics = c("accuracy"))
build_ResNet <- function(input_shape, num_classes, loss = "binary_crossentropy", optimizer = optimizer_adam(lr=0.001), metrics){
  if(num_classes < 3){
    num_units = 1
    activation = "sigmoid"
  }
  else {
    num_units = num_classes
    activation = "softmax"
    }

  model <- keras_model_sequential() %>%
    layer_conv_3d(filters = 64, kernel_size = c(7,7,7), strides = c(2,2,2), padding = 'same',
                  use_bias = F, input_shape = input_shape) %>%
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
    layer_dense(units = num_units, activation = activation)

  model <- model %>% compile(loss="binary_crossentropy",
                    optimizer = optimizer,
                    metrics = metrics)

  return(model)
}
