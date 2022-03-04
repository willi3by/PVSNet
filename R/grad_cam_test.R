library(neurobase)
tf_vis <- reticulate::import("tf_keras_vis")
load('/Users/willi3by/Desktop/ASNR_Abstract/master_PVS_dataset.R')
test_set_orig <- lapply(master_pvs_dataset, function(x){x <- x[-train_ind]})

model_modifier <- function(m){
  
  reticulate::py_set_attr(m$layers[[length(m$layers)]], "activation", tf$keras$activations$linear)
  
}

loss = function(output) { tf$keras$backend$mean(output)}


for(i in 1:length(test_set_orig$x)){
  test_sample <- x_test[i,,,,]
  dim(test_sample) <- c(1, dim(test_sample), 1)
  gradcam <- tf_vis$gradcam$GradcamPlusPlus(model, model_modifier, clone=F)
  cam <- gradcam(loss, test_sample, penultimate_layer=-1L)
  cam <- tf_vis$utils$normalize(cam)
  cam_arr <- array(0, dim=dim(test_set_orig$x[[i]])[1:3])
  cam_arr[,,c(10:25)] <- cam*10
  ref_img <- readNIfTI('/Users/willi3by/Desktop/ASNR_Abstract/data_allin_ss/460000202_T2w_allin_ss.nii')
  cam_nim <- nifti(cam_arr)
  cam_nim <- copyNIfTIHeader(img = ref_img, arr = cam_nim)
  writeNIfTI(cam_nim, filename = paste0(i, '_cam'))
  orig <- test_set_orig$x[[i]]
  dim(orig) <- c(dim(orig)[1:3])
  orig_nim <- nifti(orig)
  orig_nim <- copyNIfTIHeader(img = ref_img, arr = orig_nim)
  writeNIfTI(orig_nim, filename = paste0(i, "_orig"))
}
