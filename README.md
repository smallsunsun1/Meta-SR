# Meta-SR: A Magnification-Arbitrary Network for Super-Resolution && RDN model implementation
Super Resolution RDN and Meta-SR implementation use tf.estimator

Code is running on tensorflow-gpu==1.12

# Step by Step tutorial
- 1.Download the DIV2K_valid_HR.zip DIV2K_train_HR.zip DIV2K_test_HR.zip
- 2.Set the train_filenames, test_filenames and eval_filenames in model.py
- 3.Set num_gpus=xxx in model.py to support multiple gpus training, run python model.py to train and evaluate RDN models
- 4.For training meta-SR model, just comment the Estimator use model_fn, and uncomment the Estimator use meta-SR-model_fn, then run python model.py to train the model.
- 5.change some code in model.py to predict results
- 6.download the results folder and see the comparison pictures in personal computer.

Note: batch_size is 1 for multiple scale training for meta-SR


# 5.8 UPDATE
# add a new implementation of meta-SR, support batch_size=10 and image_size=50, speed up training speed, but data_input_fn need to replace py_func function with truely tensorflow code to accerate the training speed more.

now training mode and some config are all in config.py
