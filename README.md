# RDN
Super Resolution RDN implementation use tf.estimator

# Step bt Step tutorial
- 1.Download the DIV2K_valid_HR.zip DIV2K_train_HR.zip DIV2K_test_HR.zip
- 2.Set the train_filenames, test_filenames and eval_filenames in model.py
- 3.Set num_gpus=xxx in model.py to supporte multiple gpus training, run python model.py to train and evaluate RDN models
- 4.change some code in model.py to predict results
- 5.download the results folder and see the comparison pictures in personal computer.

# support for meta-SR is not supported now.
