# Meta-SR: A Magnification-Arbitrary Network for Super-Resolution && RDN model implementation
Super Resolution RDN and Meta-SR implementation use tf.estimator

Code is running on tensorflow-gpu==1.12

# Step by Step tutorial
- 1.Download the DIV2K_valid_HR.zip DIV2K_train_HR.zip DIV2K_test_HR.zip
- 2.Set the train_filenames, test_filenames and eval_filenames in model.py
- 3.Set num_gpus=xxx in config.py to support multiple gpus training, run python model.py or model_new to train and evaluate RDN models
- 4.For training meta-SR model, just adjust the value in config.py D.model = 'meta-SR'
- 5.change D.mode = 'predict' in config.py to predict results
- 6.download the results folder and see the comparison pictures in personal computer.

# models
链接: https://pan.baidu.com/s/1kIG0vEdxgS6WhRs0cJlMSg 提取码: 8btp
This is the training result of RDN model of 3x SR. rename or put it in proper place according to config.py, finally predict you own pictures.


# 5.8 UPDATE
add a new implementation of meta-SR,(called batch_conv in basemodel.py which support convolution for different kernels values) support batch_size=16 and image_size=50 for multiple scale training. Speed up training speed. model_new.py suppprts batch_size = 16, model.py only support batch_size = 1 in meta-SR mode.

now training mode and some config are all in config.py
# 5.23 UPDATE
add a new implementation to avoid while_loop in both training and evaluation.
# 5.24 UPDATE
Now training and evaluation support x.x scales rather than interger scale
# 5.26 UPDATE
Now training and evaluation all works well in meta-SR. The limitation is because of weight_prediction fc module, when HxW is very large,we dont't have enough gpu memory to store it. So when predict, you need to change 1 large picture into several smaller picture, and predict for each smaller picture, finally concat those results back to finally large picture.

