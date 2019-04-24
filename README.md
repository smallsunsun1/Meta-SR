# Meta-SR: A Magnification-Arbitrary Network for Super-Resolution && RDN model implementation
Super Resolution RDN and Meta-SR implementation use tf.estimator

# Step bt Step tutorial
- 1.Download the DIV2K_valid_HR.zip DIV2K_train_HR.zip DIV2K_test_HR.zip
- 2.Set the train_filenames, test_filenames and eval_filenames in model.py
- 3.Set num_gpus=xxx in model.py to supporte multiple gpus training, run python model.py to train and evaluate RDN models
- 4.For training meta-SR model, just comment the Estimator use model_fn, and uncomment the Estimator use meta-SR-model_fn, then run python model.py to train the model.
- 5.change some code in model.py to predict results
- 6.download the results folder and see the comparison pictures in personal computer.

# support for meta-SR is now Avaliable.
# Caution! the training and inference for meta-SR is extremely slowly, the problem maybe the while_loop bring a lot of conv operation which cause the very limit speed! When training meta-SR just set batch_size=1 and image_size=32,if too large, the gpu memory is not enough!
