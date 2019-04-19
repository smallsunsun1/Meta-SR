from easydict import EasyDict

D = EasyDict()
D.image_size = 100
# D.c_dim = 3  # the size of channel, if use meta-SR, c_dim=64, else 3
D.c_dim = 64
D.scale = 3  # the size of scale factor for preprocessing input image,
# if use meta-SR, training_preocess split into several period, scale ranges in 2,3,4
D.batch_size = 32
# D.D = 5  # if use meta-SR, D=16, else 5
D.D = 16
# D.C = 3  # if use meta-SR, C=8, else 3
D.C = 8
D.G = 64
D.G0 = 64
D.kernel_size = 3
D.learning_rate = 1e-4
D.stride = 50
D.meta_sr_upsample_scale = 3
D.meta_sr_kernel_size = 3
D.meta_sr_c_dim = 3
