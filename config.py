from easydict import EasyDict

D = EasyDict()
D.image_size = 30
D.num_gpus = 1
# D.c_dim = 3  # the size of channel, if use meta-SR, c_dim=64, else 3
D.c_dim = 64
D.scale = 3  # the size of scale factor for preprocessing input image,
# if use meta-SR, training_preocess split into several period, scale ranges in 2,3,4
D.batch_size = 16   # use batch_size = 16 or 32 for RDN training
D.D = 5  # if use meta-SR, D=16, else 5, however currently only support 5 for both modes
# D.D = 16
D.C = 3  # if use meta-SR, C=8, else 3, however currently only support 3 for both modes
# D.C = 8
D.G = 64
D.G0 = 64
D.kernel_size = 3
D.learning_rate = 1e-4
D.stride = 30
D.meta_sr_upsample_scale = 3
D.meta_sr_kernel_size = 3
D.meta_sr_c_dim = 3
D.model = 'meta-SR'   # RDN for train RDN model, else train meta-SR model
D.mode = 'train'  # train or predict the model, use predict to predict the model
D.model_dir = './models_metaSR_v3'
