import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 20
  training.n_iters = 1300001 #一共train多少个iter
  training.snapshot_freq = 10000 #多少个iter保存一次快照
  training.log_freq = 50 
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000 #多少个iter保存一次快照（checkpoint-meta）
  ## produce samples at each snapshot.
  training.snapshot_sampling = True#保存快照时是否采样图片
  training.likelihood_weighting = False #计算loss是否使用likelihood weighting
  training.continuous = True #这个SDE是否是连续的
  training.reduce_mean = True #计算loss时是否reduce_mean
  ## Score FPE setups
  training.scalar_fp = 'both' #choices=['True', 'False', 'both']
  training.fp_wgt_type = 'constant' #choices=['constant', 'convention', 'll']
  training.alpha = 0.15
  training.beta = 0.01
  training.m = 2
  

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = False
  sampling.probability_flow = False
  sampling.snr = 0.16

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 19
  evaluate.end_ckpt = 19
  evaluate.batch_size = 256 #采样的batchsize
  evaluate.enable_sampling = False
  evaluate.num_samples = 20000 #一共采样多少张图片
  evaluate.enable_loss = False#True
  evaluate.enable_bpd = True #是否评估似然，计算NLL
  evaluate.bpd_dataset = 'test'#计算似然使用的数据集

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CIFAR10'
  data.image_size = 32
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config