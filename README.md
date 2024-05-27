# DIFFUSION BASED NEURAL NETWORK WEIGHTS GENERATION
Neural-network-parameters-with-Diffusion
# DoTO

# Structure of the code
Folder stage 1 contains code for the VAE and folder stage 2 contains code for the diffusion process. If using a set-transformer-based dataset encoder, folder clips contain dataset alignment code.
The dataset alignment required to trained the VAE first then used the frozen encoder to in the training process.

We provide an example of config. file
# Pretrained weights preprocessing
If using our data loader. we can either loader from dictionary {dataset 1: weights tensors, dataset2: weight tensor, ...} or load from checkpoint directly.
# VAE Training process:
Fill the config file according to your pretrained weights' length. there is no specific or standard setting for any architecture. user must choose the setting that works well for their pretrained weights.
assign correctly the dataset
to train run 
' python main.py'

During training, it is better to occasionally check the reconstruction weights' performance to reduce the training time.
# After VAE training:
save the VAE checkpoints to folder checkpoints/stage1/
If using a set transformer for dataset alignment, then extract the encoder checkpoint of the VAE and save it in checkpoints/clip_models/ then configure the config file in the clips folder correspondingly
to extract the enocder, loader the pretrained VAE and instantiate clipmodel from config with checkpoint path set to null:
' clipmodel.encoder = autoencoder.encoder
    clipmodel.quant_conv = autoencoder.quant_conv
    torch.save(clipmodel.state_dict(),'checkpoints/clip_models/weight_encoder_.ckpt')'
The dataset alignment training process is done by running the code in cliptrainer file.
'python cliptrainer.py'

this step can be skipped is using mlp dataset encoder or jointly optimizing the set-transformer or using a pretrained set-transformer.

# After dataset alignment
extract the dataset encoder checkpoint and save it in checkpoints/set-transformer (model.dataset_encoder.state_dict())

# Diffusion training process
extract the conditioning images using  'compute_condition.py'
configure the config file and run 
'python dtrainer.py'

# sampling
following mlp_sampling.py file

# DOTO 
The resnet18 config file correspond to the configuration with checkpoints obtained with neural network diffusion code.
## adding dummy experiments
