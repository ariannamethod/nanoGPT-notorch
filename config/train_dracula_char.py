# Train a character-level model on Dracula with Chuck Optimizer
# Enough Shakespeare. Time for vampires. Now at Karpathy scale.

out_dir = 'weights'
eval_interval = 500
eval_iters = 50
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanoGPT-notorch'
wandb_run_name = 'dracula-chuck-10M'

dataset = 'data_dracula'
gradient_accumulation_steps = 2
batch_size = 16
block_size = 256

# Model — ~10.8M params, Karpathy-scale proof of concept
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

learning_rate = 6e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 6e-5
beta2 = 0.99

warmup_iters = 200

device = 'cpu'
compile = False
dtype = 'float32'
