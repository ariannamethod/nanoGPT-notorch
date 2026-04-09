# Train a character-level model on Dracula with Chuck Optimizer
# Enough Shakespeare. Time for vampires.

out_dir = 'weights'
eval_interval = 250
eval_iters = 50
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanoGPT-notorch'
wandb_run_name = 'dracula-chuck'

dataset = 'data_dracula'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

# Model — ~2M params, serious prototype for Dracula
n_layer = 4
n_head = 4
n_embd = 208
dropout = 0.1

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

device = 'cpu'
compile = False
dtype = 'float32'
