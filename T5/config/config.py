from sacred import Experiment

EXPERIMENT_NAME = 'T5'

ex = Experiment(EXPERIMENT_NAME, save_git_info=False)

@ex.config
def config():
    exp_name = EXPERIMENT_NAME
    mode = 'train'
    seed = 42
    
    # Model Hyper-Parameter Setting
    # GPU, CPU Environment Setting
    num_nodes = 1
    gpus = [0]
    batch_size = 256
    per_gpu_batch_size = 32    # Note that -> batch_size % (per_gpu_batch_size * len(gpus) == 0
    num_workers = 4

    # Data Setting
    input_seq_len = 118
    label_seq_len = 49

    # Main Setting
    max_steps = (132169 // batch_size) * 5
    warmup_steps = 250 # origin 700
    lr = 2e-4 # 0.00001
    betas = (0.9, 0.999)
    weight_decay = 0.01
    
    val_check_interval = 0.2
    model_name = "paust/pko-t5-large"
    
    # Path Setting
    load_path = ""
    log_dir = 'result'
    train_dataset_path = "nikluge-sc-2023-train.jsonl"
    val_dataset_path = "nikluge-sc-2023-dev.jsonl"
