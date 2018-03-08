data_config = {
    'word_size' : 16,
    'word_count_threshold' : 10,
    'char_count_threshold' : 50,
    'pickle_file' : 'vocabs.pkl',
    'glove_file': 'glove.6B.100d.txt'
}

model_config = {
    'hidden_dim'     	: 100,
    'char_convs'     	: 100,
    'char_emb_dim'   	: 8,
    'word_emb_dim'      : 100,
    'dropout'        	: 0.2,
    'highway_layers' 	: 2,
    'two_step'          : True,
    'use_cudnn'         : True,
}

training_config = {
    'logdir'            : 'logs', # logdir for log outputs and tensorboard
    'tensorboard_freq'  : 1, # tensorboard record frequence 
    'minibatch_size'    : 10,    # in samples when using ctf reader, per worker
    'epoch_size'        : 10,   # in sequences, when using ctf reader
    'log_freq'          : 100,     # in minibatchs
    'max_epochs'        : 300,
    'lr'                : 3,
    'train_data'        : 'train.ctf',  # or 'train.tsv'
    'val_data'          : 'dev.ctf',
    'val_interval'      : 1,       # interval in epochs to run validation
    'stop_after'        : 2,       # num epochs to stop if no CV improvement
    'minibatch_seqs'    : 16,      # num sequences of minibatch, when using tsv reader, per worker
    'distributed_after' : 0,       # num sequences after which to start distributed training
    'gpu_pad'           : 0, #emmmmmmm
    'gpu_cnt'           : 1, # number of gpus
    'multi_gpu'         : False, # using multi GPU training
}
