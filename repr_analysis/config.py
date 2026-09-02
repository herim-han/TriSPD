def build_config():
    return {
        'property_width': 768, 'dist_width': 768, 'embed_dim': 256,
        'batch_size': 64, 'temp': 0.07, 'mlm_probability': 0.15,
        'queue_size': 24960, 'momentum': 0.995, 'alpha': 0.4,
        'bert_config_text': './src/config/config_bert.json',
        'bert_config_property': './src/config/config_bert_property.json',
        'bert_config_dist': './src/config/config_bert_dist.json',
        'schedular': {'sched': 'cosine', 'lr': 5e-5, 'epochs': 10, 'min_lr': 5e-5,
                      'decay_rate': 1, 'warmup_lr': 5e-5, 'warmup_epochs': 20,
                      'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 5e-5, 'weight_decay': 0.02},
    }
