#!/usr/bin/env python3
"""
Generate Axolotl config from pipeline config.
"""

import yaml
import argparse
from pathlib import Path


def generate_config(pipeline_config: str, output_path: str):
    """Generate Axolotl config from pipeline config."""
    with open(pipeline_config) as f:
        config = yaml.safe_load(f)

    project = config['project']
    model = config['model']
    lora = config['lora']
    training = config['training']
    data = config['data']

    axolotl_config = {
        # Base model
        'base_model': model['name'],
        'model_type': model['type'],
        'tokenizer_type': 'AutoTokenizer',
        'trust_remote_code': model.get('trust_remote_code', True),

        # Dataset
        'datasets': [{
            'path': data['cleaned_data_path'],
            'type': 'alpaca'
        }],

        # Output
        'output_dir': f"{project['output_dir']}/{project['name']}",

        # LoRA
        'adapter': 'lora',
        'lora_r': lora['rank'],
        'lora_alpha': lora['alpha'],
        'lora_dropout': lora['dropout'],
        'lora_target_linear': True,
        'lora_target_modules': lora['target_modules'],

        # Training
        'gradient_accumulation_steps': training['gradient_accumulation_steps'],
        'micro_batch_size': training['micro_batch_size'],
        'num_epochs': training['epochs'],
        'learning_rate': training['learning_rate'],
        'weight_decay': training['weight_decay'],
        'max_grad_norm': training['max_grad_norm'],
        'lr_scheduler': training['lr_scheduler'],
        'warmup_ratio': training['warmup_ratio'],
        'optimizer': 'adamw_torch',

        # Data
        'sequence_len': training['sequence_length'],
        'sample_packing': training['sample_packing'],
        'pad_to_sequence_len': True,
        'val_set_size': training['val_set_size'],

        # Evaluation
        'eval_steps': training['eval_steps'],
        'save_steps': training['save_steps'],
        'save_strategy': 'steps',
        'eval_strategy': 'steps',
        'save_total_limit': 5,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss',
        'early_stopping_patience': training['early_stopping_patience'],

        # Precision
        'bf16': True,
        'tf32': False,
        'flash_attention': False,

        # Logging
        'logging_steps': 10,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(axolotl_config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated Axolotl config: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Axolotl config")
    parser.add_argument('--pipeline-config', required=True, help="Pipeline config YAML")
    parser.add_argument('--output', required=True, help="Output Axolotl config path")
    args = parser.parse_args()

    generate_config(args.pipeline_config, args.output)
