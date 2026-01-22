"""
Unified Configuration Loader

Provides typed access to all pipeline configuration with path resolution,
profile support, and environment variable overrides.

Usage:
    from pipeline.config_loader import load_config, PipelineConfig

    config = load_config("config/pipeline_config.yaml")
    print(config.training.base_model)
    print(config.paths.data.training)

    # With profile
    config = load_config("config/pipeline_config.yaml", profile="test")
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import copy


@dataclass
class PathsData:
    raw: Path
    training: Path
    combined: Path
    deduped: Path


@dataclass
class PathsOutput:
    checkpoints: Path
    merged: Path
    gguf: Path


@dataclass
class Paths:
    root: Path
    repos: Path
    data: PathsData
    output: PathsOutput
    logs: Path
    cache: Path


@dataclass
class GitSource:
    name: str
    url: str
    subdirs: List[str]
    type: str
    enabled: bool
    description: str


@dataclass
class WebSource:
    name: str
    start_url: str
    allowed_domain: str
    max_depth: int
    request_delay: float
    enabled: bool
    description: str


@dataclass
class Deduplication:
    enabled: bool
    min_output_length: int
    max_repetition: int
    similarity_threshold: float


@dataclass
class Processing:
    min_code_length: int
    max_code_length: int
    include_docstrings: bool
    include_functions: bool
    include_full_files: bool
    deduplication: Deduplication
    output_format: str
    val_set_size: float


@dataclass
class LoRAConfig:
    enabled: bool
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]


@dataclass
class Hyperparameters:
    num_epochs: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler: str
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    optimizer: str


@dataclass
class Training:
    base_model: str
    model_type: str
    tokenizer_type: str
    trust_remote_code: bool
    lora: LoRAConfig
    hyperparameters: Hyperparameters
    sequence_len: int
    sample_packing: bool
    pad_to_sequence_len: bool
    eval_steps: int
    save_steps: int
    save_total_limit: int
    early_stopping_patience: int
    gradient_checkpointing: bool
    flash_attention: bool
    bf16: Union[bool, str]
    fp16: bool
    num_gpus: int
    chat_template: str = "auto"  # auto, deepseek, gemma, qwen, llama, phi


@dataclass
class MergeConfig:
    torch_dtype: str
    device_map: str


@dataclass
class GGUFConfig:
    quantization: str
    llama_cpp_path: Path


@dataclass
class OllamaConfig:
    model_name: str
    temperature: float
    repeat_penalty: float
    num_predict: int
    system_prompt: str


@dataclass
class Export:
    merge: MergeConfig
    gguf: GGUFConfig
    ollama: OllamaConfig


@dataclass
class Logging:
    level: str
    format: str
    file: str
    console: bool


@dataclass
class Pipeline:
    name: str
    version: str
    description: str


@dataclass
class PipelineConfig:
    """Main configuration container with all settings."""
    pipeline: Pipeline
    paths: Paths
    git_sources: List[GitSource]
    web_sources: List[WebSource]
    processing: Processing
    training: Training
    export: Export
    logging: Logging

    # Raw config for advanced access
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def get_enabled_git_sources(self) -> List[GitSource]:
        """Get only enabled git sources."""
        return [s for s in self.git_sources if s.enabled]

    def get_enabled_web_sources(self) -> List[WebSource]:
        """Get only enabled web sources."""
        return [s for s in self.web_sources if s.enabled]

    def get_output_name(self) -> str:
        """Get full output name with version."""
        return f"{self.pipeline.name}-{self.pipeline.version}"

    def get_ollama_model_name(self) -> str:
        """Get Ollama model name with version."""
        return f"{self.export.ollama.model_name}-{self.pipeline.version}"


def _resolve_path(base: Path, path_str: str) -> Path:
    """Resolve path relative to base directory."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    # Expand ~ for home directory
    if str(path).startswith("~"):
        return Path(os.path.expanduser(str(path)))
    return base / path


def _apply_profile(config: Dict, profile_name: str) -> Dict:
    """Apply a profile's overrides to the config."""
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Profile '{profile_name}' not found. Available: {list(profiles.keys())}")

    profile = profiles[profile_name]
    config = copy.deepcopy(config)

    # Deep merge profile into config
    def deep_merge(base: Dict, override: Dict) -> Dict:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    return deep_merge(config, profile)


def _apply_env_overrides(config: Dict) -> Dict:
    """Apply environment variable overrides."""
    # Support common overrides via environment variables
    env_mappings = {
        "PIPELINE_BASE_MODEL": ("training", "base_model"),
        "PIPELINE_NUM_GPUS": ("training", "num_gpus"),
        "PIPELINE_NUM_EPOCHS": ("training", "hyperparameters", "num_epochs"),
        "PIPELINE_LEARNING_RATE": ("training", "hyperparameters", "learning_rate"),
        "PIPELINE_BATCH_SIZE": ("training", "hyperparameters", "micro_batch_size"),
        "PIPELINE_QUANTIZATION": ("export", "gguf", "quantization"),
        "PIPELINE_OLLAMA_NAME": ("export", "ollama", "model_name"),
    }

    for env_var, path in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Navigate to the right location and set value
            current = config
            for key in path[:-1]:
                current = current.setdefault(key, {})
            # Type conversion
            if path[-1] in ("num_gpus", "num_epochs", "micro_batch_size"):
                value = int(value)
            elif path[-1] == "learning_rate":
                value = float(value)
            current[path[-1]] = value

    return config


def load_config(
    config_path: str = "config/pipeline_config.yaml",
    profile: Optional[str] = None,
    project_root: Optional[Path] = None
) -> PipelineConfig:
    """
    Load and parse pipeline configuration.

    Args:
        config_path: Path to YAML config file
        profile: Optional profile name to apply
        project_root: Project root directory (auto-detected if not specified)

    Returns:
        PipelineConfig with all settings
    """
    # Determine project root
    if project_root is None:
        # Try to find project root by looking for config file
        config_file = Path(config_path)
        if config_file.is_absolute():
            project_root = config_file.parent.parent
        else:
            # Search upward for config/pipeline_config.yaml
            current = Path.cwd()
            while current != current.parent:
                if (current / config_path).exists():
                    project_root = current
                    break
                current = current.parent
            else:
                project_root = Path.cwd()

    config_file = project_root / config_path if not Path(config_path).is_absolute() else Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Load YAML
    with open(config_file) as f:
        raw_config = yaml.safe_load(f)

    # Apply profile if specified
    if profile:
        raw_config = _apply_profile(raw_config, profile)

    # Apply environment overrides
    raw_config = _apply_env_overrides(raw_config)

    # Parse into dataclasses
    cfg = raw_config

    # Paths
    paths_cfg = cfg["paths"]
    root = _resolve_path(project_root, paths_cfg.get("root", "."))

    paths = Paths(
        root=root,
        repos=_resolve_path(root, paths_cfg["repos"]),
        data=PathsData(
            raw=_resolve_path(root, paths_cfg["data"]["raw"]),
            training=_resolve_path(root, paths_cfg["data"]["training"]),
            combined=_resolve_path(root, paths_cfg["data"]["combined"]),
            deduped=_resolve_path(root, paths_cfg["data"]["deduped"]),
        ),
        output=PathsOutput(
            checkpoints=_resolve_path(root, paths_cfg["output"]["checkpoints"]),
            merged=_resolve_path(root, paths_cfg["output"]["merged"]),
            gguf=_resolve_path(root, paths_cfg["output"]["gguf"]),
        ),
        logs=_resolve_path(root, paths_cfg["logs"]),
        cache=_resolve_path(root, paths_cfg["cache"]),
    )

    # Git sources
    git_sources = [
        GitSource(
            name=s["name"],
            url=s["url"],
            subdirs=s.get("subdirs", []),
            type=s.get("type", "python"),
            enabled=s.get("enabled", True),
            description=s.get("description", ""),
        )
        for s in cfg.get("git_sources", [])
    ]

    # Web sources
    web_sources = [
        WebSource(
            name=s["name"],
            start_url=s["start_url"],
            allowed_domain=s["allowed_domain"],
            max_depth=s.get("max_depth", 3),
            request_delay=s.get("request_delay", 1.0),
            enabled=s.get("enabled", False),
            description=s.get("description", ""),
        )
        for s in cfg.get("web_sources", [])
    ]

    # Processing
    proc_cfg = cfg["processing"]
    processing = Processing(
        min_code_length=proc_cfg.get("min_code_length", 50),
        max_code_length=proc_cfg.get("max_code_length", 10000),
        include_docstrings=proc_cfg.get("include_docstrings", True),
        include_functions=proc_cfg.get("include_functions", True),
        include_full_files=proc_cfg.get("include_full_files", True),
        deduplication=Deduplication(
            enabled=proc_cfg["deduplication"].get("enabled", True),
            min_output_length=proc_cfg["deduplication"].get("min_output_length", 50),
            max_repetition=proc_cfg["deduplication"].get("max_repetition", 3),
            similarity_threshold=proc_cfg["deduplication"].get("similarity_threshold", 0.85),
        ),
        output_format=proc_cfg.get("output_format", "alpaca"),
        val_set_size=proc_cfg.get("val_set_size", 0.05),
    )

    # Training
    train_cfg = cfg["training"]
    hp_cfg = train_cfg["hyperparameters"]
    lora_cfg = train_cfg["lora"]

    training = Training(
        base_model=train_cfg["base_model"],
        model_type=train_cfg.get("model_type", "AutoModelForCausalLM"),
        tokenizer_type=train_cfg.get("tokenizer_type", "AutoTokenizer"),
        trust_remote_code=train_cfg.get("trust_remote_code", True),
        lora=LoRAConfig(
            enabled=lora_cfg.get("enabled", True),
            r=lora_cfg.get("r", 16),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.1),
            target_modules=lora_cfg.get("target_modules", []),
        ),
        hyperparameters=Hyperparameters(
            num_epochs=hp_cfg.get("num_epochs", 3),
            micro_batch_size=hp_cfg.get("micro_batch_size", 4),
            gradient_accumulation_steps=hp_cfg.get("gradient_accumulation_steps", 2),
            learning_rate=hp_cfg.get("learning_rate", 3e-5),
            lr_scheduler=hp_cfg.get("lr_scheduler", "cosine"),
            warmup_ratio=hp_cfg.get("warmup_ratio", 0.1),
            weight_decay=hp_cfg.get("weight_decay", 0.1),
            max_grad_norm=hp_cfg.get("max_grad_norm", 0.5),
            optimizer=hp_cfg.get("optimizer", "adamw_torch"),
        ),
        sequence_len=train_cfg.get("sequence_len", 2048),
        sample_packing=train_cfg.get("sample_packing", False),
        pad_to_sequence_len=train_cfg.get("pad_to_sequence_len", True),
        eval_steps=train_cfg.get("eval_steps", 100),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 5),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 10),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        flash_attention=train_cfg.get("flash_attention", False),
        bf16=train_cfg.get("bf16", "auto"),
        fp16=train_cfg.get("fp16", False),
        num_gpus=train_cfg.get("num_gpus", 1),
    )

    # Export
    exp_cfg = cfg["export"]
    export = Export(
        merge=MergeConfig(
            torch_dtype=exp_cfg["merge"].get("torch_dtype", "float16"),
            device_map=exp_cfg["merge"].get("device_map", "cpu"),
        ),
        gguf=GGUFConfig(
            quantization=exp_cfg["gguf"].get("quantization", "q4_k_m"),
            llama_cpp_path=Path(os.path.expanduser(exp_cfg["gguf"].get("llama_cpp_path", "~/llama.cpp"))),
        ),
        ollama=OllamaConfig(
            model_name=exp_cfg["ollama"].get("model_name", "python-expert"),
            temperature=exp_cfg["ollama"].get("temperature", 0.5),
            repeat_penalty=exp_cfg["ollama"].get("repeat_penalty", 1.3),
            num_predict=exp_cfg["ollama"].get("num_predict", 512),
            system_prompt=exp_cfg["ollama"].get("system_prompt", ""),
        ),
    )

    # Logging
    log_cfg = cfg.get("logging", {})
    logging_config = Logging(
        level=log_cfg.get("level", "INFO"),
        format=log_cfg.get("format", "%(asctime)s %(levelname)s %(name)s - %(message)s"),
        file=log_cfg.get("file", "pipeline.log"),
        console=log_cfg.get("console", True),
    )

    # Pipeline metadata
    pipe_cfg = cfg.get("pipeline", {})
    pipeline = Pipeline(
        name=pipe_cfg.get("name", "llm-training"),
        version=pipe_cfg.get("version", "v1"),
        description=pipe_cfg.get("description", ""),
    )

    return PipelineConfig(
        pipeline=pipeline,
        paths=paths,
        git_sources=git_sources,
        web_sources=web_sources,
        processing=processing,
        training=training,
        export=export,
        logging=logging_config,
        _raw=raw_config,
    )


def generate_axolotl_config(config: PipelineConfig, output_path: Optional[Path] = None) -> str:
    """
    Generate Axolotl training config from pipeline config.

    Args:
        config: Pipeline configuration
        output_path: If provided, save config to this path

    Returns:
        YAML string of Axolotl config
    """
    output_dir = config.paths.output.checkpoints / config.get_output_name()

    axolotl_config = {
        "base_model": config.training.base_model,
        "model_type": config.training.model_type,
        "tokenizer_type": config.training.tokenizer_type,
        "trust_remote_code": config.training.trust_remote_code,

        "load_in_8bit": False,
        "load_in_4bit": False,

        "datasets": [
            {
                "path": str(config.paths.data.deduped),
                "type": config.processing.output_format,
            }
        ],

        "dataset_prepared_path": str(config.paths.cache / "prepared_dataset"),
        "val_set_size": config.processing.val_set_size,
        "output_dir": str(output_dir),

        "adapter": "lora" if config.training.lora.enabled else None,
        "lora_r": config.training.lora.r,
        "lora_alpha": config.training.lora.alpha,
        "lora_dropout": config.training.lora.dropout,
        "lora_target_linear": True,
        "lora_target_modules": config.training.lora.target_modules,

        "sequence_len": config.training.sequence_len,
        "sample_packing": config.training.sample_packing,
        "pad_to_sequence_len": config.training.pad_to_sequence_len,

        "gradient_accumulation_steps": config.training.hyperparameters.gradient_accumulation_steps,
        "micro_batch_size": config.training.hyperparameters.micro_batch_size,
        "num_epochs": config.training.hyperparameters.num_epochs,
        "optimizer": config.training.hyperparameters.optimizer,
        "lr_scheduler": config.training.hyperparameters.lr_scheduler,
        "learning_rate": config.training.hyperparameters.learning_rate,
        "weight_decay": config.training.hyperparameters.weight_decay,
        "max_grad_norm": config.training.hyperparameters.max_grad_norm,
        "warmup_ratio": config.training.hyperparameters.warmup_ratio,

        "eval_steps": config.training.eval_steps,
        "save_steps": config.training.save_steps,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "save_total_limit": config.training.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,

        "early_stopping_patience": config.training.early_stopping_patience,

        "gradient_checkpointing": config.training.gradient_checkpointing,
        "flash_attention": config.training.flash_attention,

        "logging_steps": 5,

        "train_on_inputs": False,
        "group_by_length": False,
        "bf16": config.training.bf16,
        "fp16": config.training.fp16,
        "tf32": False,

        "special_tokens": {
            "pad_token": "<pad>",
        },

        # Disable wandb
        "wandb_project": None,
        "wandb_entity": None,
        "wandb_watch": None,
        "wandb_name": None,
        "wandb_log_model": None,
    }

    yaml_str = yaml.dump(axolotl_config, default_flow_style=False, sort_keys=False)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"# Auto-generated from {config.pipeline.name} config\n")
            f.write(f"# Version: {config.pipeline.version}\n\n")
            f.write(yaml_str)

    return yaml_str
