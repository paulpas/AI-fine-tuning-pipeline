"""
Unified Pipeline Runner

Run the complete LLM training pipeline or individual stages:
1. collect   - Clone git repositories
2. extract   - Extract training data from sources
3. combine   - Combine all datasets
4. dedupe    - Deduplicate combined dataset
5. train     - Train model with Axolotl
6. export    - Export to Ollama

Usage:
    # Run entire pipeline
    python -m pipeline.runner --config config/pipeline_config.yaml

    # Run specific stages
    python -m pipeline.runner --config config/pipeline_config.yaml --stage extract
    python -m pipeline.runner --config config/pipeline_config.yaml --stage train

    # Run with profile
    python -m pipeline.runner --config config/pipeline_config.yaml --profile test

    # Skip specific stages
    python -m pipeline.runner --config config/pipeline_config.yaml --skip collect,extract
"""

import subprocess
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Set
from dataclasses import dataclass, field

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config_loader import load_config, generate_axolotl_config, PipelineConfig
from pipeline.data_extractor import extract_from_git_source, save_examples, ExtractionResult
from pipeline.model_exporter import export_model


log = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result of a pipeline stage."""
    stage: str
    success: bool
    duration: float  # seconds
    message: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of entire pipeline run."""
    success: bool
    stages: List[StageResult]
    total_duration: float
    config_name: str
    config_version: str


# =============================================================================
# Stage Implementations
# =============================================================================

def stage_collect(config: PipelineConfig) -> StageResult:
    """Stage 1: Clone git repositories."""
    start = datetime.now()
    log.info("=" * 60)
    log.info("STAGE 1: COLLECTING DATA SOURCES")
    log.info("=" * 60)

    repos_dir = config.paths.repos
    repos_dir.mkdir(parents=True, exist_ok=True)

    cloned = 0
    skipped = 0
    errors = []

    for source in config.get_enabled_git_sources():
        target = repos_dir / source.name

        if target.exists():
            log.info(f"  [SKIP] {source.name} (already exists)")
            skipped += 1
            continue

        log.info(f"  Cloning {source.name}...")
        try:
            cmd = ["git", "clone", "--depth", "1", source.url, str(target)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                log.info(f"  [OK] {source.name}")
                cloned += 1
            else:
                log.error(f"  [FAIL] {source.name}: {result.stderr}")
                errors.append(source.name)
        except Exception as e:
            log.error(f"  [FAIL] {source.name}: {e}")
            errors.append(source.name)

    duration = (datetime.now() - start).total_seconds()
    success = len(errors) == 0

    return StageResult(
        stage="collect",
        success=success,
        duration=duration,
        message=f"Cloned {cloned}, skipped {skipped}, errors {len(errors)}",
        details={"cloned": cloned, "skipped": skipped, "errors": errors}
    )


def stage_extract(config: PipelineConfig) -> StageResult:
    """Stage 2: Extract training data from sources."""
    start = datetime.now()
    log.info("=" * 60)
    log.info("STAGE 2: EXTRACTING TRAINING DATA")
    log.info("=" * 60)

    training_dir = config.paths.data.training
    training_dir.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    source_results = {}

    for source in config.get_enabled_git_sources():
        log.info(f"  Processing {source.name}...")

        result = extract_from_git_source(
            source=source,
            repos_dir=config.paths.repos,
            min_length=config.processing.min_code_length,
            max_length=config.processing.max_code_length,
        )

        if result.examples:
            output_file = training_dir / f"{source.name}_alpaca.json"
            save_examples(result.examples, output_file, config.processing.output_format)
            log.info(f"  [OK] {source.name}: {len(result.examples)} examples")
            total_examples += len(result.examples)
            source_results[source.name] = len(result.examples)
        else:
            log.warning(f"  [WARN] {source.name}: no examples extracted")
            source_results[source.name] = 0

    duration = (datetime.now() - start).total_seconds()

    return StageResult(
        stage="extract",
        success=True,
        duration=duration,
        message=f"Extracted {total_examples} examples from {len(source_results)} sources",
        details={"total_examples": total_examples, "by_source": source_results}
    )


def stage_combine(config: PipelineConfig) -> StageResult:
    """Stage 3: Combine all datasets."""
    start = datetime.now()
    log.info("=" * 60)
    log.info("STAGE 3: COMBINING DATASETS")
    log.info("=" * 60)

    training_dir = config.paths.data.training
    combined_path = config.paths.data.combined

    combined = []
    source_counts = {}

    # Find all *_alpaca.json files
    for json_file in sorted(training_dir.glob("*_alpaca.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                log.info(f"  {json_file.name}: {len(data)} examples")
                combined.extend(data)
                source_counts[json_file.stem] = len(data)
        except Exception as e:
            log.error(f"  {json_file.name}: ERROR - {e}")

    log.info(f"\n  Total combined: {len(combined)}")

    # Save combined dataset
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    log.info(f"  Saved to: {combined_path}")

    duration = (datetime.now() - start).total_seconds()

    return StageResult(
        stage="combine",
        success=True,
        duration=duration,
        message=f"Combined {len(combined)} examples from {len(source_counts)} sources",
        details={"total": len(combined), "by_source": source_counts}
    )


def stage_dedupe(config: PipelineConfig) -> StageResult:
    """Stage 4: Deduplicate combined dataset."""
    start = datetime.now()
    log.info("=" * 60)
    log.info("STAGE 4: DEDUPLICATING DATASET")
    log.info("=" * 60)

    # Import deduplication function
    from pipeline.data_processor import deduplicate_dataset

    dedup_config = config.processing.deduplication

    stats = deduplicate_dataset(
        input_path=str(config.paths.data.combined),
        output_path=str(config.paths.data.deduped),
        min_output_length=dedup_config.min_output_length,
        max_repetition=dedup_config.max_repetition,
    )

    duration = (datetime.now() - start).total_seconds()

    return StageResult(
        stage="dedupe",
        success=True,
        duration=duration,
        message=f"Kept {stats['kept']} of {stats['original']} samples",
        details=stats
    )


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in output directory."""
    try:
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
            reverse=True
        )
        return checkpoints[0] if checkpoints else None
    except Exception:
        return None


def stage_train(config: PipelineConfig) -> StageResult:
    """Stage 5: Train model with Axolotl (with GUARANTEED resume support)."""
    start = datetime.now()
    log.info("=" * 60)
    log.info("STAGE 5: TRAINING MODEL")
    log.info("=" * 60)

    # Generate Axolotl config
    axolotl_config_path = config.paths.root / "finetune" / f"axolotl_config_{config.pipeline.name}.yaml"
    generate_axolotl_config(config, axolotl_config_path)
    log.info(f"Generated Axolotl config: {axolotl_config_path}")

    # Check for existing checkpoints
    output_dir = config.paths.output.checkpoints / config.get_output_name()
    latest_checkpoint = find_latest_checkpoint(output_dir)

    resume_checkpoint = None
    if latest_checkpoint:
        log.info(f"Found existing checkpoint: {latest_checkpoint.name}")
        # Get checkpoint info
        try:
            import json
            trainer_state = json.loads((latest_checkpoint / "trainer_state.json").read_text())
            step = trainer_state.get("global_step", 0)
            total = trainer_state.get("max_steps", "?")
            log.info(f"  Step: {step}/{total}")
            resume_checkpoint = latest_checkpoint
            log.info(f"WILL RESUME from {latest_checkpoint.name}")
        except Exception as e:
            log.warning(f"Could not read checkpoint info: {e}")

    # Run training
    num_gpus = config.training.num_gpus
    log.info(f"Starting training with {num_gpus} GPU(s)...")

    cmd = [
        "accelerate", "launch",
        f"--num_processes={num_gpus}",
        "-m", "axolotl.cli.train",
        str(axolotl_config_path)
    ]

    # EXPLICIT resume: Add resume flag if checkpoint exists
    if resume_checkpoint:
        cmd.extend([
            "--resume_from_checkpoint",
            str(resume_checkpoint)
        ])
        log.info(f"Resume flag: --resume_from_checkpoint {resume_checkpoint}")
    else:
        log.info("No checkpoint found - starting fresh training")

    log.info(f"Full command: {' '.join(cmd)}")

    # Run in foreground so we can see progress
    result = subprocess.run(cmd)

    duration = (datetime.now() - start).total_seconds()
    success = result.returncode == 0

    return StageResult(
        stage="train",
        success=success,
        duration=duration,
        message="Training complete" if success else f"Training failed with code {result.returncode}",
        details={
            "return_code": result.returncode,
            "hours": duration / 3600,
            "resumed_from": str(resume_checkpoint) if resume_checkpoint else None
        }
    )


def stage_export(config: PipelineConfig) -> StageResult:
    """Stage 6: Export model to Ollama."""
    start = datetime.now()
    log.info("=" * 60)
    log.info("STAGE 6: EXPORTING MODEL")
    log.info("=" * 60)

    result = export_model(config)

    duration = (datetime.now() - start).total_seconds()

    if result.success:
        return StageResult(
            stage="export",
            success=True,
            duration=duration,
            message=f"Exported to Ollama as {result.ollama_model}",
            details={
                "merged_path": str(result.merged_path) if result.merged_path else None,
                "gguf_path": str(result.gguf_path) if result.gguf_path else None,
                "ollama_model": result.ollama_model,
            }
        )
    else:
        return StageResult(
            stage="export",
            success=False,
            duration=duration,
            message=f"Export failed: {result.error}",
            details={"error": result.error}
        )


# =============================================================================
# Pipeline Runner
# =============================================================================

STAGES = {
    "collect": stage_collect,
    "extract": stage_extract,
    "combine": stage_combine,
    "dedupe": stage_dedupe,
    "train": stage_train,
    "export": stage_export,
}

STAGE_ORDER = ["collect", "extract", "combine", "dedupe", "train", "export"]


def run_pipeline(
    config: PipelineConfig,
    stages: Optional[List[str]] = None,
    skip_stages: Optional[Set[str]] = None,
) -> PipelineResult:
    """
    Run the pipeline.

    Args:
        config: Pipeline configuration
        stages: Specific stages to run (None = all)
        skip_stages: Stages to skip

    Returns:
        PipelineResult with all stage results
    """
    start = datetime.now()
    results = []

    # Determine stages to run
    if stages:
        stages_to_run = [s for s in STAGE_ORDER if s in stages]
    else:
        stages_to_run = STAGE_ORDER

    if skip_stages:
        stages_to_run = [s for s in stages_to_run if s not in skip_stages]

    log.info("=" * 60)
    log.info(f"PIPELINE: {config.pipeline.name} {config.pipeline.version}")
    log.info(f"Stages: {', '.join(stages_to_run)}")
    log.info("=" * 60)

    overall_success = True

    for stage_name in stages_to_run:
        stage_func = STAGES[stage_name]

        try:
            result = stage_func(config)
            results.append(result)

            if result.success:
                log.info(f"[OK] {stage_name}: {result.message} ({result.duration:.1f}s)")
            else:
                log.error(f"[FAIL] {stage_name}: {result.message}")
                overall_success = False
                break  # Stop on failure

        except Exception as e:
            log.exception(f"Stage {stage_name} raised exception")
            results.append(StageResult(
                stage=stage_name,
                success=False,
                duration=0,
                message=str(e),
            ))
            overall_success = False
            break

    total_duration = (datetime.now() - start).total_seconds()

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 60)
    for r in results:
        status = "[OK]" if r.success else "[FAIL]"
        log.info(f"  {status} {r.stage}: {r.message}")
    log.info(f"Total time: {total_duration:.1f}s ({total_duration/3600:.2f} hours)")
    log.info("=" * 60)

    return PipelineResult(
        success=overall_success,
        stages=results,
        total_duration=total_duration,
        config_name=config.pipeline.name,
        config_version=config.pipeline.version,
    )


def setup_logging(config: PipelineConfig):
    """Setup logging from config."""
    handlers = []

    if config.logging.console:
        handlers.append(logging.StreamHandler())

    if config.logging.file:
        log_dir = config.paths.logs
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / config.logging.file))

    logging.basicConfig(
        format=config.logging.format,
        level=getattr(logging, config.logging.level),
        handlers=handlers,
        force=True,
    )


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LLM training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run entire pipeline
    python -m pipeline.runner

    # Run specific stage
    python -m pipeline.runner --stage train

    # Run multiple stages
    python -m pipeline.runner --stage extract,combine,dedupe

    # Skip stages
    python -m pipeline.runner --skip collect,extract

    # Use different config
    python -m pipeline.runner --config config/my_config.yaml

    # Use profile
    python -m pipeline.runner --profile test
        """
    )

    parser.add_argument(
        "--config", "-c",
        default="config/pipeline_config.yaml",
        help="Path to pipeline config file"
    )
    parser.add_argument(
        "--profile", "-p",
        help="Config profile to use (e.g., test, production)"
    )
    parser.add_argument(
        "--stage", "-s",
        help="Specific stage(s) to run (comma-separated)"
    )
    parser.add_argument(
        "--skip",
        help="Stage(s) to skip (comma-separated)"
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List available stages and exit"
    )

    args = parser.parse_args()

    if args.list_stages:
        print("Available stages:")
        for i, stage in enumerate(STAGE_ORDER, 1):
            print(f"  {i}. {stage}")
        return 0

    # Load config
    try:
        config = load_config(args.config, profile=args.profile)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    setup_logging(config)

    # Parse stages
    stages = None
    if args.stage:
        stages = [s.strip() for s in args.stage.split(",")]
        invalid = [s for s in stages if s not in STAGES]
        if invalid:
            log.error(f"Invalid stages: {invalid}")
            log.error(f"Valid stages: {list(STAGES.keys())}")
            return 1

    skip_stages = None
    if args.skip:
        skip_stages = {s.strip() for s in args.skip.split(",")}

    # Run pipeline
    result = run_pipeline(config, stages=stages, skip_stages=skip_stages)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
