#!/usr/bin/env python3
"""
Auto-recovery system for GPU hang detection and training restart
Monitors training process and automatically resumes on GPU hangs
"""

import os
import sys
import json
import time
import signal
import subprocess
import psutil
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import logging

# Setup logging
log_dir = Path("/tmp/training_recovery")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'base_dir': Path("/home/paulpas/git/ideas/llm_training_web_data"),
    'output_dir': Path("/home/paulpas/git/ideas/llm_training_web_data/finetune/output/terraform-expert-v2"),
    'config_template': "axolotl_config_v2_resume.yaml",
    'max_recovery_attempts': 5,
    'hang_check_interval': 30,  # seconds
    'process_check_interval': 10,  # seconds
}

# Recovery levels - progressively more conservative
RECOVERY_LEVELS = [
    {
        'name': 'Level 0 (Conservative)',
        'sequence_len': 1536,
        'micro_batch_size': 1,
        'gradient_accumulation_steps': 4,
        'learning_rate': 0.00005,
    },
    {
        'name': 'Level 1 (Very Conservative)',
        'sequence_len': 1024,
        'micro_batch_size': 1,
        'gradient_accumulation_steps': 2,
        'learning_rate': 0.00003,
    },
    {
        'name': 'Level 2 (Ultra Conservative)',
        'sequence_len': 768,
        'micro_batch_size': 1,
        'gradient_accumulation_steps': 1,
        'learning_rate': 0.00002,
    },
    {
        'name': 'Level 3 (Single Step)',
        'sequence_len': 512,
        'micro_batch_size': 1,
        'gradient_accumulation_steps': 1,
        'learning_rate': 0.00001,
    },
]


class GpuHangDetector:
    """Detects GPU hangs via kernel logs"""

    def __init__(self):
        self.last_hang_count = 0

    def check_for_hangs(self) -> bool:
        """Check if new GPU hangs detected since last check"""
        try:
            result = subprocess.run(
                ["sudo", "dmesg"],
                capture_output=True,
                text=True,
                timeout=5
            )
            hang_count = result.stdout.count("GPU Hang")

            if hang_count > self.last_hang_count:
                logger.warning(f"GPU hang detected! Count: {self.last_hang_count} -> {hang_count}")
                self.last_hang_count = hang_count
                return True

            return False
        except subprocess.TimeoutExpired:
            logger.error("dmesg check timed out")
            return False
        except Exception as e:
            logger.error(f"Error checking for GPU hangs: {e}")
            return False

    def reset_hang_count(self):
        """Reset hang detection counter"""
        self.last_hang_count = 0


class TrainingMonitor:
    """Monitors training process health"""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.last_step = 0
        self.stall_count = 0
        self.stall_threshold = 5  # Allow 5 checks without progress

    def get_current_step(self) -> int:
        """Extract current training step from trainer state"""
        try:
            state_file = CONFIG['output_dir'] / 'trainer_state.json'
            if not state_file.exists():
                return 0

            with open(state_file) as f:
                data = json.load(f)
                return data.get('global_step', 0)
        except Exception as e:
            logger.debug(f"Could not read trainer state: {e}")
            return self.last_step

    def check_process_alive(self, process_handles: List) -> bool:
        """Check if training processes are still running"""
        for proc in process_handles:
            if proc.poll() is None:  # Still running
                return True
        return False

    def check_progress(self) -> tuple[bool, str]:
        """Check if training is making progress"""
        current_step = self.get_current_step()

        if current_step == self.last_step:
            self.stall_count += 1
            if self.stall_count >= self.stall_threshold:
                logger.warning(f"Training stalled at step {current_step} for {self.stall_count} checks")
                return False, f"Training stalled at step {current_step}"
        else:
            logger.info(f"Training progress: {self.last_step} -> {current_step}")
            self.stall_count = 0
            self.last_step = current_step

        return True, f"Step {current_step}"

    def reset(self):
        """Reset monitoring state"""
        self.last_step = 0
        self.stall_count = 0


class ConfigModifier:
    """Modifies training config for recovery levels"""

    def __init__(self, base_config_path: Path):
        self.base_config_path = base_config_path
        self.current_level = 0

    def create_recovery_config(self, level: int) -> Path:
        """Create a config file for a specific recovery level"""
        if level >= len(RECOVERY_LEVELS):
            logger.error(f"Recovery level {level} exceeds maximum")
            return self.base_config_path

        level_config = RECOVERY_LEVELS[level]
        logger.info(f"Creating config for {level_config['name']}")

        # Read base config
        with open(self.base_config_path) as f:
            lines = f.readlines()

        # Modify config parameters
        modified_lines = []
        for line in lines:
            if line.startswith('sequence_len:'):
                modified_lines.append(f"sequence_len: {level_config['sequence_len']}\n")
            elif line.startswith('micro_batch_size:'):
                modified_lines.append(f"micro_batch_size: {level_config['micro_batch_size']}\n")
            elif line.startswith('gradient_accumulation_steps:'):
                modified_lines.append(f"gradient_accumulation_steps: {level_config['gradient_accumulation_steps']}\n")
            elif line.startswith('learning_rate:'):
                modified_lines.append(f"learning_rate: {level_config['learning_rate']}\n")
            else:
                modified_lines.append(line)

        # Write recovery config
        recovery_config_path = CONFIG['base_dir'] / f"finetune/axolotl_config_recovery_level{level}.yaml"
        with open(recovery_config_path, 'w') as f:
            f.writelines(modified_lines)

        logger.info(f"Created recovery config: {recovery_config_path}")
        return recovery_config_path


class TrainingRunner:
    """Runs training with auto-recovery"""

    def __init__(self):
        self.training_process: Optional[subprocess.Popen] = None
        self.recovery_attempt = 0
        self.monitor = TrainingMonitor(log_file)
        self.hang_detector = GpuHangDetector()
        self.config_modifier = ConfigModifier(CONFIG['base_dir'] / f"finetune/{CONFIG['config_template']}")

    def setup_environment(self):
        """Setup ROCm environment variables"""
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '9.0.6'
        os.environ['ROCR_VISIBLE_DEVICES'] = '0,1,2,3'
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1,2,3'
        os.environ['GPU_DEVICE_ORDINAL'] = '0,1,2,3'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['NCCL_DEBUG'] = 'INFO'
        logger.info("ROCm environment configured")

    def cleanup_zombie_processes(self):
        """Kill any orphaned training processes"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "axolotl|accelerate"],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                logger.warning(f"Found {len(pids)} zombie processes, killing...")
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except Exception as e:
                        logger.debug(f"Could not kill PID {pid}: {e}")
                time.sleep(2)
        except Exception as e:
            logger.debug(f"Error cleaning zombie processes: {e}")

    def reset_gpus(self):
        """Reset GPU state"""
        logger.info("Attempting GPU reset...")
        try:
            # Try GPU reset via rocm-smi
            subprocess.run(["sudo", "rocm-smi", "--gpureset"], timeout=10)
            time.sleep(5)
            logger.info("GPU reset completed")
        except Exception as e:
            logger.warning(f"GPU reset failed: {e}")

    def start_training(self, config_path: Path, resume_checkpoint: bool = True) -> bool:
        """Start training process"""
        try:
            os.chdir(CONFIG['base_dir'])

            # Activate venv and start training
            cmd = [
                "bash", "-c",
                f"source .venv/bin/activate && "
                f"accelerate launch -m axolotl.cli.train {config_path.name}"
            ]

            if resume_checkpoint and self.recovery_attempt > 0:
                cmd[-1] += " --resume-from-checkpoint finetune/output/terraform-expert-v2/checkpoint-92"

            logger.info(f"Starting training with: {cmd[-1][:100]}...")
            self.training_process = subprocess.Popen(
                cmd,
                cwd=CONFIG['base_dir'] / 'finetune',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"Training process started (PID: {self.training_process.pid})")
            return True
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return False

    def monitor_training(self) -> tuple[bool, str]:
        """Monitor training and detect issues"""
        # Check for GPU hangs
        if self.hang_detector.check_for_hangs():
            return False, "GPU hang detected"

        # Check if process crashed
        if self.training_process.poll() is not None:
            return False, "Training process exited"

        # Check for progress stalls
        is_progressing, status = self.monitor.check_progress()
        if not is_progressing:
            return False, status

        return True, status

    def recover_from_failure(self, reason: str) -> bool:
        """Attempt recovery from training failure"""
        self.recovery_attempt += 1

        if self.recovery_attempt >= CONFIG['max_recovery_attempts']:
            logger.error(f"Maximum recovery attempts ({CONFIG['max_recovery_attempts']}) reached")
            return False

        logger.warning(f"Recovery attempt {self.recovery_attempt}/{CONFIG['max_recovery_attempts']}")
        logger.warning(f"Failure reason: {reason}")

        # Kill existing processes
        self.cleanup_zombie_processes()

        # Reset GPUs
        self.reset_gpus()

        # Create recovery config
        level = min(self.recovery_attempt - 1, len(RECOVERY_LEVELS) - 1)
        config_path = self.config_modifier.create_recovery_config(level)

        # Reset monitor state
        self.monitor.reset()
        self.hang_detector.reset_hang_count()

        logger.info(f"Resuming with recovery level {level}: {RECOVERY_LEVELS[level]['name']}")

        # Start training again
        return self.start_training(config_path, resume_checkpoint=True)

    def run(self):
        """Main training loop with auto-recovery"""
        logger.info("=" * 60)
        logger.info("Training Auto-Recovery System Started")
        logger.info("=" * 60)

        self.setup_environment()
        self.cleanup_zombie_processes()

        # Start initial training
        config_path = CONFIG['base_dir'] / f"finetune/{CONFIG['config_template']}"
        if not self.start_training(config_path):
            logger.error("Failed to start initial training")
            sys.exit(1)

        # Monitor loop
        last_hang_check = time.time()
        check_interval = CONFIG['hang_check_interval']
        process_check_interval = CONFIG['process_check_interval']

        try:
            while self.recovery_attempt < CONFIG['max_recovery_attempts']:
                # Check training health
                is_healthy, status = self.monitor_training()

                if not is_healthy:
                    logger.error(f"Training failure detected: {status}")
                    if self.recover_from_failure(status):
                        continue
                    else:
                        logger.error("Recovery failed")
                        break

                time.sleep(process_check_interval)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            # Cleanup
            if self.training_process:
                logger.info("Terminating training process...")
                self.training_process.terminate()
                try:
                    self.training_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.training_process.kill()

        logger.info("=" * 60)
        logger.info("Training Auto-Recovery System Stopped")
        logger.info(f"Recovery attempts: {self.recovery_attempt}/{CONFIG['max_recovery_attempts']}")
        logger.info(f"Log file: {log_file}")
        logger.info("=" * 60)


def main():
    """Entry point"""
    runner = TrainingRunner()
    runner.run()


if __name__ == '__main__':
    main()
