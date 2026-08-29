# scripts/monitor_epoch1_completion.py
import os
import sys
import time
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('Epoch1Monitor')

output_dir = Path('checkpoints/trocr-large-real-400k')
log_file = Path('/Users/damian/.gemini/antigravity/brain/c2619dac-5aab-4663-a9b7-ae0eadfe5529/.system_generated/tasks/task-514.log')

logger.info('Monitoring Epoch 1 completion...')

while True:
    # 1. Check if checkpoint_epoch_1.pt or finished log line exists
    ckpt1 = output_dir / 'checkpoint_epoch_1.pt'
    best_pt = output_dir / 'best_model.pt'
    best_hf = output_dir / 'best_model_hf'
    
    epoch1_done = False
    if ckpt1.exists() and ckpt1.stat().st_size > 100_000_000:
        epoch1_done = True
    elif log_file.exists():
        try:
            content = log_file.read_text(errors='ignore')
            if 'Epoch 1/' in content and 'Finished' in content:
                epoch1_done = True
        except Exception:
            pass

    if epoch1_done:
        logger.info('=== Epoch 1 Completed! Initiating stop & cleanup ===')
        
        # Place stop flag to gracefully signal training loop
        stop_flag = output_dir / 'stop_after_epoch.flag'
        try:
            stop_flag.touch()
        except Exception:
            pass

        # Give it 10 seconds to finish validation & writing state
        time.sleep(10)

        # Ensure root checkpoints/ directory has links/copies
        root_ckpt = Path('checkpoints')
        if best_pt.exists():
            try:
                shutil.copy2(best_pt, root_ckpt / 'best_model.pt')
                logger.info(f'Copied {best_pt} -> checkpoints/best_model.pt')
            except Exception as e:
                logger.warning(f'Could not copy best_model.pt: {e}')
                
        if best_hf.is_dir():
            try:
                dest_hf = root_ckpt / 'best_model_hf'
                if dest_hf.exists():
                    shutil.rmtree(dest_hf)
                shutil.copytree(best_hf, dest_hf)
                logger.info(f'Copied {best_hf} -> checkpoints/best_model_hf')
            except Exception as e:
                logger.warning(f'Could not copy best_model_hf: {e}')

        logger.info('Epoch 1 prep & cleanup complete!')
        break

    time.sleep(30)
