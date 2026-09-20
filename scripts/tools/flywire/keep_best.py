"""
RETIRED. Checkpoint promotion now lives in the learner.

This script watched tensorboard and, on seeing a fast eval lap, copied whatever weights
happened to be on disk. That was unsound twice over. The lap had been driven minutes and
hundreds of batches earlier, so the file it saved was never the network that drove the time;
and it scored only laps that finished, so a policy that crashed almost immediately could be
promoted on one fluke. Following it overwrote the one genuinely good fine-tuned policy and
recorded three checkpoints that crash at 4.5% of the track.

The replacement is in trackmania_rl/multiprocess/learner_process.py: every eval lap enters a
rolling window, laps that did not finish count at the cutoff, and a checkpoint is written --
from the live network, at that moment -- only when the window's MEDIAN beats the incumbent by
checkpoint_promote_margin_ms. See checkpoint_eval_window in config.py.

Running this alongside training would advance save/_anchor to unverified weights and fight the
learner for the same files, so it now refuses to run.
"""

import sys

print(__doc__)
sys.exit(1)
