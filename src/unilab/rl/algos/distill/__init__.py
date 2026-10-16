"""Internal single-teacher behavior distillation and persistent DAgger.

Models, atomic checkpoints and spawn/shared-weight lifecycle are adapted from
Andisyc/ReGait@06c12ee605795daa14817c9db7644ba75a453075. The generic
single-teacher slice keeps its detached action targets and mse/smooth-L1/L1
losses; task construction is injected through ``EnvFactory``.
"""
