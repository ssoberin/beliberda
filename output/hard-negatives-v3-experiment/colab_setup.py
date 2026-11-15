import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','-q','sentence-transformers'],check=True)
import torch
print('GPU',torch.cuda.get_device_name(0),flush=True)
assert 'T4' in torch.cuda.get_device_name(0)
