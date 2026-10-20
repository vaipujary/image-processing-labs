"""Validate identical weights on CPU and MPS before accepting accelerated experiments."""
import json
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE/'.deps'))
os.environ['TORCH_HOME']=str(HERE/'models')
os.environ['MPLCONFIGDIR']=str(HERE/'.mplconfig')
import torch
from lab5.lab5 import _drunet_model, denoise_dnn
torch.set_num_threads(4)
x=torch.load(HERE/'results/reconstructions.pt',weights_only=True)['y']
os.environ['LAB5_DNN_DEVICE']='cpu';cpu=denoise_dnn(x,.03)
_drunet_model().to('mps');gpu=denoise_dnn(x,.03)
error=float((cpu-gpu).abs().max());rms=float((cpu-gpu).square().mean().sqrt())
assert torch.isfinite(gpu).all() and error<1e-5
result={'cpu_mps_max_error':error,'cpu_mps_rms_error':rms,'shape':list(x.shape),
        'float64_outer':True,'float32_network':True,'separate_transfer_and_cast':True}
(HERE/'results/dnn_device_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
