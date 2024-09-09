import torch
from ASTWS import NET 
import soundfile as sf

net = NET()
net.load_state_dict(torch.load(model.ckpt))

mic, _ = sf.read('')
farend, _ = sf.read('')

mic = torch.Tensor(mic).unsqueeze(dim=0)
farend = torch.Tensor(farend).unsqueeze(dim=0)

with torch.no_grad():
    est_time = net(torch.stack([mic, farend], dim=1))

print(est_time.shape)
