import torch
from Network.ICCRN_1layer import NET as NET
import soundfile as sf

net = NET()
save_name = '/home/imu_tzf1/ddn/zf/result/wiener_attention/newDataSet/attention11-try2_order20/checkpoints/model.ckpt'
net.load_state_dict(torch.load('/home/imu_tzf1/ddn/zf/result/wiener_attention/newDataSet/attention11-try2_order20/checkpoints/best.ckpt').state_dict)
torch.save(net.state_dict(), save_name)

net = NET()
net.load_state_dict(torch.load(save_name))

mic, _ = sf.read('')
farend, _ = sf.read('')

mic = torch.Tensor(mic).unsqueeze(dim=0)
farend = torch.Tensor(farend).unsqueeze(dim=0)

with torch.no_grad():
    est_time = net(torch.stack([mic, farend], dim=1))

print(est_time.shape)
