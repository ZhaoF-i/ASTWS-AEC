import torch.nn as nn
import torch.fft
import torch
from einops import rearrange
from nnstruct.multiply_orders import *
import math


class CFB(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super(CFB, self).__init__()
        self.conv_gate = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1,
                                   padding=(0, 0), dilation=1, groups=1, bias=True)
        self.conv_input = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1,
                                    padding=(0, 0), dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), stride=1,
                              padding=(1, 0), dilation=1, groups=1, bias=True)
        self.ceps_unit = CepsUnit(ch=out_channels)
        self.LN0 = LayerNorm(in_channels, f=160)
        self.LN1 = LayerNorm(out_channels, f=160)
        self.LN2 = LayerNorm(out_channels, f=160)

    def forward(self, x):
        g = torch.sigmoid(self.conv_gate(self.LN0(x)))
        x = self.conv_input(x)
        y = self.conv(self.LN1(g * x))
        y = y + self.ceps_unit(self.LN2((1 - g) * x))
        return y


class CepsUnit(nn.Module):
    def __init__(self, ch):
        super(CepsUnit, self).__init__()
        self.ch = ch
        self.ch_lstm_f = CH_LSTM_F(ch * 2, ch, ch * 2)
        self.LN = LayerNorm(ch * 2, f=81)

    def forward(self, x0):
        x0 = torch.fft.rfft(x0, 160, 2)
        x = torch.cat([x0.real, x0.imag], 1)
        x = self.ch_lstm_f(self.LN(x))
        x = x[:, :self.ch] + 1j * x[:, self.ch:]
        x = x * x0
        x = torch.fft.irfft(x, 160, 2)
        return x


class LayerNorm(nn.Module):
    def __init__(self, c, f):
        super(LayerNorm, self).__init__()
        self.w = nn.Parameter(torch.ones(1, c, f, 1))
        self.b = nn.Parameter(torch.rand(1, c, f, 1) * 1e-4)

    def forward(self, x):
        mean = x.mean([1, 2], keepdim=True)
        std = x.std([1, 2], keepdim=True)
        x = (x - mean) / (std + 1e-8) * self.w + self.b
        return x


class WienerAttention(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.feature_dim = order
        self.frequence_dim = 160
        self.q_conv = nn.Conv2d(in_channels=2, out_channels=2*(self.feature_dim+1), kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels=2, out_channels=2*self.feature_dim, kernel_size=1)
        self.v_linear = nn.Linear(self.feature_dim+1, 1)
        self.query_vector = nn.Parameter(torch.zeros(self.frequence_dim, ), requires_grad=True)
        self.key_vector = nn.Parameter(torch.zeros(self.feature_dim, ), requires_grad=True)
        self.value_vector = nn.Parameter(torch.zeros(self.feature_dim), requires_grad=True)
        self.query_linear = nn.Linear(self.frequence_dim, self.frequence_dim)
        self.query_linear = nn.Linear(self.frequence_dim, self.frequence_dim)
        self.key_linear = nn.Linear(self.feature_dim, self.feature_dim)
        self.query_norm = nn.LayerNorm(self.frequence_dim)
        self.key_norm = nn.LayerNorm(self.feature_dim)
        self.value_linear_sig = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_linear_tan = nn.Linear(self.feature_dim, self.feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, far_complex, mix_complex):
        k = self.feature_dim
        # shape: B, C, F, T
        # far_complex = torch.complex(far[:, 0], far[:, 1])
        # mix_complex = torch.complex(mix[:, 0], mix[:, 1])
        far_complex_padded = torch.nn.functional.pad(far_complex, (k - 1, 0))
        # 执行 unfold 操作，设置 size 参数为 (F, 4)，即 T+3 维度展开为大小为 4 的窗口
        far_complex_unfolded = far_complex_padded.unfold(3, k, 1)
        far_complex_unfolded_save = torch.complex(far_complex_unfolded[:,0], far_complex_unfolded[:,1])
        b,c,f,t,k = far_complex_unfolded.shape

        far_complex_unfolded[:, 1] *= -1
        X_transpose = torch.transpose(far_complex_unfolded.unsqueeze(-2), -1, -2)
        XTX = torch.matmul(X_transpose, far_complex_unfolded.unsqueeze(-2))
        XTY = torch.matmul(X_transpose, mix_complex.unsqueeze(-1).unsqueeze(-1))

        query = far_complex_unfolded.permute(0,1,3,4,2)
        key = self.k_conv(mix_complex).view(b,c,self.feature_dim,f,t).contiguous().permute(0,1,4,3,2)
        value = torch.cat([XTX, XTY], dim=-1).contiguous().transpose(-1,-2)

        query = self.query_linear(query)
        query = self.query_norm(query)
        q_vector = self.sigmoid(self.query_vector)
        query = query * q_vector

        key = self.key_linear(key)
        key = self.key_norm(key)
        k_vector = self.sigmoid(self.key_vector)
        key = (key * k_vector)

        v_vector = self.sigmoid(self.value_vector)
        value = value * v_vector

        weight = query.matmul(key / math.sqrt(self.feature_dim))
        weight = self.softmax(weight)

        out = value.matmul(weight.unsqueeze(dim=2))
        XTX_new = out.contiguous().transpose(-1,-2)[..., 0:self.feature_dim]
        XTY_new = out.contiguous().transpose(-1,-2)[..., self.feature_dim:]
        XTX_new = torch.complex(XTX_new[:,0], XTX_new[:,1])
        XTY_new = torch.complex(XTY_new[:,0], XTY_new[:,1])

        # inver_XTX = XTX_new # torch.linalg.pinv(XTX_new)
        eps = 1e-8
        eye = torch.view_as_complex(torch.stack([torch.eye(XTX_new.shape[-1])+eps, torch.eye(XTX_new.shape[-1])+eps],dim=-1)).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
        inver_XTX = torch.linalg.inv(XTX_new+eye)
        WienerSolution = torch.matmul(inver_XTX, XTY_new).squeeze(-1)
        out = far_complex_unfolded_save*WienerSolution
        out = torch.sum(out, dim=-1)
        out = torch.stack([out.real, out.imag], dim=1)

        return out


class NET(nn.Module):
    def __init__(self, order=10, channels=20):
        super().__init__()
        self.act = nn.ELU()
        self.n_fft = 319
        self.hop_length = 160
        self.window = torch.hamming_window(self.n_fft)
        self.order = order

        self.in_ch_lstm = CH_LSTM_F(6, channels, channels)
        self.in_conv = nn.Conv2d(in_channels=6+channels, out_channels=channels, kernel_size=(1, 1))
        self.cfb_e1 = CFB(channels+2, channels)

        self.ln = LayerNorm(channels, 160)
        self.ch_lstm = CH_LSTM_T(in_ch=channels, feat_ch=channels * 2, out_ch=channels, num_layers=2)

        self.cfb_d1 = CFB(1 * channels, channels)

        self.out_ch_lstm = CH_LSTM_T(2 * channels, channels, channels * 2)
        self.out_conv = nn.Conv2d(in_channels=channels * 3, out_channels=2, kernel_size=(1, 1),
                                  padding=(0, 0), bias=True)
        self.wiener_attenton = WienerAttention(20)
        # self.linear = nn.Linear(self.order*2, self.order)

    def stft(self, x):
        b, m, t = x.shape[0], x.shape[1], x.shape[2],
        x = x.reshape(-1, t)
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(x.device))
        F, T = X.shape[1], X.shape[2]
        X = X.reshape(b, m, F, T, 2)
        X = torch.cat([X[..., 0], X[..., 1]], dim=1)
        return X

    def istft(self, Y, t):
        b, c, F, T = Y.shape
        m_out = int(c // 2)
        Y_r = Y[:, :m_out]
        Y_i = Y[:, m_out:]
        Y = torch.stack([Y_r, Y_i], dim=-1)
        Y = Y.reshape(-1, F, T, 2)
        y = torch.istft(Y, n_fft=self.n_fft, hop_length=self.hop_length, length=t, window=self.window.to(Y.device))
        y = y.reshape(b, m_out, y.shape[-1])
        return y

    # def wiener_solution(self, far, mix, k):
    #     # shape: B, C, F, T
    #     far_complex = torch.complex(far[:, 0], far[:, 1])
    #     mix_complex = torch.complex(mix[:, 0], mix[:, 1])
    #     far_complex_padded = torch.nn.functional.pad(far_complex, (k - 1, 0))
    #     # 执行 unfold 操作，设置 size 参数为 (F, 4)，即 T+3 维度展开为大小为 4 的窗口
    #     far_complex_unfolded = far_complex_padded.unfold(2, k, 1)
    #
    #     X_transpose = torch.transpose(torch.conj(far_complex_unfolded.unsqueeze(-2)), -1, -2)
    #     XTX = torch.matmul(X_transpose, far_complex_unfolded.unsqueeze(-2))
    #     XTY = torch.matmul(X_transpose, mix_complex.unsqueeze(-1).unsqueeze(-1))
    #     # inver_XTX = XTX # torch.linalg.pinv(XTX)
    #     inver_XTX = torch.linalg.pinv(XTX)
    #     WienerSolution = torch.matmul(inver_XTX, XTY).squeeze(-1)
    #     WienerSolution = torch.stack([WienerSolution.real, WienerSolution.imag], dim=1)
    #
    #     return WienerSolution.permute(0, 1, 4, 2, 3)

    def forward(self, x):
        # x:[batch, channel, frequency, time]
        X0 = self.stft(x)
        mix_comp = torch.stack([X0[:, 0], X0[:, 2]], dim=1)
        far_comp = torch.stack([X0[:, 1], X0[:, 3]], dim=1)
        b, c, f, t = far_comp.shape
        out_wiener_attention = self.wiener_attenton(far_comp, mix_comp)

        e0 = self.in_ch_lstm(torch.cat([X0, out_wiener_attention], 1))
        e0 = self.in_conv(torch.cat([e0, torch.cat([X0, out_wiener_attention], 1)], 1))
        e1 = self.cfb_e1(torch.cat([e0, out_wiener_attention], 1))

        lstm_out = self.ch_lstm(self.ln(e1))

        d1 = self.cfb_d1(torch.cat([e1 * lstm_out], dim=1))

        d0 = self.out_ch_lstm(torch.cat([e0, d1], dim=1))
        out = self.out_conv(torch.cat([d0, d1], dim=1))
        # b, c, f, t = Y.shape
        # estEchoPath = Y.reshape(Y.shape[0], 2, self.order, Y.shape[2], Y.shape[3])
        # estEchoPath = self.linear(torch.cat([estEchoPath, Wiener_Solution], dim=2).permute(0,1,3,4,2)).permute(0,1,4,2,3)
        # out = mix_comp - multiply_orders_(far_comp, estEchoPath, self.order)

        y = self.istft(out, t=x.shape[-1])[:, 0]
        # far = self.istft(far_comp, t=x.shape[-1])[:, 0]

        return y


class CH_LSTM_T(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=False, num_layers=1):
        super().__init__()
        self.lstm2 = nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
        self.bi = 1 if bi == False else 2
        self.linear = nn.Linear(self.bi * feat_ch, out_ch)
        self.out_ch = out_ch

    def forward(self, x):
        self.lstm2.flatten_parameters()
        b, c, f, t = x.shape
        x = rearrange(x, 'b c f t -> (b f) t c')
        x, _ = self.lstm2(x.float())
        x = self.linear(x)
        x = rearrange(x, '(b f) t c -> b c f t', b=b, f=f, t=t)
        return x


class CH_LSTM_F(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch, bi=True, num_layers=1):
        super().__init__()
        self.lstm2 = nn.LSTM(in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bi)
        self.linear = nn.Linear(2 * feat_ch, out_ch)
        self.out_ch = out_ch

    def forward(self, x):
        self.lstm2.flatten_parameters()
        b, c, f, t = x.shape
        x = rearrange(x, 'b c f t -> (b t) f c')
        x, _ = self.lstm2(x.float())
        x = self.linear(x)
        x = rearrange(x, '(b t) f c -> b c f t', b=b, f=f, t=t)
        return x


def complexity():
    # inputs = torch.randn(1,1,16000)
    model = NET().cuda()
    # output = model(inputs)
    # print(output.shape)

    from ptflops import get_model_complexity_info
    mac, param = get_model_complexity_info(model, (2, 16000), as_strings=True, print_per_layer_stat=True, verbose=True)
    print(mac, param)
    '''
    963.38 MMac 148.87 k
    '''


if __name__ == '__main__':
    complexity()


