import torch
import torch.nn as nn
import torch.nn.functional as F


class _LayerNorm(nn.Module):

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


EPS = 1e-8


class GlobLN(_LayerNorm):

    def forward(self, x):
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())




class SEModel(nn.Module):
    def __init__(self):
        super(SEModel, self).__init__()
        self.cnn_layers = nn.Sequential(

            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.cnn_layers(x)
        return x





class Conv1DBlock(nn.Module):

    def __init__(self, in_chan, hid_chan, kernel_size, padding,
                 dilation, ):
        super(Conv1DBlock, self).__init__()
        conv_norm = GlobLN
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation, groups=hid_chan)
        self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(), conv_norm(hid_chan), depth_conv1d, nn.PReLU(),
                                          conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x):
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        return res_out


class TCN(nn.Module):
    def __init__(self, in_chan=40, n_src=1, out_chan=(6, 14, 4), n_blocks=5, n_repeats=2, bn_chan=64, hid_chan=128,
                 kernel_size=3, ):
        super(TCN, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size

        layer_norm = GlobLN(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2 ** x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, kernel_size, padding=padding, dilation=2 ** x))

        self.out = nn.ModuleList()
        for o in out_chan:
            out_conv = nn.Linear(bn_chan, n_src * o)
            self.out.append(nn.Sequential(nn.PReLU(), out_conv))

    def forward(self, mixture_w):
        output = self.bottleneck(mixture_w)
        for i in range(len(self.TCN)):
            residual = self.TCN[i](output)
            output = output + residual

        logits = [out(output.mean(-1)) for out in self.out]

        return tuple(logits)

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'bn_chan': self.bn_chan,
            'hid_chan': self.hid_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
        }
        return config
        

if __name__ == "__main__":
    inp = torch.rand(1, 512, 240)
    m = SEModel()
    print(m(inp).shape)
    inp = torch.rand(2, 40, 600)
    m = TCN(out_chan=(31,))
    o = m(inp)
    print(type(o))
    print(o)
