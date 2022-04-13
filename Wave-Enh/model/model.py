import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out), nn.LeakyReLU(negative_slope=0.1))

    def forward(self, ip):
        return self.main(ip)


class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                                            stride=stride, padding=padding),
                                  nn.BatchNorm1d(channel_out), nn.LeakyReLU(negative_slope=0.1))

    def forward(self, ip):
        return self.main(ip)


EPS = 1e-8


class _LayerNorm(nn.Module):
    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    def forward(self, x):
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())


class CumLN(_LayerNorm):
    def forward(self, x):
        batch, chan, spec_len = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=-1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=-1)
        cnt = torch.arange(start=chan, end=chan * (spec_len + 1), step=chan, dtype=x.dtye).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = cum_pow_sum - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / (cum_var + EPS).sqrt())


class FeatsGlobLN(_LayerNorm):
    def forward(self, x):
        stop = len(x.size())
        dims = list(range(2, stop))
        mean = torch.mean(x, dim=dims, keepdim=True)
        var = torch.var(x, dim=dims, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())

class Conv1DBlock(nn.Module):
    def __init__(self, in_chan, hid_chan, kernel_size, padding, dilation):
        super(Conv1DBlock, self).__init__()
        conv_norm = GlobLN
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation, groups=hid_chan)
        self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(),conv_norm(hid_chan), depth_conv1d, nn.PReLU(),
                                          conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x):
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        return res_out


class IE_classifier(nn.Module):
    def __init__(self, in_chan=1, n_src=1, out_chan=(31, ), n_blocks=5, n_repeats=2, bn_chan=64, hid_chan=128,
                 kernel_size=3,):
        super(IE_classifier, self).__init__()
        self.inchan = in_chan
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
        self.IE_classifier = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2 ** x // 2
                self.IE_classifier.append(Conv1DBlock(bn_chan, hid_chan, kernel_size, padding=padding, dilation=2 ** x))
        #out_conv = nn.Linear(bn_chan, n_src * out_chan)
        #self.out = nn.Sequential(nn.PReLU(), out_conv)
        self.out = nn.ModuleList()
        for o in out_chan:
            out_conv = nn.Linear(bn_chan, n_src * o)
            self.out.append(nn.Sequential(nn.PReLU(), out_conv))

    def forward(self, mixture_w):
        output = self.bottleneck(mixture_w)
        for i in range(len(self.IE_classifier)):
            residual = self.IE_classifier[i](output)
            output = output + residual

        logits = [out(output.mean(-1)) for out in self.out]
        return tuple(logits)

    def get_config(self):
        config = {'in_chan': self.inchan,
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




class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)


print('Welcome to Old Model')
class SEModel(nn.Module):
    def __init__(self, n_layers=12, channels_interval=24):
        super(SEModel, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input):
        tmp = []
        o = input

        # Up Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            #print(o.shape)
            
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]
        #print(o.shape)

        o = self.middle(o)
        #print(o.shape)

        # Down Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            #print(o.shape)
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            #print(o.shape)
            # Skip Connection
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)
            #print(o.shape)

        o = torch.cat([o, input], dim=1)
        o = self.out(o)
        return o











if __name__ == "__main__":
    inp = torch.rand(2, 1, 600)
    m = IE_classifier(out_chan=(31,))
    o = m(inp)
    print(o, type(o))
