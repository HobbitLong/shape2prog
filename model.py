from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from misc import render_block


class BlockOuterNet(nn.Module):
    """
    predict block-level programs and parameters
    block-LSTM
    """
    def __init__(self, opt):
        super(BlockOuterNet, self).__init__()

        self.feat_size = opt.shape_feat_size
        self.input_size = opt.outer_input_size
        self.rnn_size = opt.outer_rnn_size
        self.num_layers = opt.outer_num_layers
        self.drop_prob = opt.outer_drop_prob
        self.seq_length = opt.outer_seq_length
        self.is_cuda = opt.is_cuda

        self.shape_feat = Conv3DNet(self.feat_size, input_channel=2, power=2)
        self.core = nn.LSTM(self.input_size,
                            self.rnn_size,
                            self.num_layers,
                            bias=False,
                            dropout=self.drop_prob)
        self.inner_net = BlockInnerNet(opt)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))

    def forward(self, x, y, sample_prob=None):
        batch_size = x.size(0)
        state = self.init_hidden(batch_size)

        outputs_pgm = []
        outputs_param = []

        rendered_shapes = np.zeros((batch_size, 32, 32, 32), dtype=np.uint8)

        def combine(x, y):
            y = torch.from_numpy(np.copy(y).astype(np.float32))
            y = Variable(torch.unsqueeze(y, 1), requires_grad=False)
            if self.is_cuda:
                y = y.cuda()
            return torch.cat([x, y], dim=1)

        fc_feats = self.shape_feat(combine(x, rendered_shapes))

        seq = y

        for i in range(seq.size(1)):
            if i == 0:
                xt = fc_feats
            else:
                prob_pre = torch.exp(outputs_pgm[-1].data)
                _, it1 = torch.max(prob_pre, dim=2)
                it2 = outputs_param[-1].data.clone()
                it1 = it1.cpu().numpy()
                it2 = it2.cpu().numpy()
                rendered_shapes = render_block(rendered_shapes, it1, it2)
                xt = self.shape_feat(combine(x, rendered_shapes))

            output, state = self.core(xt.unsqueeze(0), state)
            output = F.relu(output)
            pgm, param = self.inner_net(output.squeeze(), y[:, i], sample_prob)

            outputs_pgm.append(pgm)
            outputs_param.append(param)

        return [torch.cat([_.unsqueeze(1) for _ in outputs_pgm], 1).contiguous(),
                torch.cat([_.unsqueeze(1) for _ in outputs_param], 1).contiguous()]

    def decode(self, x):
        batch_size = x.size(0)
        state = self.init_hidden(batch_size)

        outputs_pgm = []
        outputs_param = []

        rendered_shapes = np.zeros((batch_size, 32, 32, 32), dtype=np.uint8)

        def combine(x, y):
            y = torch.from_numpy(np.copy(y).astype(np.float32))
            y = Variable(torch.unsqueeze(y, 1), requires_grad=False)
            if self.is_cuda:
                y = y.cuda()
            return torch.cat([x, y], dim=1)

        fc_feats = self.shape_feat(combine(x, rendered_shapes))

        for i in range(self.seq_length):
            if i == 0:
                xt = fc_feats
            else:
                prob_pre = torch.exp(outputs_pgm[-1].data)
                _, it1 = torch.max(prob_pre, dim=2)
                it2 = outputs_param[-1].data.clone()
                it1 = it1.cpu().numpy()
                it2 = it2.cpu().numpy()
                rendered_shapes = render_block(rendered_shapes, it1, it2)
                xt = self.shape_feat(combine(x, rendered_shapes))

            output, state = self.core(xt.unsqueeze(0), state)
            output = F.relu(output)
            pgm, param = self.inner_net.decode(output.squeeze())

            outputs_pgm.append(pgm)
            outputs_param.append(param)

        return [torch.cat([_.unsqueeze(1) for _ in outputs_pgm], 1).contiguous(),
                torch.cat([_.unsqueeze(1) for _ in outputs_param], 1).contiguous()]


class BlockInnerNet(nn.Module):
    """
    Inner Block Net
    use last pgm as input for each time step
    step-LSTM
    """
    def __init__(self, opt):
        super(BlockInnerNet, self).__init__()

        self.vocab_size = opt.program_size
        self.max_param = opt.max_param
        self.input_size = opt.inner_input_size
        self.rnn_size = opt.inner_rnn_size
        self.num_layers = opt.inner_num_layers
        self.drop_prob = opt.inner_drop_prob
        self.seq_length = opt.inner_seq_length
        self.cls_feat_size = opt.inner_cls_feat_size
        self.reg_feat_size = opt.inner_reg_feat_size
        self.sample_prob = opt.inner_sample_prob
        self.is_cuda = opt.is_cuda

        self.pgm_embed = nn.Embedding(self.vocab_size + 1, self.input_size)
        self.core = nn.LSTM(self.input_size,
                            self.rnn_size,
                            self.num_layers,
                            bias=False,
                            dropout=self.drop_prob)
        self.logit1 = nn.Linear(self.rnn_size, self.cls_feat_size)
        self.logit2 = nn.Linear(self.cls_feat_size, self.vocab_size + 1)
        self.regress1 = nn.Linear(self.rnn_size, self.reg_feat_size)
        self.regress2 = nn.Linear(self.vocab_size + 1 + self.reg_feat_size, (self.vocab_size + 1) * self.max_param)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.pgm_embed.weight.data.uniform_(-initrange, initrange)
        self.logit1.bias.data.fill_(0)
        self.logit1.weight.data.uniform_(-initrange, initrange)
        self.logit2.bias.data.fill_(0)
        self.logit2.weight.data.uniform_(-initrange, initrange)
        self.regress1.bias.data.fill_(0)
        self.regress1.weight.data.uniform_(-initrange, initrange)
        self.regress2.bias.data.fill_(0)
        self.regress2.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))

    def forward(self, x, y, sample_prob=None):
        if sample_prob is not None:
            self.sample_prob = sample_prob
        batch_size = x.size(0)
        state = self.init_hidden(batch_size)
        outputs_pgm = []
        outputs_param = []
        seq = y

        for i in range(seq.size(1)):
            if i == 0:
                xt = x
            else:
                if i >= 1 and self.sample_prob > 0:
                    sample_prob = x.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.sample_prob
                    if sample_mask.sum() == 0:
                        it1 = seq[:, i-1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it1 = seq[:, i-1].data.clone()
                        prob_prev = torch.exp(outputs_pgm[-1].data)
                        it1.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                        it1 = Variable(it1, requires_grad=False)
                else:
                    print("obtain last ground truth")
                    it1 = seq[:, i-1].clone()
                xt = self.pgm_embed(it1)

            output, state = self.core(xt.unsqueeze(0), state)

            pgm_feat1 = F.relu(self.logit1(output.squeeze(0)))
            pgm_feat2 = self.logit2(pgm_feat1)
            pgm_score = F.log_softmax(pgm_feat2, dim=1)

            trans_prob = F.softmax(pgm_feat2, dim=1).detach()
            param_feat1 = F.relu(self.regress1(output.squeeze(0)))
            param_feat2 = torch.cat([trans_prob, param_feat1], dim=1)
            param_score = self.regress2(param_feat2)
            param_score = param_score.view(batch_size, self.vocab_size + 1, self.max_param)

            index = seq[:, i].unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.max_param).detach()
            param_score = param_score.gather(1, index).squeeze(1)

            outputs_pgm.append(pgm_score)
            outputs_param.append(param_score)

        return [torch.cat([_.unsqueeze(1) for _ in outputs_pgm], 1).contiguous(),
                torch.cat([_.unsqueeze(1) for _ in outputs_param], 1).contiguous()]

    def decode(self, x):
        batch_size = x.size(0)
        state = self.init_hidden(batch_size)
        outputs_pgm = []
        outputs_param = []

        for i in range(self.seq_length):
            if i == 0:
                xt = x
            else:
                prob_pre = torch.exp(outputs_pgm[-1].data)
                _, it1 = torch.max(prob_pre, dim=1)
                it1 = Variable(it1, requires_grad=False)
                xt = self.pgm_embed(it1)

            output, state = self.core(xt.unsqueeze(0), state)

            pgm_feat1 = F.relu(self.logit1(output.squeeze(0)))
            pgm_feat2 = self.logit2(pgm_feat1)
            pgm_score = F.log_softmax(pgm_feat2, dim=1)

            trans_prob = F.softmax(pgm_feat2, dim=1).detach()
            param_feat1 = F.relu(self.regress1(output.squeeze(0)))
            param_feat2 = torch.cat([trans_prob, param_feat1], dim=1)
            param_score = self.regress2(param_feat2)
            param_score = param_score.view(batch_size, self.vocab_size + 1, self.max_param)

            _, index = torch.max(trans_prob, dim=1)
            index = index.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.max_param).detach()
            param_score = param_score.gather(1, index).squeeze(1)

            outputs_pgm.append(pgm_score)
            outputs_param.append(param_score)

        return [torch.cat([_.unsqueeze(1) for _ in outputs_pgm], 1).contiguous(),
                torch.cat([_.unsqueeze(1) for _ in outputs_param], 1).contiguous()]


class Conv3DNet(nn.Module):
    """
    encode 3D voxelized shape into a vector
    """
    def __init__(self, feat_size, input_channel=1, power=1):
        super(Conv3DNet, self).__init__()

        power = int(power)

        self.conv1 = nn.Conv3d(input_channel, 8*power, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2))
        self.conv2 = nn.Conv3d(8*power, 16*power, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(16*power, 16*power, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(16*power, 32*power, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv5 = nn.Conv3d(32*power, 32*power, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv6 = nn.Conv3d(32*power, 64*power, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv7 = nn.Conv3d(64*power, 64*power, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv8 = nn.Conv3d(64*power, 64*power, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        # self.BN1 = nn.BatchNorm3d(8*power)
        # self.BN2 = nn.BatchNorm3d(16*power)
        # self.BN3 = nn.BatchNorm3d(16*power)
        # self.BN4 = nn.BatchNorm3d(32*power)
        # self.BN5 = nn.BatchNorm3d(32*power)
        # self.BN6 = nn.BatchNorm3d(64*power)
        # self.BN7 = nn.BatchNorm3d(64*power)
        # self.BN8 = nn.BatchNorm3d(64*power)

        self.BN1 = nn.BatchNorm3d(8 * power, track_running_stats=False)
        self.BN2 = nn.BatchNorm3d(16 * power, track_running_stats=False)
        self.BN3 = nn.BatchNorm3d(16 * power, track_running_stats=False)
        self.BN4 = nn.BatchNorm3d(32 * power, track_running_stats=False)
        self.BN5 = nn.BatchNorm3d(32 * power, track_running_stats=False)
        self.BN6 = nn.BatchNorm3d(64 * power, track_running_stats=False)
        self.BN7 = nn.BatchNorm3d(64 * power, track_running_stats=False)
        self.BN8 = nn.BatchNorm3d(64 * power, track_running_stats=False)

        self.avgpool = nn.AvgPool3d(kernel_size=(4, 4, 4))

        self.fc = nn.Linear(64*power, feat_size)

    def forward(self, x):
        x = F.relu(self.BN1(self.conv1(x)))
        x = F.relu(self.BN2(self.conv2(x)))
        x = F.relu(self.BN3(self.conv3(x)))
        x = F.relu(self.BN4(self.conv4(x)))
        x = F.relu(self.BN5(self.conv5(x)))
        x = F.relu(self.BN6(self.conv6(x)))
        x = F.relu(self.BN7(self.conv7(x)))
        x = F.relu(self.BN8(self.conv8(x)))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc(x))

        return x


class RenderNet(nn.Module):
    """
    Multiple Step Render
    """
    def __init__(self, opt):
        super(RenderNet, self).__init__()

        # program LSTM parameter
        self.vocab_size = opt.program_size
        self.max_param = opt.max_param
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.program_vector_size = opt.program_vector_size
        self.nc = opt.nc

        self.pgm_embed = nn.Linear(self.vocab_size + 1, int(self.input_encoding_size / 2))
        self.param_embed = nn.Linear(self.max_param, self.input_encoding_size - int(self.input_encoding_size / 2))
        self.lstm = nn.LSTM(self.input_encoding_size, self.rnn_size, self.num_layers, bias=False,
                            dropout=self.drop_prob_lm)
        self.pgm_param_feat = nn.Linear(self.rnn_size, self.program_vector_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.program_vector_size, 64, 4, 1, 0, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            # 4 x 4 x 4
            nn.ConvTranspose3d(64, 16, 4, 2, 1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            # 8 x 8 x 8
            nn.ConvTranspose3d(16, 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU(True),
            nn.Conv3d(4, 4, 3, 1, 1, bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU(True),
            # 16 x 16 x 16
            nn.ConvTranspose3d(4, self.nc, 4, 2, 1, bias=False),
        )

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))

    def forward(self, program, parameters, index):
        program = program.permute(1, 0, 2)
        parameters = parameters.permute(1, 0, 2)
        bsz = program.size(1)
        init = self.init_hidden(bsz)

        # program linear transform
        dim1 = program.size()
        program = program.contiguous().view(-1, self.vocab_size + 1)
        x1 = F.relu(self.pgm_embed(program))
        x1 = x1.view(dim1[0], dim1[1], -1)

        # parameter linear transform
        dim2 = parameters.size()
        parameters = parameters.contiguous().view(-1, self.max_param)
        x2 = F.relu(self.param_embed(parameters))
        x2 = x2.view(dim2[0], dim2[1], -1)

        # LSTM to aggregate programs and parameters
        x = torch.cat((x1, x2), dim=2)
        out, hidden = self.lstm(x, init)

        # select desired step aggregated features
        index = index.unsqueeze(1).expand(-1, out.size(2)).unsqueeze(0)
        pgm_param_feat = out.gather(dim=0, index=index).squeeze()
        pgm_param_feat = F.relu(self.pgm_param_feat(pgm_param_feat))

        pgm_param_feat = pgm_param_feat.view(bsz, self.program_vector_size, 1, 1, 1)
        shape = self.decoder(pgm_param_feat)

        return shape

    def compute_LSTM_feat(self, program, parameters, index):
        program = program.permute(1, 0, 2)
        parameters = parameters.permute(1, 0, 2)
        bsz = program.size(1)
        init = self.init_hidden(bsz)

        # program linear transform
        dim1 = program.size()
        program = program.contiguous().view(-1, self.vocab_size + 1)
        x1 = F.relu(self.pgm_embed(program))
        x1 = x1.view(dim1[0], dim1[1], -1)

        # parameter linear transform
        dim2 = parameters.size()
        parameters = parameters.contiguous().view(-1, self.max_param)
        x2 = F.relu(self.param_embed(parameters))
        x2 = x2.view(dim2[0], dim2[1], -1)

        # LSTM to aggregate programs and parameters
        x = torch.cat((x1, x2), dim=2)
        out, hidden = self.lstm(x, init)

        # select desired step aggregated features
        index = index.unsqueeze(1).expand(-1, out.size(2)).unsqueeze(0)
        pgm_param_feat = out.gather(dim=0, index=index).squeeze()
        # pgm_param_feat = F.relu(self.pgm_param_feat(pgm_param_feat))

        return pgm_param_feat

    def compute_shape_from_feat(self, pgm_param_feat):
        bsz = pgm_param_feat.size(0)
        pgm_param_feat = F.relu(self.pgm_param_feat(pgm_param_feat))
        pgm_param_feat = pgm_param_feat.view(bsz, self.program_vector_size, 1, 1, 1)
        shape = self.decoder(pgm_param_feat)

        return shape


if __name__ == '__main__':

    from easydict import EasyDict as edict
    from programs.label_config import stop_id, max_param

    opt = edict()
    opt.shape_feat_size = 64

    opt.outer_input_size = 64
    opt.outer_rnn_size = 64
    opt.outer_num_layers = 1
    opt.outer_drop_prob = 0
    opt.outer_seq_length = 6
    opt.is_cuda = False
    opt.program_size = stop_id - 1
    opt.max_param = max_param - 1
    opt.inner_input_size = 64
    opt.inner_rnn_size = 64
    opt.inner_num_layers = 1
    opt.inner_drop_prob = 0
    opt.inner_seq_length = 3
    opt.inner_cls_feat_size = 64
    opt.inner_reg_feat_size = 64
    opt.inner_sample_prob = 1.0

    net = BlockOuterNet(opt)

    x = torch.zeros((10, 1, 32, 32, 32))
    x = Variable(x, requires_grad=False)
    y = torch.zeros((10, 6, 3)).long()
    pgms, params = net(x, y)
    print(pgms.size())
    print(params.size())
    pgms, params = net.decode(x)
    print(pgms.size())
    print(params.size())
