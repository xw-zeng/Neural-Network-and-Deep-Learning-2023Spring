from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from .model import AttModel
from torch.nn.parameter import Parameter
import pdb


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.min_value = -1e8
        # self.batch_norm = nn.BatchNorm1d(self.rnn_size)

    def forward(self, h, att_feats, p_att_feats):
        # The p_att_feats here is already projected
        batch_size = h.size(0)
        att_size = att_feats.numel() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        # dot = F.dropout(dot, 0.3, training=self.training)
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size
        weight = F.softmax(dot, dim=1)  # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        # att_res = self.batch_norm(att_res)

        return att_res


class Attention2(nn.Module):
    def __init__(self, opt):
        super(Attention2, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.min_value = -1e8
        # self.batch_norm = nn.BatchNorm1d(self.rnn_size)

    def forward(self, h, att_feats, p_att_feats, mask):
        # The p_att_feats here is already projected
        batch_size = h.size(0)
        att_size = att_feats.numel() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.sigmoid(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        # dot = F.dropout(dot, 0.3, training=self.training)
        hAflat = self.alpha_net(dot)  # (batch * att_size) * 1
        hAflat = hAflat.view(-1, att_size)  # batch * att_size
        hAflat.masked_fill_(mask.bool(), self.min_value)
        weight = F.softmax(hAflat, dim=1)  # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        # att_res = self.batch_norm(att_res)

        return att_res


class Attention1(nn.Module):
    def __init__(self, opt):
        super(Attention1, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.h2att = nn.Linear(self.rnn_size, 1024)
        self.h2att2 = nn.Linear(1024, 1024)
        self.h2att3 = nn.Linear(1024, 512)
        self.att2att = nn.Linear(512, self.att_hid_size)
        # self.alpha_net =  nn.Linear(self.att_hid_size, self.att_hid_size)
        self.alpha_net1 = nn.Linear(self.att_hid_size, 1)
        self.min_value = -1e8
        # self.batch_norm = nn.BatchNorm1d(self.rnn_size)

    def forward(self, h, att_feats, p_att_feats):
        # The p_att_feats here is already projected
        batch_size = h.size(0)
        att_size = att_feats.numel() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = (self.h2att(h))
        # att_h = torch.sigmoid(att_h)
        att_h = (self.h2att2(att_h))
        att_h = (self.h2att3(att_h))
        # att_h = torch.sigmoid(att_h)
        att_h = (self.att2att(att_h))
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        # dot = F.dropout(dot, 0.3, training=self.training)
        # dot = self.alpha_net(dot)
        # dot = torch.tanh(dot)
        dot = self.alpha_net1(dot)
        # dot = torch.tanh(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size
        weight = F.softmax(dot, dim=1)  # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        # att_res = self.batch_norm(att_res)

        return att_res


class Attention3(nn.Module):
    def __init__(self, opt):
        super(Attention3, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.h2att = nn.Linear(self.rnn_size, 1024)
        self.h2att2 = nn.Linear(1024, 1024)
        self.h2att3 = nn.Linear(1024, 512)
        self.att2att = nn.Linear(512, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.min_value = -1e8

    def forward(self, h, att_feats, p_att_feats, mask):
        # The p_att_feats here is already projected
        batch_size = h.size(0)
        att_size = att_feats.numel() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = (self.h2att(h))
        att_h = (self.h2att2(att_h))
        att_h = (self.h2att3(att_h))
        att_h = (self.att2att(att_h))  # batch * att_hid_size)
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        hAflat = self.alpha_net(dot)

        hAflat = hAflat.view(-1, att_size)  # batch * att_size
        hAflat.masked_fill_(mask.bool(), self.min_value)
        weight = F.softmax(hAflat, dim=1)  # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        # att_res = self.batch_norm(att_res)

        return att_res


class adaPnt(nn.Module):
    def __init__(self, conv_size, rnn_size, att_hid_size, dropout, min_value, beta):
        super(adaPnt, self).__init__()
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.att_hid_size = att_hid_size
        self.min_value = min_value
        self.conv_size = conv_size

        # fake region embed
        self.f_fc1 = nn.Linear(self.rnn_size, self.rnn_size)
        self.f_fc2 = nn.Linear(self.rnn_size, self.att_hid_size)
        # h out embed
        self.h_fc1 = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.inplace = False
        self.beta = beta

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed, mask):
        # batch_size = h_out.size(0)
        # View into three dimensions
        # conv_feat = conv_feat.view(batch_size, -1, self.conv_size)
        roi_num = conv_feat_embed.size(1)
        conv_feat_embed = conv_feat_embed.view(-1, roi_num, self.att_hid_size)
        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = F.relu(self.f_fc1(fake_region.view(-1, self.rnn_size)), inplace=self.inplace)
        fake_region_embed = self.f_fc2(fake_region)
        # fake_region_embed = self.f_fc1(fake_region.view(-1, self.rnn_size))
        h_out_embed = self.h_fc1(h_out)
        # img_all = torch.cat([fake_region.view(-1,1,self.conv_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1, 1, self.att_hid_size), conv_feat_embed], 1)
        hA = torch.tanh(img_all_embed + h_out_embed.view(-1, 1, self.att_hid_size))
        # hA = F.dropout(hA, 0.3, self.training)
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        hAflat = hAflat.view(-1, roi_num + 1)
        hAflat.masked_fill_(mask.bool(), self.min_value)
        # hAflat= F.softmax(hAflat, dim=1)
        # det_prob = hAflat
        return hAflat


class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.min_value = -1e8

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size)  # we, fc, h^2_t-1

        # self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.attention2 = Attention2(opt)

        self.adaPnt = adaPnt(opt.input_encoding_size, opt.rnn_size, opt.att_hid_size, self.drop_prob_lm, self.min_value,
                             opt.beta)
        self.i2h_2 = nn.Linear(opt.rnn_size * 2, opt.rnn_size)
        self.h2h_2 = nn.Linear(opt.rnn_size, opt.rnn_size)

    def forward(self, xt, fc_feats, conv_feats, p_conv_feats, pool_feats, p_pool_feats, att_mask, pnt_mask, state):
        # prev_h = state[0][-1]
        # att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
        att_lstm_input = torch.cat([fc_feats, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))
        att = self.attention(h_att, conv_feats, p_conv_feats)
        att2 = self.attention2(h_att, pool_feats, p_pool_feats, att_mask[:, 1:])
        lang_lstm_input = torch.cat([att + att2, h_att], 1)

        ada_gate_point = torch.sigmoid(self.i2h_2(lang_lstm_input) + self.h2h_2(state[0][1]))
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        fake_box = F.dropout(ada_gate_point * torch.tanh(state[1][1]), self.drop_prob_lm, training=self.training)
        det_prob = self.adaPnt(output, fake_box, pool_feats, p_pool_feats, pnt_mask)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        return output, det_prob, state


################################################################################################################################################
class NewTopDownCore(nn.Module):
    """
    This module was inspired by the original TopDownCore module by Jiasen Lu. et al.
    We designed a cascade LSTM network that feeds the context vector and hidden vector
    from the frist attention LSTM to first language lstm in the next layer. Respectively we feed the context and hidden vectors
    from the second attention LSTM that is present in the same layer as first attention LSTM, to the second language LSTM that is present in the same layer that first language LSTM is present at.
    """

    def __init__(self, opt, use_maxout=False):
        super(NewTopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.min_value = -1e8
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size)  # we, fc, h^2_t-1
        self.att_lstm2 = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size)  # we, fc, h^2_t-1
        # self.att_lstm3 = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size)
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.lang_lstm2 = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.lang_lstm3 = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        # self.lang_lstm4 = nn.LSTMCell(opt.rnn_size*2, opt.rnn_size)
        # self.lang_lstm5 = nn.LSTMCell(opt.rnn_size*2, opt.rnn_size)
        # self.lang_lstm6 = nn.LSTMCell(opt.rnn_size*2, opt.rnn_size)
        self.attention = Attention(opt)
        self.attention2 = Attention2(opt)
        self.adaPnt = adaPnt(opt.input_encoding_size, opt.rnn_size, opt.att_hid_size, self.drop_prob_lm, self.min_value,
                             opt.beta)
        # self.i2h_2 = nn.Linear(opt.rnn_size*2, opt.rnn_size)
        # self.i2h_1 = nn.Linear(opt.rnn_size*2, opt.rnn_size)
        # self.i2h_3 = nn.Linear(opt.rnn_size*2, opt.rnn_size)
        self.i2h_4 = nn.Linear(opt.rnn_size * 2, opt.rnn_size)
        # self.h2h_2 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.h2h_1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        # self.h2h_3 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.h2h_4 = nn.Linear(opt.rnn_size, opt.rnn_size)
        # self.h2h_5 = nn.Linear(opt.rnn_size, opt.rnn_size)
        # self.h2h_6 = nn.Linear(opt.rnn_size, opt.rnn_size)
        # self.h2h_7 = nn.Linear(opt.rnn_size, opt.rnn_size)

    def forward(self, xt, fc_feats, conv_feats, p_conv_feats, pool_feats, p_pool_feats, att_mask, pnt_mask, state):
        att_lstm_input = torch.cat([fc_feats, xt], 1)
        h_att1, c_att1 = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))
        h_att2, c_att2 = self.att_lstm2(att_lstm_input, (state[0][1], state[1][1]))  ##
        # h_att3, c_att3 = self.att_lstm3(att_lstm_input, (h_att2, c_att2))
        # att = self.attention(h_att3, conv_feats, p_conv_feats)
        # att2 = self.attention2(h_att3, pool_feats, p_pool_feats, att_mask[:,1:])
        # lstm3_in = torch.cat([att+att2, h_att3], 1)
        # lstm3_in1 = torch.cat([att+att2, c_att3], 1)
        ########################################################################
        att = self.attention(h_att1, conv_feats, p_conv_feats)
        att2 = self.attention2(h_att1, pool_feats, p_pool_feats, att_mask[:, 1:])
        # att1 = self.attention(h_att2, conv_feats, p_conv_feats)
        # att12 = self.attention2(h_att2, pool_feats, p_pool_feats, att_mask[:,1:])
        ########################################################################
        # lstm2_in = torch.cat([att1+att12, h_att2], 1)
        # lstm2_in1 = torch.cat([att1+att12, c_att2], 1)
        ########################################################################
        lstm1_in = torch.cat([att + att2, h_att1], 1)
        # lstm1_in1 = torch.cat([att1+att12, c_att1], 1)
        ########################################################################
        att_gate1 = torch.sigmoid(self.h2h_1(h_att1 + c_att1))  ##
        ################APPLY LSTM
        h_lang, c_lang = self.lang_lstm(lstm1_in, (h_att1, att_gate1 * (c_att1)))  ##
        # h_lang = torch.sigmoid(self.h2h_1(h_lang))
        att = self.attention(h_att2, conv_feats, p_conv_feats)
        att2 = self.attention2(h_att2, pool_feats, p_pool_feats, att_mask[:, 1:])
        lstm2_in = torch.cat([att + att2, h_att2], 1)
        ########################################################################
        att_gate2 = torch.sigmoid(self.h2h_1(h_att2 + c_att2))  ##
        att_gate2 = att_gate2 + att_gate1
        ################APPLY LSTM state[0][1], state[1][1]
        h1_lang, c1_lang = self.lang_lstm2(lstm2_in, (h_att2, att_gate2 * (c_att2)))
        # h1_lang = torch.sigmoid(self.h2h_6(h1_lang)) * c1_lang * att_gate2 ##
        ########################################################################
        h_lang_t = torch.add(h_lang, h1_lang)
        c_lang_t = torch.add(c1_lang, c_lang)

        att_gate3 = torch.sigmoid(self.h2h_1(h_lang_t + c_lang_t))  ##
        att_gate3 = att_gate2 + att_gate3
        ################APPLY LSTM
        # mix1 = torch.mul(lstm1_in, lstm2_in)
        # att = self.attention(h_lang_t, conv_feats, p_conv_feats)
        # att2 = self.attention2(h_lang_t, pool_feats, p_pool_feats, att_mask[:,1:])
        # lstm1_in = torch.cat([att+att2, h_lang_t], 1)
        h2_lang, c2_lang = self.lang_lstm3(lstm1_in + lstm2_in, (h_lang_t, att_gate3 * c_lang_t))
        # h2_lang = torch.sigmoid(self.h2h_7(h2_lang* c2_lang * att_gate3))

        # c_lang_t = c1_lang + c_lang + c2_lang/3
        # h_lang_t = h1_lang + h_lang + h2_lang/3
        # c_lang_t = self.h2h_2(c2_lang)
        # h_lang_t = self.h2h_3(h2_lang)
        # att_gate4 = torch.tanh(self.i2h_1(lstm1_in) + self.h2h_1(c2_lang))
        # att_gate4 = att_gate4 + att_gate3
        # h3_lang, c3_lang = self.lang_lstm4(lstm1_in, (h2_lang,att_gate4*c2_lang))
        # att_gate5 = torch.sigmoid(self.i2h_1(lstm1_in1) + self.h2h_1(c3_lang))
        # h4_lang, c4_lang = self.lang_lstm5(lstm1_in, (h3_lang, att_gate5*c3_lang))
        # att_gate6 = torch.sigmoid(self.i2h_1(lstm1_in1) + self.h2h_1(c4_lang))
        # h5_lang, c5_lang = self.lang_lstm6(lstm1_in, (h4_lang, att_gate6*c4_lang))

        # att_gate7 = torch.sigmoid(self.i2h_1(lstm1_in1) + self.h2h_1(c5_lang))
        # h6_lang, c6_lang = self.lang_lstm7(lstm1_in, (h5_lang, att_gate7*c5_lang))
        # att_gate8 = torch.sigmoid(self.i2h_1(lstm1_in1) + self.h2h_1(c6_lang)
        # h7_lang, c7_lang = self.lang_lstm8(lstm1_in, (h6_lang, att_gate8*c6_lang))

        # att_gate9 = torch.sigmoid(self.i2h_1(lstm1_in1) + self.h2h_1(c6_lang)
        # h8_lang, c8_lang = self.lang_lstm9(lstm1_in, (h7_lang, att_gate9*c7_lang))

        # att_gate10 = torch.sigmoid(self.i2h_1(lstm1_in1) + self.h2h_1(c8_lang)
        # h9_lang, c9_lang = self.lang_lstm10(lstm1_in, (h8_lang, att_gate10*c8_lang))

        # h_att1 += h_att2
        # c_att1 *= c_att2
        ada_gate_point = torch.sigmoid(self.i2h_4(lstm1_in + lstm2_in) + self.h2h_4(h2_lang))
        output0 = F.dropout(h_lang, 0.3, self.training)
        # output0 = torch.tanh(self.h2h_5(output0))
        output1 = F.dropout(h1_lang, 0.7, self.training)
        # output1 = torch.tanh(self.h2h_6(output1))
        output2 = F.dropout(h2_lang, 0.8, self.training)
        # output2 = torch.tanh(self.h2h_7(output2))
        # output3 = F.dropout(h3_lang, 0.5, self.training)
        # output0 = F.dropout(att_gate1*h_lang, 0.1, self.training)
        # output1 = F.dropout(att_gate2*h1_lang, 0.1, self.training)
        # output2 = F.dropout(att_gate3*h2_lang, 0.1, self.training)
        # output = output0+output1+output2
        # output = torch.tanh(self.h2h_1(output))
        # output = torch.tanh(self.h2h_2(output))
        # output = torch.tanh(self.h2h_3(output))
        # output = torch.tanh(self.h2h_4(output))
        output = output1 + output2 + output0
        # output = (self.h2h_5(output))
        output = F.dropout(output, 0.5, self.training)
        # output = output2
        # output = torch.tanh(output)
        # output /= 3
        # output = F.dropout(h_lang_t, 0.1, self.training)
        fake_box = F.dropout(ada_gate_point * torch.tanh(c2_lang), 0.5, training=self.training)
        # fake_box = F.relu(ada_gate_point*torch.tanh(c_lang_t))
        det_prob = self.adaPnt(output, fake_box, pool_feats, p_pool_feats, pnt_mask)
        # state = (torch.stack([h_att1, h_att2, h_att3, h_lang, h1_lang, h2_lang, h4_lang, h5_lang]), torch.stack([c_att1, c_att2, c_att3, c_lang, c1_lang, c2_lang, c4_lang, c5_lang]))
        state = (torch.stack([h_att1, h_att2]), torch.stack([c_att1, c_att2]))

        return output, det_prob, state


################################################################################################################################################################################################################

class Att2in2Core(nn.Module):
    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        # self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size
        self.min_value = -1e8

        self.adaPnt = adaPnt(opt.input_encoding_size, opt.rnn_size, opt.att_hid_size, self.drop_prob_lm, self.min_value,
                             opt.beta)

        # Build a LSTM
        self.a2c1 = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.a2c2 = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 6 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 6 * self.rnn_size)
        self.dropout1 = nn.Dropout(self.drop_prob_lm)
        self.dropout2 = nn.Dropout(self.drop_prob_lm)
        self.attention = Attention(opt)
        self.attention2 = Attention2(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, pool_feats, p_pool_feats, att_mask, pnt_mask, state):
        att_res1 = self.attention(state[0][-1], att_feats, p_att_feats)
        att_res2 = self.attention2(state[0][-1], pool_feats, p_pool_feats, att_mask[:, 1:])

        # xt_input = torch.cat([fc_feats, xt], 1)
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 4 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        s_gate = sigmoid_chunk.narrow(1, self.rnn_size * 3, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 4 * self.rnn_size, 2 * self.rnn_size) + \
                       self.a2c1(att_res1) + self.a2c2(att_res2)

        in_transform = torch.max(
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)
        fake_box = s_gate * torch.tanh(next_c)

        output = self.dropout1(next_h)
        fake_box = self.dropout2(fake_box)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        det_prob = self.adaPnt(output, fake_box, pool_feats, p_pool_feats, pnt_mask)
        return output, det_prob, state


class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt)
        self.ccr_core = CascadeCore(opt)


class NewTopDownModel(AttModel):
    """
    This module was inspired by the original TopDownCore module by Jiasen Lu. et al.
    We designed a cascade LSTM network that feeds the context vector and hidden vector
    from the frist attention LSTM to first language lstm in the next layer. Respectively we feed the context and hidden vectors
    from the second attention LSTM that is present in the same layer as first attention LSTM, to the second language LSTM that is present in the same layer that first language LSTM is present at.
    """

    def __init__(self, opt):
        super(NewTopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = NewTopDownCore(opt)
        self.ccr_core = CascadeCore(opt)


class Att2in2Model(AttModel):
    def __init__(self, opt):
        super(Att2in2Model, self).__init__(opt)
        self.num_layers = 1
        self.core = Att2in2Core(opt)
        self.ccr_core = CascadeCore(opt)


class CascadeCore(nn.Module):
    def __init__(self, opt):
        super(CascadeCore, self).__init__()
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.fg_size = opt.fg_size
        self.fg_size = opt.fg_size

        self.bn_fc = nn.Sequential(
            nn.Linear(opt.rnn_size + opt.rnn_size, opt.rnn_size),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(opt.rnn_size, 2))

        self.fg_fc = nn.Sequential(
            nn.Linear(opt.rnn_size + opt.rnn_size, opt.rnn_size),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(opt.rnn_size, 300))

        # initialize the fine-grained glove embedding.
        self.fg_emb = Parameter(opt.glove_fg)
        self.fg_emb.requires_grad_()

        # setting the fg mask for the cascadeCore.
        self.fg_mask = Parameter(opt.fg_mask.byte(), requires_grad=False)
        # self.fg_mask.requires_grad_()
        self.min_value = -1e8
        self.beta = opt.beta

    def forward(self, fg_idx, pool_feats, rnn_outs, roi_labels, seq_batch_size, seq_cnt):

        roi_num = pool_feats.size(1)
        pool_feats = pool_feats.view(seq_batch_size, 1, roi_num, self.rnn_size) * \
                     roi_labels.view(seq_batch_size, seq_cnt, roi_num, 1)

        # get the average of the feature. # size:  seq_batch_size, seq_cnt, rnn_size.
        pool_cnt = roi_labels.sum(2)
        pool_cnt[pool_cnt == 0] = 1
        pool_feats = pool_feats.sum(2) / pool_cnt.view(seq_batch_size, seq_cnt, 1)

        # concate with the rnn_output feature.
        pool_feats = torch.cat((rnn_outs, pool_feats), 2)
        bn_logprob = F.log_softmax(self.bn_fc(pool_feats), dim=2)

        fg_out = self.fg_fc(pool_feats)
        # construct the mask for finegrain classification.
        # fg_out 

        fg_score = torch.mm(fg_out.view(-1, 300), self.fg_emb.t()).view(seq_batch_size, -1, self.fg_size + 1)

        fg_mask = self.fg_mask[fg_idx]
        fg_score.masked_fill_(fg_mask.view_as(fg_score).bool(), self.min_value)
        fg_logprob = F.log_softmax(fg_score, dim=2)

        return bn_logprob, fg_logprob
