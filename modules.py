from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class TGATLayer(nn.Module, ABC):
    """ Token-Token GAT Layer (propagate through dependency parsing trees) """

    def __init__(self, in_dim, feat_embed_size, out_dim, num_heads):
        super(TGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.feat_fc = nn.Linear(feat_embed_size, out_dim * num_heads, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)
        self.num_heads = num_heads
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.feat_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        feat_emb = self.feat_fc(edges.data['dep_emb'])
        num_tokens, emb_size = feat_emb.size()
        feat_emb = feat_emb.reshape([num_tokens, self.num_heads, emb_size // self.num_heads])
        z2 = torch.cat([edges.src['z'], edges.dst['z'], feat_emb], dim=2)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, edge_type=None):
        z = self.fc(h)
        num_tokens, emb_size = z.size()
        z = z.reshape([num_tokens, self.num_heads, emb_size // self.num_heads])
        token_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 0)
        # print(token_node_ids.size())
        # print(z.size())
        tt_edge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 0)
        g.nodes[token_node_ids].data['z'] = z
        g.apply_edges(self.edge_attention, edges=tt_edge_id)
        g.pull(token_node_ids, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('h')
        # h = h.permute(1, 0, 2)
        return h[token_node_ids]


class TAGATLayer(nn.Module, ABC):
    """ Token-Argument GATLayer """

    def __init__(self, in_dim, feat_embed_size, out_dim, num_heads):
        super(TAGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.num_heads = num_heads
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=2)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, edge_type=None):
        z = self.fc(h)
        num_tokens, emb_size = z.size()
        z = z.reshape([num_tokens, self.num_heads, emb_size // self.num_heads])
        token_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 0)
        arg_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 2)
        ta_edge_id = g.filter_edges(lambda edges: (edges.src['dtype'] == 0) & (edges.dst['dtype'] == 2))
        g.nodes[token_node_ids].data['z'] = z
        g.apply_edges(self.edge_attention, edges=ta_edge_id)
        g.pull(arg_node_ids, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('h')
        return h[arg_node_ids]


class ATGATLayer(nn.Module, ABC):
    """ Argument-Token GATLayer """

    def __init__(self, in_dim, feat_embed_size, out_dim, num_heads):
        super(ATGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.num_heads = num_heads
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=2)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, edge_type=None):
        z = self.fc(h)
        num_tokens, emb_size = z.size()
        z = z.reshape([num_tokens, self.num_heads, emb_size // self.num_heads])
        token_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 0)
        arg_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 2)
        at_edge_id = g.filter_edges(lambda edges: (edges.src['dtype'] == 2) & (edges.dst['dtype'] == 0))
        g.nodes[arg_node_ids].data['z'] = z
        g.apply_edges(self.edge_attention, edges=at_edge_id)
        g.pull(token_node_ids, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('h')
        return h[token_node_ids]


class APGATLayer(nn.Module, ABC):
    """ Argument-Predicate GATLayer """

    def __init__(self, in_dim, feat_embed_size, out_dim, num_heads):
        super(APGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.feat_fc = nn.Linear(feat_embed_size, out_dim * num_heads, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)
        self.num_heads = num_heads
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.feat_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        feat_emb = self.feat_fc(edges.data['srl_emb'])
        num_tokens, emb_size = feat_emb.size()
        feat_emb = feat_emb.reshape([num_tokens, self.num_heads, emb_size // self.num_heads])
        z2 = torch.cat([edges.src['z'], edges.dst['z'], feat_emb], dim=2)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, edge_type=None):
        z = self.fc(h)
        num_tokens, emb_size = z.size()
        z = z.reshape([num_tokens, self.num_heads, emb_size // self.num_heads])
        arg_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 2)
        pred_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
        ap_edge_id = g.filter_edges(lambda edges: (edges.src['dtype'] == 2) & (edges.dst['dtype'] == 1))
        g.nodes[arg_node_ids].data['z'] = z
        g.apply_edges(self.edge_attention, edges=ap_edge_id)
        g.pull(pred_node_ids, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('h')
        return h[pred_node_ids]


class PAGATLayer(nn.Module, ABC):
    """ Predicate-Argument GATLayer """

    def __init__(self, in_dim, feat_embed_size, out_dim, num_heads):
        super(PAGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.feat_fc = nn.Linear(feat_embed_size, out_dim * num_heads, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)
        self.num_heads = num_heads
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.feat_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        feat_emb = self.feat_fc(edges.data['srl_emb'])
        num_tokens, emb_size = feat_emb.size()
        feat_emb = feat_emb.reshape([num_tokens, self.num_heads, emb_size // self.num_heads])
        z2 = torch.cat([edges.src['z'], edges.dst['z'], feat_emb], dim=2)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, edge_type=None):
        z = self.fc(h)
        num_tokens, emb_size = z.size()
        z = z.reshape([num_tokens, self.num_heads, emb_size // self.num_heads])
        arg_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 2)
        pred_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
        pa_edge_id = g.filter_edges(lambda edges: (edges.src['dtype'] == 1) & (edges.dst['dtype'] == 1))
        g.nodes[pred_node_ids].data['z'] = z
        g.apply_edges(self.edge_attention, edges=pa_edge_id)
        g.pull(arg_node_ids, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('h')
        return h[arg_node_ids]


class MultiHeadGATLayer(nn.Module, ABC):
    def __init__(self, layer, in_size, out_size, feat_embed_size, num_heads, config, merge='cat', layer_norm_eps=1e-12):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        out_dim = out_size // num_heads
        self.layer = layer(in_size, feat_embed_size, out_dim, num_heads)
        self.merge = merge
        self.dropout = nn.Dropout(p=config['dep_dropout_rate'])

    def forward(self, g, o, h, edge_type=None):
        head_outs = self.layer(g, self.dropout(h), edge_type)
        num_tokens = head_outs.size()[0]
        if self.merge == 'cat':
            out = head_outs.reshape([num_tokens, -1])
        else:
            out = torch.mean(head_outs, dim=1)
        out = o + F.elu(out)
        return out
