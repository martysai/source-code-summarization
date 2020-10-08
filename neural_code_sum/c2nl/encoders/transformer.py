"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from c2nl.modules.util_class import LayerNorm
from c2nl.modules.multi_head_attn import MultiHeadedAttention
from c2nl.modules.position_ffn import PositionwiseFeedForward
from c2nl.encoders.encoder import EncoderBase
from c2nl.utils.misc import sequence_mask, get_rel_mask


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self,
                 d_model,
                 heads,
                 d_ff,
                 d_k,
                 d_v,
                 dropout,
                 max_relative_positions=0,
                 use_neg_dist=True,
                 use_tree_relative_attn=False,\
                 tree_rel_vocab_size=0):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadedAttention(heads,
                                              d_model,
                                              d_k,
                                              d_v,
                                              dropout=dropout,
                                              max_relative_positions=max_relative_positions,
                                              use_neg_dist=use_neg_dist,
                                               use_tree_relative_attn=\
                                              use_tree_relative_attn,\
                                              tree_rel_vocab_size=\
                                              tree_rel_vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask, rel_matrix=None, rel_mask=None):
        """
        Transformer Encoder Layer definition.
        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        context, attn_per_head, _ = self.attention(inputs, inputs, inputs,
                                                   mask=mask, rel_matrix=rel_matrix,\
                                                   rel_mask=rel_mask,\
                                                   attn_type="self")
        out = self.layer_norm(self.dropout(context) + inputs)
        return self.feed_forward(out), attn_per_head


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    Returns:
        (`FloatTensor`, `FloatTensor`):
        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self,
                 num_layers,
                 d_model=512,
                 heads=8,
                 d_k=64,
                 d_v=64,
                 d_ff=2048,
                 dropout=0.2,
                 max_relative_positions=0,
                 use_neg_dist=True,
                 use_tree_relative_attn=False,
                 tree_rel_vocab_size=0,\
                 always_add_type_emb=False,\
                 type_vocab_size=0):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        if isinstance(max_relative_positions, int):
            max_relative_positions = [max_relative_positions] * self.num_layers
        assert len(max_relative_positions) == self.num_layers

        self.layer = nn.ModuleList(
            [TransformerEncoderLayer(d_model,
                                     heads,
                                     d_ff,
                                     d_k,
                                     d_v,
                                     dropout,
                                     max_relative_positions=max_relative_positions[i],
                                     use_neg_dist=use_neg_dist,
                                     use_tree_relative_attn=use_tree_relative_attn,\
                                     tree_rel_vocab_size=tree_rel_vocab_size)
             for i in range(num_layers)])
        
        self.always_add_type_emb = always_add_type_emb
        if always_add_type_emb:
            self.type_embeddings = nn.ModuleList([nn.Embedding(type_vocab_size,
                                   d_model) for i in range(num_layers)])

    def count_parameters(self):
        params = list(self.layer.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self, src, lengths=None, rel_matrix=None, src_type=None):
        """
        Args:
            src (`FloatTensor`): `[batch_size x src_len x model_dim]`
            lengths (`LongTensor`): length of each sequence `[batch]`
            rel_matrix (`LongTensor`): [batch_size x src_len x src_len]`
            src_type (`LongTensor`): [batch_size x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        self._check_args(src, lengths)

        out = src
        mask = None if lengths is None else \
            ~sequence_mask(lengths, out.shape[1]).unsqueeze(1)
        rel_mask = None if rel_matrix is None else \
            get_rel_mask(lengths, out.shape[1])
        # Run the forward pass of every layer of the tranformer.
        representations = []
        attention_scores = []
        for i in range(self.num_layers):
            if self.always_add_type_emb and i > 0:
                out = out + self.type_embeddings[i](src_type)
            out, attn_per_head = self.layer[i](out, mask, rel_matrix, rel_mask)
            representations.append(out)
            attention_scores.append(attn_per_head)

        return representations, attention_scores
