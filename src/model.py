import math
import torch
from torch import nn
import logging


### REF: Dive into Deep Learning, https://d2l.ai
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.alpha_mult = nn.Parameter(torch.tensor([1.0]))

    def forward(self, queries, keys, values, num_heads=0, valid_lens=None, q_mult=None, kv_mult=None, after_softmax=True):
        # Shape of queries: (batch_size, no. of queries, d)
        # Shape of keys: (batch_size, no. of key-value pairs, d)
        # Shape of values: (batch_size, no. of key-value pairs, value dimension)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        logging.debug(">>> DotProductAttention.forward ")
        logging.debug(f"Shape of queries: {queries.shape}")
        logging.debug(f"Shape of keys: {keys.shape}")
        logging.debug(f"Shape of values: {values.shape}")
        
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        # Shape of score: (batch_size * num_heads, no. of queries, no. of key-value paris)
        scores = torch.bmm(queries, keys.transpose(1, 2))/ math.sqrt(d)
        # Shape of k_mult: (batch_size, no. of queries, 1)
        # Shape of kv_mult: (batch_size, no. of kv paris, 1)
        mult_ = None
        if q_mult is not None and kv_mult is not None:
            mult_ = torch.bmm(q_mult, kv_mult.transpose(1, 2))
            frobenius_norms = torch.norm(mult_, p='fro', dim=[1,2], keepdim=True)
            epsilon = 1e-5
            mult_ = mult_ / (frobenius_norms + epsilon)
            if num_heads:
                mult_ = torch.repeat_interleave(mult_, num_heads, dim=0)
            if not after_softmax:
                scores = torch.add(scores, mult_)
        if after_softmax and mult_ is not None: 
            self.attention_weights = torch.add(masked_softmax(scores, valid_lens), mult_)
        else:
            self.attention_weights = masked_softmax(scores, valid_lens)
        # if not self.training:
            # logging.info(f"alpha_mult={self.alpha_mult}")
        logging.debug("DotProductAttention <<<<<<<<<<<")
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=use_bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=use_bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=use_bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=use_bias)    

    def forward(self, q, k, v, valid_lens=None, q_mult=None, kv_mult=None, q_ln=None, kv_ln=None, alpha_mult=None):
        # Shape of q, k, or v:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output q, k, or v:
        # (batch_size * num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)
        # Shape of output: (batch_size * num_heads, no. of queries, num_hiddens / num_heads)
        q = self.W_q(q)
        if q_ln:
            q = q_ln(q)
        q = self.transpose_qkv(q)

        k = self.W_k(k)
        v = self.W_v(v)
        if kv_ln:
            k = kv_ln(k)
            v = kv_ln(v)
        k = self.transpose_qkv(k)
        v = self.transpose_qkv(v)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
        
        if alpha_mult is not None and kv_mult is not None:
            kv_mult = kv_mult * alpha_mult
        output = self.attention(q, k, v, self.num_heads, valid_lens, q_mult, kv_mult)

        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # Shape of output: (batch_size * num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    # @d2l.add_to_class(MultiHeadAttention)  #@save
    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


class PositionWiseFFN(nn.Module):  # @save
    """The positionwise feed-forward network."""

    def __init__(self, ffn_num_hiddens, ffn_num_outputs, use_bias=False):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens, bias=use_bias)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs, bias=use_bias)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class MAB(nn.Module):
    """Multiset Multihead Attention Block"""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False, pre_ln=False, num_out=None):
        # d_q, d_kv only required if pre_ln is True
        super().__init__()
        self.pre_ln = pre_ln
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        if pre_ln:
            self.ln0 = nn.LayerNorm(num_hiddens)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(num_hiddens)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(num_hiddens)
        self.alpha_mult = torch.nn.Parameter(torch.tensor([1.0]))
        self.num_out = num_out
        if self.num_out is not None:
            self.fc = nn.LazyLinear(num_out, bias=use_bias)

    def forward(self, Q, X, valid_lens=None, q_mult=None, kv_mult=None, use_alpha=True):
        alpha_mult = self.alpha_mult if use_alpha else None
        if not self.pre_ln:
            A = self.attention(Q, X, X, valid_lens, q_mult, kv_mult, alpha_mult=alpha_mult)
            H = self.ln1(self.dropout1(A) + self.attention.W_q(Q))
            O = self.ln2(self.dropout2(self.ffn(H)) + H)
        else:
            A = self.attention(Q, X, X, valid_lens, q_mult, kv_mult, q_ln=self.ln0, kv_ln=self.ln1, alpha_mult=alpha_mult)
            H = self.attention.W_q(Q) + self.dropout1(A)
            O = H + self.dropout2(self.ffn(self.ln2(H)))
        return O if self.num_out is None else self.fc(O)

class SAB(nn.Module):
    """Multiset Self Attention Block, Permutation Equivalence"""

    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False, pre_ln=False):
        super().__init__()
        self.mab = MAB(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias, pre_ln)

    def forward(self, X, valid_lens=None, X_mult=None):
        return self.mab(X, X, valid_lens, X_mult, X_mult)


class MAB_Q(nn.Module):
    """Multiset Multihead Attention Block with Learnable Queries"""

    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False, pre_ln=False, num_queries=1, num_out=None):
        super().__init__()
        self.q = nn.Parameter(torch.Tensor(1, num_queries, num_hiddens))
        self.q_mult = nn.Parameter(torch.Tensor(1, num_queries, 1))
        nn.init.xavier_uniform_(self.q)
        nn.init.xavier_uniform_(self.q_mult)
        self.mab = MAB(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias, pre_ln, num_out=num_out)

    def forward(self, X, valid_lens=None, kv_mult=None):
        return self.mab(self.q.repeat(X.size(0), 1, 1).to(X.device), X, valid_lens, 
                        self.q_mult.repeat(X.size(0), 1, 1).to(X.device), kv_mult, use_alpha=False)

class ISAB(nn.Module):
    """Induced multiSet Attention Block"""

    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, num_inds, dropout, use_bias=False, pre_ln=False):
        super().__init__()
        self.mab_q = MAB_Q(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias, pre_ln, num_inds)
        self.mab = MAB(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias, pre_ln)

    def forward(self, X, valid_lens=None, X_mult=None):
        H = self.mab_q(X, valid_lens, X_mult)
        return self.mab(X, H, valid_lens, X_mult, torch.ones(H.size(0), H.size(1), 1).to(X.device))


class Transformer(nn.Module):
    """Multiset Transformer"""
    def __init__(self, num_out, num_hiddens, ffn_num_hiddens, num_heads, num_inds, 
                 dropout, use_bias=False, pre_ln=False, equiv='sab', num_equiv=2, 
                 num_queries=1, pooling='MAB_Q'):
        super().__init__()
        self.equiv = equiv
        if equiv.lower() == 'sab':
            self.equivs = nn.ModuleList([SAB(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias, pre_ln)
                                    for _ in range(num_equiv)])
        elif equiv.lower() == 'isab':
            self.equivs = nn.ModuleList([ISAB(num_hiddens, ffn_num_hiddens, num_heads, num_inds, dropout, use_bias, pre_ln)
                                    for _ in range(num_equiv)])
        else:
            self.equivs = nn.ModuleList([nn.Sequential(
                nn.LazyLinear(num_hiddens, bias=use_bias),
                nn.ReLU(),
                nn.LazyLinear(num_hiddens, bias=use_bias),
                nn.ReLU()
            ) for _ in range(num_equiv)])
        self.pooling = pooling.lower()
        if self.pooling == 'mab_q':
            self._pooling = MAB_Q(num_hiddens, num_out, num_heads, dropout, use_bias, pre_ln, num_queries, num_out=num_out)

    def forward(self, X, valid_lens=None, X_mult=None, mult_in_equiv=False):
        for E_ in self.equivs:
            if self.equiv.lower() in ['sab', 'isab']:
                if mult_in_equiv:
                    X = E_(X, valid_lens, X_mult)
                else:
                    X = E_(X, valid_lens, X_mult=None)
            else:
                X = E_(X)
        if self.pooling == 'sum':
            logging.debug(X_mult.shape)
            if X_mult is not None:
                X = torch.cat((X, X_mult), dim=2)
            X = X.sum(dim=1)
            X = X.reshape(X.size(0), -1)
        else:
            X = self._pooling(X, valid_lens, X_mult)
            X = X.reshape(X.size(0), -1)
