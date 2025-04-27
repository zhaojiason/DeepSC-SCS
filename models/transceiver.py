# -*- coding: utf-8 -*-
"""
Transformer includes:
    Encoder
        1. Positional coding
        2. Multihead-attention
        3. PositionwiseFeedForward
    Decoder
        1. Positional coding
        2. Multihead-attention
        3. Multihead-attention
        4. PositionwiseFeedForward
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
import reedsolo

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) #math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x
  
class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)
        
        #        query, key, value = \
        #            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.num_heads * self.d_k)
             
        x = self.dense(x)
        x = self.dropout(x)
        
        return x
    
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        #print(mask.shape)
        if mask is not None:
            # 根据mask，指定位置填充 -1e9  
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim = -1)
        return torch.matmul(p_attn, value), p_attn
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x

    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        
        return x
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        #self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        #m = memory
        
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)
        
        src_output = self.src_mha(x, memory, memory, trg_padding_mask) # q, k, v
        x = self.layernorm2(x + src_output)
        
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x

    
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, num_layers, src_vocab_size, max_len, 
                 d_model, num_heads, dff, dropout = 0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout) 
                                            for _ in range(num_layers)])
        
    def forward(self, x, src_mask):
        "Pass the input (and mask) through each layer in turn."
        # the input size of x is [batch_size, seq_len]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)
        
        return x
        


class Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size, max_len, 
                 d_model, num_heads, dff, dropout = 0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout) 
                                            for _ in range(num_layers)])
    
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)
            
        return x


class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()
        
        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        # self.linear4 = nn.Linear(size1, d_model)
        
        self.layernorm = nn.LayerNorm(size1, eps=1e-6)
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        
        output = self.layernorm(x1 + x5)

        return output
        
class DeepSC(nn.Module):
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len, 
                 trg_max_len, d_model, num_heads, dff, dropout = 0.1):
        super(DeepSC, self).__init__()
        
        self.encoder = Encoder(num_layers, src_vocab_size, src_max_len, 
                               d_model, num_heads, dff, dropout)
        
        self.channel_encoder = nn.Sequential(nn.Linear(d_model, 256), 
                                             #nn.ELU(inplace=True),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(256, 16))


        self.channel_decoder = ChannelDecoder(16, d_model, 512)
        
        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len, 
                               d_model, num_heads, dff, dropout)
        
        self.dense = nn.Linear(d_model, trg_vocab_size)
import heapq
import pickle
from collections import Counter, namedtuple

# 用于构造 Huffman 树的节点
HuffmanNode = namedtuple("HuffmanNode", ["freq", "symbol", "left", "right"])

class HuffmanCoding:
    def __init__(self):
        self.code_map = {}
        self.reverse_map = {}

    def build_tree(self, data: bytes):
        # 1) 统计所有符号（byte）的频率
        freq = Counter(data)
        # 2) 初始化小顶堆，heap 中每个元素都是一个 Leaf 节点
        heap = [HuffmanNode(f, s, None, None) for s, f in freq.items()]
        heapq.heapify(heap)

        # 3) 合并最小的两个节点，直到只剩一个根节点
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(left.freq + right.freq, None, left, right)
            heapq.heappush(heap, merged)

        return heap[0]

    def _build_codes(self, node, path=""):
        if node is None:
            return
        # 如果是叶子节点，记录下当前的编码路径
        if node.symbol is not None:
            self.code_map[node.symbol] = path
            self.reverse_map[path] = node.symbol
        else:
            self._build_codes(node.left,  path + "0")
            self._build_codes(node.right, path + "1")

    def fit(self, data: bytes):
        """
        生成 Huffman 树，并建立好 code_map / reverse_map。
        在第一次 encode 之前务必先调用。
        """
        root = self.build_tree(data)
        self._build_codes(root)

    def encode(self, data: bytes) -> bytes:
        """
        返回一个包含：
         [4-byte 树长度][pickled code_map][1-byte pad_len][压缩后的 payload]
        """
        # 序列化 code_map，用于 decode 时重建
        tree_bytes = pickle.dumps(self.code_map)
        tree_len   = len(tree_bytes).to_bytes(4, "big")

        # 将 data 中每个字节替换成 Huffman code
        bitstr = "".join(self.code_map[b] for b in data)
        # 补齐到整字节
        pad_len = (8 - len(bitstr) % 8) % 8
        bitstr += "0" * pad_len
        payload = int(bitstr, 2).to_bytes(len(bitstr)//8, "big")

        return tree_len + tree_bytes + bytes([pad_len]) + payload

    def decode(self, encoded: bytes) -> bytes:
        """
        反向 restore：先读树信息，再读 payload，最后按 code_map 解码。
        """
        tree_len = int.from_bytes(encoded[:4], "big")
        tree_bytes = encoded[4:4+tree_len]
        pad_len = encoded[4+tree_len]
        payload = encoded[5+tree_len:]

        # 重建编码表
        self.code_map = pickle.loads(tree_bytes)
        self.reverse_map = {v:k for k,v in self.code_map.items()}

        # 提取 bit 字符串
        bitstr = bin(int.from_bytes(payload, "big"))[2:].zfill(len(payload)*8)
        if pad_len:
            bitstr = bitstr[:-pad_len]

        # 按前缀码逐位解码
        decoded = bytearray()
        cur = ""
        for bit in bitstr:
            cur += bit
            if cur in self.reverse_map:
                decoded.append(self.reverse_map[cur])
                cur = ""
        return bytes(decoded)

# --------------------------------------------------
# 1) 自定义 straight-through Autograd Function
# --------------------------------------------------
class _ChannelCodecST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, nsym: int) -> torch.Tensor:
        """
        在 forward 中完整执行 Huffman + RS 编解码，模拟信道，
        但不破坏梯度图（detach 后在 backward 中透传）。
        """
        # 转为 numpy bytes
        arr = features.detach().cpu().numpy().astype(np.float32)
        raw = arr.tobytes()

        # Huffman 编码
        h = HuffmanCoding()
        h.fit(raw)
        comp = h.encode(raw)

        # RS 编码
        rs = reedsolo.RSCodec(nsym)
        coded = rs.encode(comp)

        # RS 解码 + Huffman 解码
        dec, _, _ = rs.decode(coded)
        raw2 = h.decode(dec)

        # 重构 tensor
        arr2 = np.frombuffer(raw2, dtype=np.float32).reshape(*features.shape)
        out = torch.from_numpy(arr2).to(features.device)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_nsym):
        """
        straight-through：梯度原样透传给 features，nsym 不参与反向
        """
        return grad_out, None


# --------------------------------------------------
# 2) ChannelCodec Module
# --------------------------------------------------
class ChannelCodec(nn.Module):
    """
    把上面的 ST-Function 包装成 nn.Module。
    """
    def __init__(self, nsym: int):
        super(ChannelCodec, self).__init__()
        self.nsym = nsym

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, T, feature_dim]
        return _ChannelCodecST.apply(features, self.nsym)


# --------------------------------------------------
# 3) 带 Huffman+RS 的 DeepSC 主模型
# --------------------------------------------------
class Huffman_RS(nn.Module):
    def __init__(self,
                 num_layers:    int,
                 src_vocab_size:int,
                 trg_vocab_size:int,
                 src_max_len:   int,
                 trg_max_len:   int,
                 d_model:       int,
                 num_heads:     int,
                 dff:           int,
                 dropout:      float = 0.1,
                 rs_nsym:       int   = 32):
        super(Huffman_RS, self).__init__()
        # 原 Encoder
        self.encoder = Encoder(num_layers,
                               src_vocab_size,
                               src_max_len,
                               d_model,
                               num_heads,
                               dff,
                               dropout)

        # 原 Channel Encoder → 输出维度 16
        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16)
        )

        # Huffman + RS Codec 插件
        self.codec = ChannelCodec(rs_nsym)

        # 原 Channel Decoder
        self.channel_decoder = ChannelDecoder(16, d_model, 512)

        # 原 Decoder + 最后线性
        self.decoder = Decoder(num_layers,
                               trg_vocab_size,
                               trg_max_len,
                               d_model,
                               num_heads,
                               dff,
                               dropout)
        self.dense = nn.Linear(d_model, trg_vocab_size)

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask:           torch.Tensor = None,
                look_ahead_mask:    torch.Tensor = None,
                tgt_padding_mask:   torch.Tensor = None) -> torch.Tensor:
        # 1) 标准编码器
        enc_out = self.encoder(src, src_mask)           # [B, T_src, d_model]

        # 2) Channel Encoder → features
        feats = self.channel_encoder(enc_out)           # [B, T_src, 16]

        # 3) Huffman + RS 模拟信道（自带 encode+decode）
        recon_feats = self.codec(feats)                 # [B, T_src, 16]

        # 4) Channel Decoder
        dec_in = self.channel_decoder(recon_feats)      # [B, T_src, d_model]

        # 5) 标准解码器
        dec_out = self.decoder(dec_in,
                               enc_out,
                               look_ahead_mask,
                               tgt_padding_mask)        # [B, T_trg, d_model]

        # 6) 输出 logits
        logits = self.dense(dec_out)                    # [B, T_trg, trg_vocab_size]
        return logits
    
        
        
        
        
        

    

    
    
    
    
    


    


