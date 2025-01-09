"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.tril: The elements above the specified diagonal are set to zero
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            # q*k/sqrt(d_k), d_k = hs 
            # (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # (B, nh, T, hs) -> (B, T, hs*nh)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # 模型在处理输入序列时的最大长度
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # 词嵌入层
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # 位置嵌入层
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # 多个Transformer块
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # 将Transformer的输出映射到词汇表大小, 用于生成下一个词的概率分布
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # 词嵌入层的权重与语言模型头的权重绑定在一起, 以减少参数数量并提高模型的效率 ???
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # 如果 non_embedding 为 True, 则从总参数数量中减去位置嵌入层(wpe)的参数数量
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        '''
            输入:
                idx: 形状为 (batch_size, sequence_length)
                targets: 目标序列, 形状为 (batch_size, sequence_length)
            输出:
                logits: (b, t, vocab_size) or (b, 1, vocab_size)
                loss: ???

        '''
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # 将词嵌入 tok_emb 和位置嵌入 pos_emb 相加, 并通过 Dropout 层 drop 进行随机丢弃, 形状为 (b, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # targets: 目标序列, 形状为 (batch_size, sequence_length)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            # 将归一化后的输出 x 通过语言模型头 lm_head 转换为词汇表大小的概率分布, 形状为 (b, t, vocab_size)
            logits = self.lm_head(x)
            # logits.view(-1, logits.size(-1)) 维度是：(batch_size * sequence_length, vocab_size)
            # targets.view(-1)：将目标序列展平为一维张量, 以便与展平后的 logits 进行匹配
            # 计算损失时忽略所有值为 -1 的目标
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # 如果没有提供目标序列(即在推理或生成阶段), 模型通常只需要预测下一个词或字符因此, 在这种情况下, 模型只需要计算最后一个位置的输出, 而不是整个序列的输出
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # 调整位置嵌入层 (wpe) 的权重, 通过切片操作 [:block_size], 将权重截断为新的 block_size, 并重新赋值给 wpe.weight
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        # 调整注意力机制中的偏置 (bias):
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    # classmethod 修饰符对应的函数不需要实例化，不需要 self 参数，但第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        '''
            1 实例化自定义gpt模型
            2 实例化 transformers GPT2LMHeadModel 模型, 加载 Hugging Face 的模型库预训练模型的权重
            3 将预训练权重复制给自定义gpt模型, 如果涉及某些线性层, 需要做个转置
        '''
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        # from_pretrained 方法会从 Hugging Face 的模型库中下载预训练模型的权重和配置文件，并加载到指定的模型类中
        model_type = "./checkpoints/gpt2"
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # 由于 Conv1D 和 Linear 模块的权重矩阵形状不同, 直接加载 Conv1D 的权重到 Linear 模块会导致形状不匹配
        # Conv1D 的权重矩阵形状为 (out_features, in_features)
        # Linear 的权重矩阵形状为 (in_features, out_features)
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        # 遍历预训练模型的状态字典, 将权重复制到当前模型的状态字典中
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        '''
            功能: 在训练过程中调用, 用于初始化优化器并设置优化器的参数, 返回模型的优化器

            参数:

                weight_decay: 权重衰减(L2 正则化)的系数

                learning_rate: 学习率

                betas: Adam 优化器的 beta 参数, 通常是一个包含两个值的元组 (beta1, beta2)

                device_type: 设备类型, 通常是 'cpu' 或 'cuda'
            
            示例:

            optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=0.001, betas=(0.9, 0.999), device_type='cuda')

            这个函数需要单元测试下

        '''
        # start with all of the candidate parameters
        # 获取模型的所有参数, 并将其存储在一个字典中, 键是参数名称, 值是参数张量
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        # 过滤不需要梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # 将参数分为两组, 一个是需要权重衰减的参数(通常是权重矩阵), 一个是不需要权重衰减的参数(通常是偏置和层归一化参数)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 打印需要权重衰减和不需要权重衰减的参数数量, 以便用户了解模型的参数分布
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # 检查 torch.optim.AdamW 是否支持 fused 参数
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        '''
            功能: 函数用于估计模型的浮点运算利用率(Model FLOPs Utilization, MFU)
            MFU 是衡量模型在特定硬件上执行浮点运算效率的指标, 通常以硬件的理论峰值浮点运算性能(如 A100 GPU 的峰值 FLOPS)为基准

            参数:

                fwdbwd_per_iter: 每个迭代的前向和后向传播次数

                dt: 每个迭代的时间（以秒为单位）
            
            示例:

            optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=0.001, betas=(0.9, 0.999), device_type='cuda')

            这个函数需要单元测试下

        '''
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        # 6N means 2N for forward and 4N for backward
        flops_per_token = 6*N + 12*L*H*Q*T
        # 每个token的flops乘以序列长度
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        功能: 它接受一个输入序列 idx, 并根据模型的输出逐步生成新的 token, 直到达到指定的最大生成 token 数量

        参数
            idx: 输入序列, 形状为 (batch_size, sequence_length), 类型为 LongTensor

            max_new_tokens: 要生成的最大新 token 数量

            temperature: 温度参数, 用于调整生成概率分布的平滑度, 默认值为 1.0

            top_k: 可选参数, 用于限制生成时考虑的 token 数量, 默认值为 None
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # 如果输入序列的长度超过 block_size，则截取最后 block_size 长度的序列作为输入
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # logits 的形状为 (batch_size, sequence_length, vocab_size)
            # 语法介绍: self(idx_cond) 实际上是调用了 self.__call__(idx_cond)，而 __call__ 方法会自动调用 self.forward(idx_cond)
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            # 选择最后一个位置的输出，形状变为 (batch_size, vocab_size), 并根据 temperature 参数进行缩放
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            # 如果指定了 top_k，则只保留概率最高的 top_k 个 token，其他 token 的概率设为 -Inf
            if top_k is not None:
                # v shape is : (batch_size, top_k)
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # v[:, [-1]] 是 v 每行最小的值 (因为v中是降序的)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            # shape is : (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            # shape is : (batch_size, 1), 第二维度存储着[0, vocab_size]中的某个值，表示最后一个字的token
            idx_next = torch.multinomial(probs, num_samples=1) 
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
