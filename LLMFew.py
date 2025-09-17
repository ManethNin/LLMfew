import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from CasualCNN import CausalCNNEncoder
from Embed import PatchEmbedding
from LLMprepare import LLMprepare


class LLMFew(nn.Module):
    def __init__(self, configs):
        super(LLMFew, self).__init__()
        self.configs = configs
        self.num_class = configs.num_class
        self.length = configs.length
        self.num_class = configs.num_class
        self.dimensions = configs.dimensions
        self.llm_type = configs.llm_type
        self.lora = configs.lora
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.llm_model, self.d_model = LLMprepare(configs)
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, configs.dropout)  # d_model, patch_len, stride, dropout
        self.patch_nums = int((self.length - self.patch_len) / self.stride + 2)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.encoder = CausalCNNEncoder(in_channels=self.dimensions, channels=configs.channels, depth=configs.depth,
                                        reduced_size=configs.reduced_size, out_channels=self.d_model,
                                        kernel_size=configs.kernel_size)
        self.dropout = nn.Dropout(configs.dropout)
        # classification head
        self.relu = nn.LeakyReLU()
        self.act = F.relu
        self.ln_proj = nn.LayerNorm(self.d_model * self.patch_nums)
        self.mapping = nn.Sequential(nn.Linear(self.d_model * self.patch_nums, self.num_class),
                                     nn.Dropout(configs.dropout))
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        # x: batch, dim(dimensions), seq(length)
        x = x.contiguous()
        B, L, M = x.shape
        input_x = self.padding_patch_layer(x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        input_x = rearrange(input_x, 'b m n p-> (b n) m p')
        output_x = self.encoder(input_x)
        output_x = self.dropout(output_x)
        output_x = rearrange(output_x, '(b n) o-> b n o', b=B)
        llm_out = self.llm_model(inputs_embeds=output_x.contiguous()).last_hidden_state
        outputs = self.relu(llm_out + output_x).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.mapping(outputs)
        outputs = self.activation(outputs)
        return outputs
