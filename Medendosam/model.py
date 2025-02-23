import torch 
import torch.nn as nn 
from einops import rearrange
from PCSA import PartialConv3WithSpatialAttention
from WMS import WeightedMultiScaleMLA


class Prompt_enhance_adapter(nn.Module):
    def __init__(self, dim, num_classes=7):
        super(Prompt_enhance_adapter, self).__init__()
        self.num_classes = num_classes
        self.WMS = WeightedMultiScaleMLA(in_channels=dim, out_channels=dim).cuda()
        self.PCSA = PartialConv3WithSpatialAttention(dim, 2, 'split_cat').cuda()
        self.channel_projection = nn.Conv2d(dim, num_classes, kernel_size=1)

    def forward(self, feat, cls_prompts):
        cls_prompts_expanded = cls_prompts.expand(-1, -1, -1, feat.size(2)).transpose(2, 3)
        combined = torch.cat([feat, cls_prompts_expanded], dim=1)
        fused_1 = self.WMS(combined)
        fused_2 = self.PCSA(combined)
        fused_3 = fused_1 + fused_2
        fused = self.channel_projection(fused_3)
        return fused


class Prototype_Prompt_Encoder(nn.Module):
    def __init__(self, feat_dim=256, 
                        hidden_dim_dense=128, 
                        hidden_dim_sparse=128, 
                        size=64, 
                        num_tokens=8):
                
        super(Prototype_Prompt_Encoder, self).__init__()
        self.dense_fc_1 = nn.Conv2d(feat_dim, hidden_dim_dense, 1)
        self.dense_fc_2 = nn.Conv2d(hidden_dim_dense, feat_dim, 1)
        self.adapter = Prompt_enhance_adapter(14)
        
        self.relu = nn.ReLU()

        self.sparse_fc_1 = nn.Conv1d(size*size, hidden_dim_sparse, 1)
        self.sparse_fc_2 = nn.Conv1d(hidden_dim_sparse, num_tokens, 1)
        
        
        pn_cls_embeddings = [nn.Embedding(num_tokens, feat_dim) for _ in range(2)] # one for positive and one for negative 

            
        self.pn_cls_embeddings = nn.ModuleList(pn_cls_embeddings)
                
    def forward(self, feat, prototypes, cls_ids):
  
        cls_prompts = prototypes.unsqueeze(-1)
        cls_prompts = torch.stack([cls_prompts for _ in range(feat.size(0))], dim=0)

        
        feat = torch.stack([feat for _ in range(cls_prompts.size(1))], dim=1)

        feat = self.adapter(feat,cls_prompts)
        #Use CAPE to compute dense and sparse embeddings
        feat_sparse = feat.clone()
        # compute dense embeddings
        one_hot = torch.nn.functional.one_hot(cls_ids-1,7) 
        feat = feat[one_hot ==1]
        feat = rearrange(feat,'b (h w) c -> b c h w', h=64, w=64)
        dense_embeddings = self.dense_fc_2(self.relu(self.dense_fc_1(feat)))
        
        # compute sparse embeddings
        feat_sparse = rearrange(feat_sparse,'b num_cls hw c -> (b num_cls) hw c')
        sparse_embeddings = self.sparse_fc_2(self.relu(self.sparse_fc_1(feat_sparse)))
        sparse_embeddings = rearrange(sparse_embeddings,'(b num_cls) n c -> b num_cls n c', num_cls=7)
        pos_embed = self.pn_cls_embeddings[1].weight.unsqueeze(0).unsqueeze(0) * one_hot.unsqueeze(-1).unsqueeze(-1)
        neg_embed = self.pn_cls_embeddings[0].weight.unsqueeze(0).unsqueeze(0) * (1-one_hot).unsqueeze(-1).unsqueeze(-1)
        
        sparse_embeddings = sparse_embeddings + pos_embed.detach() + neg_embed.detach()
            
        sparse_embeddings = rearrange(sparse_embeddings,'b num_cls n c -> b (num_cls n) c')
        
        return dense_embeddings, sparse_embeddings


class Learnable_Prototypes(nn.Module):
    def __init__(self, num_classes=7 , feat_dim=256):
        super(Learnable_Prototypes, self).__init__()
        self.class_embeddings = nn.Embedding(num_classes, feat_dim)
        
    def forward(self):
        return self.class_embeddings.weight