import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    def __init__(self, num_items, embed_dim, img_feat_dim=768, dropout_rate=0.1):
        super(CrossModalFusion, self).__init__()
        self.id_embedding = nn.Embedding(num_items, embed_dim)
        self.img_projection = nn.Linear(img_feat_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, item_ids, img_features):
        id_embed = self.id_embedding(item_ids)
        img_proj = self.img_projection(img_features)
        
        # Element-wise addition of ID and Image features
        v_i = self.layer_norm(id_embed + img_proj)
        return self.dropout(v_i)