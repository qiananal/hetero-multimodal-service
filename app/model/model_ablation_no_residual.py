"""

基于 model_geo_fusion_single_reg_fast.py 修改:
- 移除: EdgeConv2-4 的残差连接
- 移除: res_adapt_3, res_adapt_4 适配层
- 保留: 其他所有组件

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class FastGeometryExtractor(nn.Module):
    """快速批量几何特征提取器"""
    def __init__(self):
        super().__init__()
        self.geo_dim = 6
    
    def forward(self, x):
        points = x.transpose(1, 2)
        min_coords = points.min(dim=1)[0]
        max_coords = points.max(dim=1)[0]
        bbox_size = max_coords - min_coords
        
        bbox_volume = bbox_size[:, 0] * bbox_size[:, 1] * bbox_size[:, 2]
        bbox_surface = 2 * (bbox_size[:, 0] * bbox_size[:, 1] + 
                           bbox_size[:, 1] * bbox_size[:, 2] + 
                           bbox_size[:, 0] * bbox_size[:, 2])
        
        centroid = points.mean(dim=1, keepdim=True)
        pts_centered = points - centroid
        std_xyz = pts_centered.std(dim=1)
        ellipsoid_volume = (4/3) * 3.14159 * std_xyz[:, 0] * std_xyz[:, 1] * std_xyz[:, 2] * 8
        
        sphericity = torch.clamp(
            (3.14159 ** (1/3) * (6 * ellipsoid_volume.abs()) ** (2/3)) / (bbox_surface + 1e-6), 0, 1)
        
        sorted_size = torch.sort(bbox_size, dim=1, descending=True)[0]
        aspect_ratio = torch.clamp(sorted_size[:, 0] / (sorted_size[:, 2] + 1e-6), 1, 10)
        flatness = sorted_size[:, 1] / (sorted_size[:, 0] + 1e-6)
        
        return torch.stack([bbox_volume/1000, ellipsoid_volume/1000, bbox_surface/100,
                           sphericity, aspect_ratio/5, flatness], dim=1)


class GeoAttentionFusion(nn.Module):
    """几何特征注意力融合"""
    def __init__(self, dgcnn_dim, geo_dim, hidden_dim=64):
        super().__init__()
        self.dgcnn_proj = nn.Sequential(nn.Linear(dgcnn_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.geo_proj = nn.Sequential(nn.Linear(geo_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2), nn.Softmax(dim=1))
        self.output_dim = hidden_dim
    
    def forward(self, dgcnn_feat, geo_feat):
        dgcnn_proj = self.dgcnn_proj(dgcnn_feat)
        geo_proj = self.geo_proj(geo_feat)
        combined = torch.cat([dgcnn_proj, geo_proj], dim=1)
        attention_weights = self.attention(combined)
        fused_feat = attention_weights[:, 0:1] * dgcnn_proj + attention_weights[:, 1:2] * geo_proj
        return fused_feat, attention_weights


class DGCNN_Ablation_NoResidual(nn.Module):
 
    def __init__(self, k=20, emb_dims=256, num_classes=2, dropout=0.2):
        super().__init__()
        self.k = k
        
        # ========== DGCNN 编码器 (无残差连接) ==========
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, 1, bias=False), nn.BatchNorm1d(emb_dims), nn.LeakyReLU(0.2))
        
        # 注意: 移除了 res_adapt_3 和 res_adapt_4
        
        dgcnn_dim = emb_dims * 2  # 512
        
        # ========== 几何特征分支 ==========
        self.geo_extractor = FastGeometryExtractor()
        self.geo_encoder = nn.Sequential(
            nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU()
        )
        geo_encoded_dim = 64
        
        # ========== 注意力融合 ==========
        self.attention_fusion = GeoAttentionFusion(dgcnn_dim, geo_encoded_dim, hidden_dim=128)
        fusion_dim = 128
        reg_input_dim = fusion_dim + dgcnn_dim + geo_encoded_dim  # 704
        
        # ========== 回归头 ==========
        self.reg_head = nn.Sequential(
            nn.Linear(reg_input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # ========== 分类头 ==========
        self.cls_head = nn.Sequential(
            nn.Linear(dgcnn_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # ========== 几何直接回归 ==========
        self.geo_reg_head = nn.Sequential(nn.Linear(geo_encoded_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        
        self._print_params_info()
    
    def _print_params_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Ablation-NoResidual] 总参数量: {total_params:,}")
        print(f"[Ablation-NoResidual] 移除残差连接")
    
    def forward(self, x):
        if x.dim() == 3 and x.size(-1) == 3:
            x = x.transpose(1, 2).contiguous()
        
        batch_size = x.size(0)
        
        # ========== DGCNN编码 (无残差连接) ==========
        x1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1).max(dim=-1)[0]
        
        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2).max(dim=-1)[0]
        # 移除: x2 = x2 + x1
        
        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3).max(dim=-1)[0]
        # 移除: x3 = x3 + self.res_adapt_3(x2)
        
        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4).max(dim=-1)[0]
        # 移除: x4 = x4 + self.res_adapt_4(x3)
        
        dgcnn_feat = self.conv5(torch.cat((x1, x2, x3, x4), dim=1))
        dgcnn_max = F.adaptive_max_pool1d(dgcnn_feat, 1).view(batch_size, -1)
        dgcnn_avg = F.adaptive_avg_pool1d(dgcnn_feat, 1).view(batch_size, -1)
        dgcnn_global = torch.cat((dgcnn_max, dgcnn_avg), dim=1)
        
        # ========== 几何特征 ==========
        geo_feat = self.geo_extractor(x)
        geo_encoded = self.geo_encoder(geo_feat)
        
        # ========== 注意力融合 ==========
        fused_feat, attention_weights = self.attention_fusion(dgcnn_global, geo_encoded)
        
        # ========== 回归 ==========
        reg_input = torch.cat([fused_feat, dgcnn_global, geo_encoded], dim=1)
        weight_pred = self.reg_head(reg_input).squeeze(-1)
        
        # ========== 分类 ==========
        cls_logits = self.cls_head(dgcnn_global)
        
        # ========== 几何直接回归 ==========
        geo_weight_pred = self.geo_reg_head(geo_encoded).squeeze(-1)
        
        return cls_logits, weight_pred, geo_weight_pred, attention_weights


if __name__ == '__main__':
    model = DGCNN_Ablation_NoResidual(k=32, dropout=0.2)
    x = torch.randn(4, 1024, 3)
    out = model(x)
    print(f"cls={out[0].shape}, pred={out[1].shape}, geo_pred={out[2].shape}")
