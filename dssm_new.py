import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import os
from datetime import datetime

# --- 全局常量 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #检查当前环境是否有可用的 NVIDIA GPU
print(f"Using device: {DEVICE}")
PAD_TOKEN = 0  # 用于序列填充的token #定义“填充符”的取值，后续把可变长序列（如用户点击序列）补齐到同一长度时会用这个数。
#能保证 RNN/Transformer/Embedding 在处理不同长度序列时对“补齐部分”不产生梯度贡献

# --- 1. 数据处理与工具函数 --- 同一张表中的不同列（特征），往往需要不同的数据处理方式

#让数值特征“标准化”→ 模型更稳定
#数据：用户/商品的数值属性（价格、年龄、时长等）等连续变量
def standardize(feature_values):
    """对特征值进行标准化处理"""
    mean_val = feature_values.mean()
    #std_val = feature_values.std() if feature_values.std() > 0 else 1
    std_val = feature_values.std()
    std_val = std_val if std_val > 0 else 1.0
    return (feature_values - mean_val) / std_val

#让序列特征“结构化”→ 能喂进 embedding 层
#数据：用户或物料的行为序列（ID序列）等离散变量（用户的历史点击 / 浏览 / 购买记录等）
def parse_item_seq(seq_str):
    """将 "[1, 2, 3]" 这样的字符串解析为整数列表"""
    if pd.isna(seq_str) or str(seq_str).strip() == "[]":
        return []
    try:
        return [int(x) for x in str(seq_str).strip('[] ').split(',')]
    except:
        return []

# --- 2. 自定义数据集 (AdDataset) ---
# 这个数据集能够处理用户行为序列 item_seq
class AdDataset(Dataset):
    def __init__(self, X, y, max_seq_len=20):
        # 用户特征
        self.user_cont_features = {
            "user_avg_ctr": torch.tensor(standardize(X["user_avg_ctr"].values), dtype=torch.float32),
            "user_total_interactions": torch.tensor(standardize(X["user_total_interactions"].values), dtype=torch.float32),
        }
        self.user_cat_features = {
            "age_price": torch.tensor(X["age_price"].values, dtype=torch.long),
            'gender_cate': torch.tensor(X["gender_cate"].values, dtype=torch.long),
            'cms_group_id': torch.tensor(X["cms_group_id"].values, dtype=torch.long),
        }

        # 广告特征
        self.ad_cont_features = {
            "ad_ctr": torch.tensor(standardize(X["ad_ctr"].values), dtype=torch.float32),
            "price": torch.tensor(standardize(X["price"].values), dtype=torch.float32),
            "brand_total_impressions": torch.tensor(standardize(X["brand_total_impressions"].values), dtype=torch.float32),
            "brand_ctr": torch.tensor(standardize(X["brand_ctr"].values), dtype=torch.float32),
            "ad_total_clicks": torch.tensor(standardize(X["ad_total_clicks"].values), dtype=torch.float32),
            "ad_total_impressions": torch.tensor(standardize(X["ad_total_impressions"].values), dtype=torch.float32),
            'cate_total_clicks': torch.tensor(standardize(X['cate_total_clicks'].values), dtype=torch.float32),
            'cate_ctr': torch.tensor(standardize(X['cate_ctr'].values), dtype=torch.float32),
            'cate_total_impressions': torch.tensor(standardize(X['cate_total_impressions'].values), dtype=torch.float32),
        }
        self.ad_cat_features = {
            "brand": torch.tensor(X["brand"].values, dtype=torch.long),
            "cate_id": torch.tensor(X["cate_id"].values, dtype=torch.long),
            "adgroup_id": torch.tensor(X["adgroup_id"].values, dtype=torch.long),
            "customer": torch.tensor(X["customer"].values, dtype=torch.long),
        }

        self.labels = torch.tensor(y.values, dtype=torch.float32)
        
        # --- 新增：处理用户行为序列 ---
        self.max_seq_len = max_seq_len
        self.item_seq = []
        for seq in X['item_seq']:
            if len(seq) > max_seq_len:
                seq = seq[-max_seq_len:]  # 取最近的 max_seq_len 个
            else:
                seq = [PAD_TOKEN] * (max_seq_len - len(seq)) + seq # 左侧填充PAD
            self.item_seq.append(torch.tensor(seq, dtype=torch.long))
        self.item_seq = torch.stack(self.item_seq)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item_seq = self.item_seq[idx]
        user_cont = [self.user_cont_features[col][idx] for col in self.user_cont_features]
        user_cat = [self.user_cat_features[col][idx] for col in self.user_cat_features]
        ad_cont = [self.ad_cont_features[col][idx] for col in self.ad_cont_features]
        ad_cat = [self.ad_cat_features[col][idx] for col in self.ad_cat_features]
        
        # 返回顺序：item_seq, user_cont..., user_cat..., ad_cont..., ad_cat..., label
        return (item_seq, *user_cont, *user_cat, *ad_cont, *ad_cat, self.labels[idx])


# --- 3. 模型定义 (DualTower with Transformer and GRU) ---

class InfoNCELoss(nn.Module):
    """InfoNCE损失函数"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, sim_matrix, targets):
        pos_scores = sim_matrix.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_sum_exp = torch.logsumexp(sim_matrix / self.temperature, dim=1)
        per_sample_loss = -(pos_scores / self.temperature - log_sum_exp)
        return per_sample_loss.mean()

# --- 新增：自定义带数值裁剪的TransformerEncoderLayer ---
# --- 新增：自定义带数值裁剪的TransformerEncoderLayer（修复后）---
# --- 重新定义CustomTransformerEncoderLayer（手动实现自注意力）---
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False, 
                 norm_first=False, bias=True, device=None, dtype=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, 
                         layer_norm_eps, batch_first, norm_first, bias, device, dtype)
        # 关键调整：更小的数值限制（进一步收紧）
        self.qk_scale = 0.05  # 从0.1→0.05，进一步降低QK^T数值
        self.attn_weight_clip = 3.0  # QK^T裁剪范围（保持3）
        self.ffn_output_clip = 2.0   # FFN输出裁剪范围（从3→2，更严格）
        self.qkv_clip = 1.0          # Q/K/V的数值裁剪范围（新增）

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        batch_size, seq_len, d_model = x.shape
        num_heads = self.self_attn.num_heads
        d_k = d_model // num_heads

        # 1. 分割Q、K、V权重并扩展批量维度
        q_weight = self.self_attn.in_proj_weight[:d_model, :].unsqueeze(0).expand(batch_size, -1, -1)
        k_weight = self.self_attn.in_proj_weight[d_model:2*d_model, :].unsqueeze(0).expand(batch_size, -1, -1)
        v_weight = self.self_attn.in_proj_weight[2*d_model:, :].unsqueeze(0).expand(batch_size, -1, -1)

        # 2. 计算Q、K、V（新增：计算后立即裁剪，避免数值膨胀）
        x_transposed = x.transpose(1, 2)
        q = torch.bmm(q_weight, x_transposed).transpose(1, 2)
        k = torch.bmm(k_weight, x_transposed).transpose(1, 2)
        v = torch.bmm(v_weight, x_transposed).transpose(1, 2)

        # 新增：裁剪Q、K、V的数值范围，防止后续QK^T溢出
        q = torch.clamp(q, min=-self.qkv_clip, max=self.qkv_clip)
        k = torch.clamp(k, min=-self.qkv_clip, max=self.qkv_clip)
        v = torch.clamp(v, min=-self.qkv_clip, max=self.qkv_clip)

        # 新增：检查Q、K、V是否有NaN/Inf（提前发现数值异常）
        if torch.isnan(q).any() or torch.isinf(q).any():
            raise ValueError(f"❌ Q存在NaN/Inf！max: {q.max()}, min: {q.min()}")
        if torch.isnan(k).any() or torch.isinf(k).any():
            raise ValueError(f"❌ K存在NaN/Inf！max: {k.max()}, min: {k.min()}")
        if torch.isnan(v).any() or torch.isinf(v).any():
            raise ValueError(f"❌ V存在NaN/Inf！max: {v.max()}, min: {v.min()}")

        # 3. 分头操作（不变）
        q = q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

        # 4. QK^T计算（双重缩放+裁剪，不变）
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (torch.sqrt(torch.tensor(d_k, dtype=torch.float32)) * self.qk_scale)
        attn_scores = torch.clamp(attn_scores, min=-self.attn_weight_clip, max=self.attn_weight_clip)

        # 5. 应用padding mask（不变）
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).bool()
            attn_scores = attn_scores.masked_fill(key_padding_mask, -1e9)

        # 6. Softmax + Dropout（不变）
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.self_attn.dropout, training=self.training)

        # 7. 注意力加权V + 输出投影（新增：投影后裁剪）
        x2 = torch.matmul(attn_weights, v)
        x2 = x2.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        x2 = self.self_attn.out_proj(x2)
        
        # 新增：裁剪输出投影后的数值
        x2 = torch.clamp(x2, min=-self.attn_weight_clip, max=self.attn_weight_clip)
        if torch.isnan(x2).any() or torch.isinf(x2).any():
            raise ValueError(f"❌ 自注意力输出存在NaN/Inf！max: {x2.max()}, min: {x2.min()}")

        # 8. 残差连接 + 层归一化（不变）
        x = x + self.dropout1(x2)
        if (x.abs() < 1e-10).all():  # 输入全接近零
            x += torch.randn_like(x) * 1e-10  # 微小噪声打破全零
        x = self.norm1(x)
        return x
    def _ff_block(self, x):
        # 第一步：linear1 + 裁剪 + ReLU
        x1 = self.linear1(x)
        # 新增：裁剪linear1输出，防止ReLU后数值放大
        x1 = torch.clamp(x1, min=-self.ffn_output_clip, max=self.ffn_output_clip)
        if torch.isnan(x1).any() or torch.isinf(x1).any():
            raise ValueError(f"❌ FFN linear1输出存在NaN/Inf！max: {x1.max()}, min: {x1.min()}")
        
        # ReLU激活（不变）
        x1 = self.activation(x1)
        
        # 第二步：Dropout + linear2 + 裁剪
        x2 = self.linear2(self.dropout(x1))
        # 新增：裁剪linear2输出
        x2 = torch.clamp(x2, min=-self.ffn_output_clip, max=self.ffn_output_clip)
        if torch.isnan(x2).any() or torch.isinf(x2).any():
            raise ValueError(f"❌ FFN linear2输出存在NaN/Inf！max: {x2.max()}, min: {x2.min()}")

        # 残差连接 + 层归一化（不变）
        x = x + self.dropout2(x2)
        if (x.abs() < 1e-10).all():
            x += torch.randn_like(x) * 1e-10
        x = self.norm2(x)
        return x


class DualTower(nn.Module):
    def __init__(
        self,
        # --- 新增：用户行为序列参数 ---
        num_item_ids,
        max_seq_len,
        user_seq_embedding_dim,
        nhead,
        num_layers,
        
        # --- 新增：物品塔门控网络参数 ---
        item_gru_hidden_dim,
        
        # 用户特征参数
        num_age_price,
        num_gender_cate,
        num_cms_group_id,
        
        # 广告特征参数
        num_brand,
        num_cate_id,
        num_adgroup_id,
        num_customer,
        
        # 超参数
        embedding_dim=64,
        hidden_dim=32,
        final_dim=16
    ):
        super().__init__()

        # --- 1. 用户塔 (User Tower) - 包含 Transformer ---
        self.item_seq_embedding = nn.Embedding(num_item_ids, user_seq_embedding_dim, padding_idx=PAD_TOKEN)

        # 关键修改：初始权重标准差从0.01→0.001，进一步缩小初始数值
        # nn.init.normal_(self.item_seq_embedding.weight, mean=0.0, std=0.001) 

        # 关键修改：替换为自定义的CustomTransformerEncoderLayer
        transformer_layer = CustomTransformerEncoderLayer(
            d_model=user_seq_embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,  # 保持32不变
            dropout=0.3,  # 正则化
            layer_norm_eps=1e-6,  # 提升数值稳定性
            batch_first=True,
            # 自定义层的裁剪参数（已在CustomTransformerEncoderLayer中定义，这里可不用重复传）
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)  # 层数1不变

        # --- 新增：用户塔所有层的稳定初始化（关键！）---
        # 1. 序列嵌入层：Xavier初始化（适合线性变换）
        # nn.init.xavier_uniform_(self.item_seq_embedding.weight)
        nn.init.normal_(self.item_seq_embedding.weight, mean=0.0, std=0.0001) 
        # 手动处理padding_idx的权重（避免padding位置的嵌入值异常）
        if self.item_seq_embedding.padding_idx is not None:
            self.item_seq_embedding.weight.data[self.item_seq_embedding.padding_idx] = 0.0

        # 2. Transformer层：遍历所有子层初始化
        for name, param in self.transformer_encoder.named_parameters():
            if 'weight' in name:
                if 'self_attn' in name or 'linear' in name or 'proj' in name:
                    # 关键：用更小的标准差初始化（0.0001替代Xavier）
                    nn.init.normal_(param, mean=0.0, std=0.0001)
                    # 自注意力和线性层用Xavier初始化
                    # nn.init.xavier_uniform_(param)
                elif 'layer_norm' in name:
                    # LayerNorm权重初始化为1（保持均值和方差稳定）
                    # nn.init.ones_(param)
                    # LayerNorm权重初始化为1（不变），但可轻微缩小避免放大
                    nn.init.normal_(param, mean=1.0, std=0.0001)
            elif 'bias' in name:
                # 所有偏置初始化为0（避免初始偏移过大）
                nn.init.zeros_(param)

        # transformer_layer = nn.TransformerEncoderLayer(
        #     d_model=user_seq_embedding_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=0.1, batch_first=True
        # )
        # self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        

        self.user_avg_ctr_emb = nn.Linear(1, embedding_dim)
        self.user_total_interactions_emb = nn.Linear(1, embedding_dim)
        # 线性层初始化
        for linear_layer in [self.user_avg_ctr_emb, self.user_total_interactions_emb]:
            # nn.init.xavier_uniform_(linear_layer.weight)
            # 替换Xavier为小标准差初始化
            nn.init.normal_(linear_layer.weight, mean=0.0, std=0.0001)
            nn.init.zeros_(linear_layer.bias)
        # 分类特征嵌入层初始化
        self.age_price_emb = nn.Embedding(num_age_price, embedding_dim)
        self.gender_cate_emb = nn.Embedding(num_gender_cate, embedding_dim)
        self.cms_group_id_emb = nn.Embedding(num_cms_group_id, embedding_dim)

        for emb_layer in [self.age_price_emb, self.gender_cate_emb, self.cms_group_id_emb]:
            nn.init.xavier_uniform_(emb_layer.weight)

        
        # Transformer输出维度 + 其他用户特征总维度
        user_mlp_input_dim = user_seq_embedding_dim + embedding_dim * 5
        self.user_mlp = nn.Sequential(
            nn.Linear(user_mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, final_dim)
        )

        # --- 2. 物品塔 (Item Tower) - 包含 GRU ---
        self.ad_ctr_emb = nn.Linear(1, embedding_dim)
        self.price_emb = nn.Linear(1, embedding_dim)
        self.brand_total_impressions_emb = nn.Linear(1, embedding_dim)
        self.brand_ctr_emb = nn.Linear(1, embedding_dim)
        self.ad_total_clicks_emb = nn.Linear(1, embedding_dim)
        self.ad_total_impressions_emb = nn.Linear(1, embedding_dim)
        self.cate_total_clicks_emb = nn.Linear(1, embedding_dim)
        self.cate_ctr_emb = nn.Linear(1, embedding_dim)
        self.cate_total_impressions_emb = nn.Linear(1, embedding_dim)
        self.adgroup_id_emb = nn.Embedding(num_adgroup_id, embedding_dim)
        self.cate_id_emb = nn.Embedding(num_cate_id, embedding_dim)
        self.brand_emb = nn.Embedding(num_brand, embedding_dim)
        self.customer_emb = nn.Embedding(num_customer, embedding_dim)
        
        # 物品塔线性层（同理修改）
        item_linear_layers = [
            self.ad_ctr_emb, self.price_emb, self.brand_total_impressions_emb,
            self.brand_ctr_emb, self.ad_total_clicks_emb, self.ad_total_impressions_emb,
            self.cate_total_clicks_emb, self.cate_ctr_emb, self.cate_total_impressions_emb
        ]
        for linear_layer in item_linear_layers:
            nn.init.normal_(linear_layer.weight, mean=0.0, std=0.0001)
            nn.init.zeros_(linear_layer.bias)
       

        # --- 然后继续定义物品塔的其他部分 ---
        self.adgroup_id_emb = nn.Embedding(num_adgroup_id, embedding_dim)
        self.cate_id_emb = nn.Embedding(num_cate_id, embedding_dim)
        self.brand_emb = nn.Embedding(num_brand, embedding_dim)
        self.customer_emb = nn.Embedding(num_customer, embedding_dim)

        # GRU的输入维度是所有物品特征拼接后的维度
        item_gru_input_dim = embedding_dim * 13

        self.item_gru = nn.GRU(
            input_size=item_gru_input_dim, hidden_size=item_gru_hidden_dim, num_layers=1, batch_first=True
        )
        
        # 物品塔 MLP (输入维度调整为GRU的输出维度)
        self.item_mlp = nn.Sequential(
            nn.Linear(item_gru_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, final_dim)
        )

    def forward_user(self, item_seq, user_avg_ctr, user_total_interactions, age_price, gender_cate, cms_group_id):
        """计算用户向量（带数值限制和逐步骤检查）"""
        # a. 物品序列嵌入（新增：裁剪嵌入值）
        seq_emb = self.item_seq_embedding(item_seq)
        seq_emb = seq_emb * 0.5  # 缩小初始嵌入值
        seq_emb = torch.clamp(seq_emb, min=-1.0, max=1.0)  # 从[-2,2]→[-1,1]，更严格
        if torch.isnan(seq_emb).any() or torch.isinf(seq_emb).any():
            raise ValueError(f"❌ seq_emb存在NaN/Inf！max: {seq_emb.max()}, min: {seq_emb.min()}")

        # b. Transformer编码（恢复全padding检查）
        src_key_padding_mask = (item_seq == PAD_TOKEN).bool()
        if src_key_padding_mask.all():
            raise ValueError(f"❌ 序列全是padding！item_seq: {item_seq}")
        seq_output = self.transformer_encoder(seq_emb, src_key_padding_mask=src_key_padding_mask)
        seq_output = torch.clamp(seq_output, min=-5.0, max=5.0)  # 保持不变
        if torch.isnan(seq_output).any() or torch.isinf(seq_output).any():
            raise ValueError(f"❌ Transformer输出存在NaN/Inf！max: {seq_output.max()}, min: {seq_output.min()}")

        # c. 平均池化（不变）
        non_pad_mask = (item_seq != PAD_TOKEN).unsqueeze(2).float()
        seq_rep = (seq_output * non_pad_mask).sum(dim=1) / (non_pad_mask.sum(dim=1) + 1e-8)
        seq_rep = torch.clamp(seq_rep, min=-5.0, max=5.0)
        if torch.isnan(seq_rep).any() or torch.isinf(seq_rep).any():
            raise ValueError(f"❌ seq_rep存在NaN/Inf！max: {seq_rep.max()}, min: {seq_rep.min()}")

        # d. 其他用户特征嵌入（新增：裁剪嵌入值）
        age_price_embbed = self.age_price_emb(age_price)
        gender_cate_embbed = self.gender_cate_emb(gender_cate)
        cms_group_id_embbed = self.cms_group_id_emb(cms_group_id)
        user_avg_ctr_embbed = self.user_avg_ctr_emb(user_avg_ctr.unsqueeze(1))
        user_total_interactions_embbed = self.user_total_interactions_emb(user_total_interactions.unsqueeze(1))
        
        # 新增：裁剪所有用户特征嵌入值
        all_embeddings = [age_price_embbed, gender_cate_embbed, cms_group_id_embbed, user_avg_ctr_embbed, user_total_interactions_embbed]
        for i, emb in enumerate(all_embeddings):
            emb = torch.clamp(emb, min=-2.0, max=2.0)
            all_embeddings[i] = emb
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                raise ValueError(f"❌ 用户特征嵌入{i}存在NaN/Inf！max: {emb.max()}, min: {emb.min()}")
        age_price_embbed, gender_cate_embbed, cms_group_id_embbed, user_avg_ctr_embbed, user_total_interactions_embbed = all_embeddings

        # e. 拼接特征并通过MLP（不变）
        user_features = torch.cat(
            [seq_rep, user_avg_ctr_embbed, user_total_interactions_embbed, age_price_embbed, gender_cate_embbed, cms_group_id_embbed],
            dim=1
        )
        user_vector = self.user_mlp(user_features)
        user_vector = torch.clamp(user_vector, min=-5.0, max=5.0)
        if torch.isnan(user_vector).any() or torch.isinf(user_vector).any():
            raise ValueError(f"❌ MLP输出存在NaN/Inf！max: {user_vector.max()}, min: {user_vector.min()}")

        # f. L2归一化（不变）
        return F.normalize(user_vector, p=2, dim=1)
    def forward_item(self, ad_ctr, price, brand_total_impressions, brand_ctr, ad_total_clicks, ad_total_impressions,
                     cate_total_clicks, cate_ctr, cate_total_impressions, brand, cate_id, adgroup_id, customer):
        """计算物品向量（使用门控网络融合特征）"""
        # a. 处理所有物品特征
        brand_embbed = self.brand_emb(brand)
        cate_id_embbed = self.cate_id_emb(cate_id)
        adgroup_id_embbed = self.adgroup_id_emb(adgroup_id)
        customer_embbed = self.customer_emb(customer)
        ad_ctr_embbed = self.ad_ctr_emb(ad_ctr.unsqueeze(1))
        price_embbed = self.price_emb(price.unsqueeze(1))
        brand_total_impressions_embbed = self.brand_total_impressions_emb(brand_total_impressions.unsqueeze(1))
        brand_ctr_embbed = self.brand_ctr_emb(brand_ctr.unsqueeze(1))
        ad_total_clicks_embbed = self.ad_total_clicks_emb(ad_total_clicks.unsqueeze(1))
        ad_total_impressions_embbed = self.ad_total_impressions_emb(ad_total_impressions.unsqueeze(1))
        cate_total_clicks_embbed = self.cate_total_clicks_emb(cate_total_clicks.unsqueeze(1))
        cate_ctr_embbed = self.cate_ctr_emb(cate_ctr.unsqueeze(1))
        cate_total_impressions_embbed = self.cate_total_impressions_emb(cate_total_impressions.unsqueeze(1))

        # b. 拼接所有物品特征向量
        item_features = torch.cat(
            [ad_ctr_embbed, price_embbed, brand_total_impressions_embbed, brand_ctr_embbed,
             ad_total_clicks_embbed, ad_total_impressions_embbed, cate_total_clicks_embbed,
             cate_ctr_embbed, cate_total_impressions_embbed, brand_embbed, cate_id_embbed,
             adgroup_id_embbed, customer_embbed], dim=1
        )
        
        # c. 通过门控网络 (GRU)
        item_features_gru = item_features.unsqueeze(1) # GRU需要一个时间步维度
        _, h_n = self.item_gru(item_features_gru)
        item_gru_output = h_n.squeeze(0)

        # d. 通过MLP生成最终物品向量
        return F.normalize(self.item_mlp(item_gru_output), p=2, dim=1)

    def forward(self, user_features_batch, item_features_batch):
        """计算相似度矩阵"""
        user_vectors = self.forward_user(*user_features_batch)
        item_vectors = self.forward_item(*item_features_batch)
        return torch.matmul(user_vectors, item_vectors.T)


# --- 4. 完整的训练与评估函数 ---
def train_model_infonce(model, train_loader, test_loader, test_df, K=100, epochs=10, lr=0.001, temperature=0.07):
    model.to(DEVICE)
    criterion = InfoNCELoss(temperature=temperature)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

    # 2. 设置梯度裁剪的范数阈值
    grad_clip_value = 0.05 

    test_user_ids = test_df['user'].values

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            # --- 修改：解包数据 ---
            item_seq = batch[0]
            user_cont_features = batch[1:3]
            user_cat_features = batch[3:6]
            ad_cont_features = batch[6:15]
            ad_cat_features = batch[15:-1]
            labels = batch[-1]

            user_features_batch = [item_seq.to(DEVICE), *[x.to(DEVICE) for x in user_cont_features], *[x.to(DEVICE) for x in user_cat_features]]
            item_features_batch = [*[x.to(DEVICE) for x in ad_cont_features], *[x.to(DEVICE) for x in ad_cat_features]]
            
            sim_matrix = model(user_features_batch, item_features_batch)
            batch_size = sim_matrix.size(0)
            targets = torch.arange(batch_size, device=DEVICE)
            loss = criterion(sim_matrix, targets)
            
            # --- 检查损失是否为 NaN ---
            if torch.isnan(loss):
                print(f"!!! Loss is NaN at Epoch {epoch+1}, Batch {batch_idx+1} !!!")
                # 在这里可以设置断点进行调试

            optimizer.zero_grad()
            loss.backward()

            # --- 新增：检查梯度是否为 NaN ---
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"!!! Gradient of parameter '{name}' is NaN at Epoch {epoch+1}, Batch {batch_idx+1} !!!")
                    # 在这里设置断点进行调试

            # --- 3. 添加梯度裁剪 ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

            optimizer.step()
            # --- 新增：检查参数是否为 NaN ---
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"!!! Parameter '{name}' is NaN at Epoch {epoch+1}, Batch {batch_idx+1} !!!")
                    # 在这里设置断点进行调试
                    # 一旦参数变为NaN，训练就无法继续了

            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")

        # --- 评估阶段 ---
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]")):
            # for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]"):
                # --- 修改：解包数据 ---
                item_seq = batch[0]
                user_cont_features = batch[1:3]
                user_cat_features = batch[3:6]
                ad_cont_features = batch[6:15]
                ad_cat_features = batch[15:-1]
                labels = batch[-1]

                user_features_batch = [item_seq.to(DEVICE), *[x.to(DEVICE) for x in user_cont_features], *[x.to(DEVICE) for x in user_cat_features]]
                item_features_batch = [*[x.to(DEVICE) for x in ad_cont_features], *[x.to(DEVICE) for x in ad_cat_features]]
                
                # sim_matrix = model(user_features_batch, item_features_batch)

                # --- 详细的NaN检查 ---
                try:
                    # 1. 计算用户向量并检查
                    user_vectors = model.forward_user(*user_features_batch)
                    if torch.isnan(user_vectors).any():
                        print(f"!!! NaN detected in user_vectors at Test Epoch {epoch+1}, Batch {batch_idx+1} !!!")
                        # 在这里设置断点调试

                    # 2. 计算物品向量并检查
                    item_vectors = model.forward_item(*item_features_batch)
                    if torch.isnan(item_vectors).any():
                        print(f"!!! NaN detected in item_vectors at Test Epoch {epoch+1}, Batch {batch_idx+1} !!!")
                        # 在这里设置断点调试

                    # 3. 计算相似度矩阵并检查
                    sim_matrix = torch.matmul(user_vectors, item_vectors.T)
                    if torch.isnan(sim_matrix).any():
                        print(f"!!! NaN detected in sim_matrix at Test Epoch {epoch+1}, Batch {batch_idx+1} !!!")
                        # 在这里设置断点调试

                except Exception as e:
                    print(f"!!! Exception occurred during forward pass at Test Epoch {epoch+1}, Batch {batch_idx+1}: {e} !!!")
                    raise # 重新抛出异常以中断程序  


                batch_size = sim_matrix.size(0)
                targets = torch.arange(batch_size, device=DEVICE)
                loss = criterion(sim_matrix, targets)
                test_loss += loss.item()

                predictions = torch.diag(sim_matrix).cpu().numpy()
                all_preds.extend(predictions)
                all_labels.extend(labels.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)

        if np.isnan(all_preds_np).any():
            print("!!! NaN detected in final predictions array before AUC calculation !!!")
            # 可以打印出含有NaN的索引和值进行分析
            nan_indices = np.where(np.isnan(all_preds_np))
            print(f"NaN indices: {nan_indices}")
            print(f"NaN values count: {len(nan_indices[0])}")
            # 为了不让程序中断，可以将NaN替换为0或其他值，但这只是为了调试
            all_preds_np = np.nan_to_num(all_preds_np, nan=0.0)

        test_auc = roc_auc_score(all_labels_np, all_preds_np)

        eval_df = pd.DataFrame({'user': test_user_ids, 'pred_score': all_preds_np, 'label': all_labels_np})
        user_metrics = []
        for user_id, group_df in eval_df.groupby('user'):
            if group_df['label'].sum() == 0 or len(group_df) <= 1:
                continue
            group_df_sorted = group_df.sort_values('pred_score', ascending=False)
            top_k_predictions = group_df_sorted.head(K)
            num_relevant = group_df['label'].sum()
            num_correct_in_top_k = top_k_predictions['label'].sum()
            recall_at_k = num_correct_in_top_k / num_relevant if num_relevant > 0 else 0.0
            precision_at_k = num_correct_in_top_k / K if K > 0 else 0.0
            user_metrics.append({'recall@k': recall_at_k, 'precision@k': precision_at_k})
        
        avg_recall_at_k = pd.DataFrame(user_metrics)['recall@k'].mean() if user_metrics else 0.0
        avg_precision_at_k = pd.DataFrame(user_metrics)['precision@k'].mean() if user_metrics else 0.0

        print(f"Test Loss: {avg_test_loss:.4f} | Test AUC: {test_auc:.4f}")
        print(f"Test Recall@{K}: {avg_recall_at_k:.4f} | Test Precision@{K}: {avg_precision_at_k:.4f}")
        print("-" * 80)
        scheduler.step(test_auc)

    return model


# --- 5. 主程序：数据加载、模型初始化与训练 ---
# --- 5. 主程序：数据加载、模型初始化与训练 ---
if __name__ == "__main__":
    # --- 配置 ---
    DATA_PATH_PART1 = "/workspace/data1/program/train_csv.csv"
    DATA_PATH_MAY13 = "/workspace/data1/program/may13_data.csv"
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 5e-7 # 保持较小的学习率

    # --- Transformer 和 GRU 超参数 ---
    MAX_SEQ_LEN = 20
    USER_SEQ_EMBEDDING_DIM = 32
    NHEAD = 2
    NUM_LAYERS = 2
    ITEM_GRU_HIDDEN_DIM = 64

    # --- 加载和预处理数据 ---
    print("Loading and preprocessing data...")
    # 读取数据，可以考虑限制行数以加快调试
    data_part1 = pd.read_csv(DATA_PATH_PART1)
    data_part1['time_stamp'] = pd.to_datetime(data_part1['time_stamp'])
    data_part1 = data_part1[data_part1['time_stamp'].dt.date != pd.to_datetime('2017-05-13').date()]

    data_may13 = pd.read_csv(DATA_PATH_MAY13)
    
    con = pd.concat([data_part1, data_may13], ignore_index=True).copy()
    del data_part1, data_may13

    # 确保所有需要的列都存在
    required_cols = ["user", "item_seq", "brand", "ad_ctr", "price", "brand_total_impressions",
                     "user_avg_ctr", "brand_ctr", "ad_total_clicks", "cate_id", "adgroup_id",
                     "customer", "age_price", "ad_total_impressions", 'gender_cate',
                     'user_total_interactions', 'cate_total_clicks', 'cms_group_id',
                     'cate_ctr', 'cate_total_impressions', "clk"]
    # 过滤出存在的列
    existing_cols = [col for col in required_cols if col in con.columns]
    con = con[existing_cols]

    # --- 核心修改：处理 item_seq 字段 ---
    con['item_seq'] = con['item_seq'].apply(parse_item_seq)

    # --- 在此处插入您提供的代码片段 ---
    # 核心修改：处理item_seq并过滤有效token<2的样本
    initial_count = len(con)
    # 保留至少2个有效token的样本（避免注意力计算极端化）
    con = con[con['item_seq'].apply(lambda seq: len(seq) >= 2)]
    filtered_count = initial_count - len(con)
    print(f"Filtered {filtered_count} samples (empty or <2 valid tokens), remaining: {len(con)}")

    # 若剩余样本过少，终止程序（避免无意义训练）
    if len(con) < 2000:
        raise ValueError(f"Remaining samples ({len(con)}) too few! Check raw data quality.")
    # --- 代码片段结束 ---

    # 为行为序列中的物品ID创建映射
    all_item_ids = set()
    for seq in con['item_seq']:
        all_item_ids.update(seq)
    if 'adgroup_id' in con.columns:
        all_item_ids.update(con['adgroup_id'].unique())
    item_id_mapping = {id: idx+1 for idx, id in enumerate(all_item_ids)} # PAD=0
    NUM_ITEM_IDS = len(item_id_mapping) + 1
    con['item_seq'] = con['item_seq'].apply(lambda seq: [item_id_mapping[id] for id in seq if id in item_id_mapping])

    # 映射其他离散变量
    columns_to_map = ['brand', 'cate_id', 'adgroup_id', "customer", "age_price", 'gender_cate', 'cms_group_id']
    columns_to_map = [col for col in columns_to_map if col in con.columns]
    for col in columns_to_map:
        con[col] = con[col].astype('category').cat.codes
        con[col] = con[col] + 1 # 确保没有-1

    # 类型转换
    for col in ['brand', 'clk']:
        if col in con.columns:
            con[col] = con[col].astype("int")

    # 划分数据集
    # 这里划分具体按照自己的数据集进行划分 ，而不是50000
    train_df = con.iloc[:50000].sample(n=5000, random_state=42)
    test_df = con.iloc[50000:].sample(n=5000, random_state=42)
    
    X_train = train_df.drop(["clk"], axis=1, errors='ignore')
    y_train = train_df["clk"]
    X_test = test_df.drop(["clk"], axis=1, errors='ignore')
    y_test = test_df["clk"]

    # 创建数据加载器
    train_dataset = AdDataset(X_train, y_train, max_seq_len=MAX_SEQ_LEN)
    test_dataset = AdDataset(X_test, y_test, max_seq_len=MAX_SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 准备特征计数
    feature_counts = {}
    for col in ['age_price', 'gender_cate', 'cms_group_id', 'brand', 'cate_id', 'adgroup_id', 'customer']:
        if col in con.columns:
            feature_counts[f"num_{col}"] = con[col].nunique() + 1 # +1 for 0-based indexing

    # --- 初始化和训练模型 ---
    print("Initializing model...")
    model = DualTower(
        # --- 新增参数 ---
        num_item_ids=NUM_ITEM_IDS,
        max_seq_len=MAX_SEQ_LEN,
        user_seq_embedding_dim=USER_SEQ_EMBEDDING_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        item_gru_hidden_dim=ITEM_GRU_HIDDEN_DIM,
        
        # --- 原有参数 ---
        num_age_price=feature_counts.get("num_age_price", 1),
        num_gender_cate=feature_counts.get("num_gender_cate", 1),
        num_cms_group_id=feature_counts.get("num_cms_group_id", 1),
        num_brand=feature_counts.get("num_brand", 1),
        num_cate_id=feature_counts.get("num_cate_id", 1),
        num_adgroup_id=feature_counts.get("num_adgroup_id", 1),
        num_customer=feature_counts.get("num_customer", 1),
        embedding_dim=64,
        hidden_dim=32,
        final_dim=16,
    )

    print("Starting training...")
    trained_model = train_model_infonce(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        test_df=test_df,
        K=100,
        epochs=EPOCHS,
        lr=LR,
        temperature=0.07,
    )
    
    # 可选：保存模型
    # torch.save(trained_model.state_dict(), f'dual_tower_transformer_gru_epochs_{EPOCHS}.pth')
    print("Training finished.")