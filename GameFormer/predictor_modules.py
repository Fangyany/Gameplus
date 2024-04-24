import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe
        
        return self.dropout(x)
    

class AgentEncoder(nn.Module):
    def __init__(self, agent_dim):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(agent_dim, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs)
        output = traj[:, -1]

        return output
    



class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet, self).__init__()
        input_size = 7
        output_size = 256
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)  
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)  
        self.fc3 = nn.Linear(64, output_size)
        self._lane_feature = 7

    def forward(self, x):
        x = x.reshape(x.shape[0], -1, self._lane_feature)
        batch_size, seq_len, _ = x.size()
        
        # pooling
        x = x.transpose(1, 2) 
        x = F.avg_pool1d(x, kernel_size=2, stride=2)  
        x = x.transpose(1, 2)  
        
        # 更新维度信息
        _, seq_len, _ = x.size()  
        x = x.view(-1, self._lane_feature).float()
        
        x = self.fc1(x)
        x = self.bn1(x) 
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x) 
        x = torch.relu(x)
        x = self.fc3(x)
        
        # 将x的形状重塑回 [32, 1000, 256]
        x = x.view(batch_size, seq_len, -1)
        mask = torch.eq(x[:, :, ::].sum(-1), 0)
        return x, mask

class RouteNet(nn.Module):
    def __init__(self):
        super(RouteNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64) 
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear = nn.Linear(128, 256)
        self._route_len = 50
        self._route_feature = 3

    def forward(self, x):
        x = x.reshape(x.shape[0], -1, self._route_feature)
        # 输入形状: [32, 500, 3]
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) 
        x = F.relu(x)  
        batch_size, channels, seq_len = x.shape
        x = x.permute(0, 2, 1) 
        x = x.contiguous().view(-1, channels)
        
        x = self.linear(x)
        x = x.view(batch_size, seq_len, -1)
        x = F.max_pool1d(x.permute(0, 2, 1), kernel_size=500, stride=500).permute(0, 2, 1)
        return x
 
class M2M(nn.Module):
    def __init__(self):
        super(M2M, self).__init__()
        self.linear1 = nn.Linear(512, 1024)
        self.bn1 = nn.BatchNorm1d(1024)  
        self.linear2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)  
        self.linear3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256) 

        # 初始化模型参数
        for layer in [self.linear1, self.linear2, self.linear3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, lanes, route):
        batch_size, num_lanes, len_feat = lanes.size()
        
        route_features = []
        for i in range(batch_size):
            route_feature = route[i].repeat(num_lanes, 1)
            route_features.append(route_feature)

        route_features = torch.cat(route_features, dim=0)
        # print(route_features.shape, lanes.reshape(-1, len_feat).shape)
        lanes_concat = torch.cat((lanes.reshape(-1, len_feat), route_features), dim=1)
        
        x = F.relu(self.bn1(self.linear1(lanes_concat)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = F.relu(self.bn3(self.linear3(x)))
        x = x.reshape(batch_size, num_lanes, len_feat)
        return x
    

class VectorMapEncoder(nn.Module):
    def __init__(self, map_dim, map_len):
        super(VectorMapEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(map_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
        self.position_encode = PositionalEncoding(max_len=map_len)

    def segment_map(self, map, map_encoding):
        B, N_e, N_p, D = map_encoding.shape 
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, 10))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)

        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, N_e, N_p//10, N_p//(N_p//10))
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

        return map_encoding, map_mask

    def forward(self, input):
        output = self.position_encode(self.point_net(input))
        encoding, mask = self.segment_map(input, output)

        return encoding, mask
    

class FutureEncoder(nn.Module):
    def __init__(self):
        super(FutureEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 256))

    def state_process(self, trajs, current_states):
        M = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, M, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-6)).unsqueeze(-1)
        trajs = torch.cat([trajs, theta, v], dim=-1) # (x, y, heading, vx, vy)

        return trajs

    def forward(self, trajs, current_states):
        trajs = self.state_process(trajs, current_states)
        trajs = self.mlp(trajs.detach())
        output = torch.max(trajs, dim=-2).values

        return output


class GMMPredictor(nn.Module):
    def __init__(self, modalities=6):
        super(GMMPredictor, self).__init__()
        self.modalities = modalities
        self._future_len = 80
        self.gaussian = nn.Sequential(nn.Linear(256, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, self._future_len*4))
        self.score = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, 1))
    
    def forward(self, input):
        B, N, M, _ = input.shape
        traj = self.gaussian(input).view(B, N, M, self._future_len, 4) # mu_x, mu_y, log_sig_x, log_sig_y
        score = self.score(input).squeeze(-1)

        return traj, score


class SelfTransformer(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(SelfTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, inputs, mask=None):
        attention_output, _ = self.self_attention(inputs, inputs, inputs, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class CrossTransformer(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        attention_output = query + attention_output   # 原文没有
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class InitialPredictionDecoder(nn.Module):
    def __init__(self, modalities, neighbors, dim=256):
        super(InitialPredictionDecoder, self).__init__()
        self._modalities = modalities
        self._agents = neighbors + 1
        self.multi_modal_query_embedding = nn.Embedding(modalities, dim)
        self.agent_query_embedding = nn.Embedding(self._agents, dim)
        self.query_encoder = CrossTransformer()
        self.predictor = GMMPredictor()
        self.register_buffer('modal', torch.arange(modalities).long())
        self.register_buffer('agent', torch.arange(self._agents).long())

    def forward(self, current_states, encoding, mask):
        N = self._agents
        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        agent_query = self.agent_query_embedding(self.agent)
        query = encoding[:, :N, None] + multi_modal_query[None, :, :] + agent_query[:, None, :]
        query_content = torch.stack([self.query_encoder(query[:, i], encoding, encoding, mask) for i in range(N)], dim=1)
        predictions, scores = self.predictor(query_content)
        predictions[..., :2] += current_states[:, :N, None, None, :2]

        return query_content, predictions, scores


class InteractionDecoder(nn.Module):
    def __init__(self, modalities, future_encoder):
        super(InteractionDecoder, self).__init__()
        self.modalities = modalities
        self.interaction_encoder = SelfTransformer()
        self.query_encoder = CrossTransformer()
        self.future_encoder = future_encoder
        self.decoder = GMMPredictor()

    def forward(self, current_states, actors, scores, last_content, encoding, mask):
        N = actors.shape[1]
        
        # using future encoder to encode the future trajectories
        multi_futures = self.future_encoder(actors[..., :2], current_states[:, :N])
        
        # using scores to weight the encoded futures
        futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2)    
        
        # using self-attention to encode the interaction
        interaction = self.interaction_encoder(futures, mask[:, :N])
        
        # append the interaction encoding to the common content
        encoding = torch.cat([interaction, encoding], dim=1)

        # mask out the corresponding agents
        mask = torch.cat([mask[:, :N], mask], dim=1)
        mask = mask.unsqueeze(1).expand(-1, N, -1).clone()
        for i in range(N):
            mask[:, i, i] = 1

        # using cross-attention to decode the future trajectories
        query = last_content + multi_futures
        query_content = torch.stack([self.query_encoder(query[:, i], encoding, encoding, mask[:, i]) for i in range(N)], dim=1)
        trajectories, scores = self.decoder(query_content)
        
        # add the current states to the trajectories
        trajectories[..., :2] += current_states[:, :N, None, None, :2]

        return query_content, trajectories, scores
    
    
class ContentFeatureExtractor(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(ContentFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x

class FeatureFusionAttention(nn.Module):
    def __init__(self, feature_size, num_features):
        super(FeatureFusionAttention, self).__init__()
        self.query = nn.Linear(feature_size, feature_size)
        self.key = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)
        self.num_features = num_features

    def forward(self, x):
        # print(x.shape)
        # x 的形状是 [batch_size, num_tensors, seq_len, num_features, feature_size]
        batch_size, num_tensors, seq_len, num_features, feature_size = x.shape
        
        # 将x变形为 [batch_size * seq_len * num_features, num_tensors, feature_size]
        x_reshaped = x.view(-1, num_tensors, feature_size)
        
        # 计算查询（Q）、键（K）和值（V）
        Q = self.query(x_reshaped)
        K = self.key(x_reshaped)
        V = self.value(x_reshaped)

        # 计算注意力分数
        attention_scores = Q @ K.transpose(-2, -1) / (feature_size ** 0.5)
        
        # 应用softmax函数获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended = attention_weights @ V
        # 将结果变形回 [batch_size, seq_len, num_features, num_tensors, feature_size]
        attended = attended.view(batch_size, seq_len, num_features, num_tensors, feature_size)
        
        # 在num_tensors维度上求和或者平均
        # fused_features = attended.mean(dim=3)  # 使用 mean(dim=3) 来取平均
        fused_features = attended.sum(dim=3)  # 使用 sum(dim=3) 来取和
        
        return fused_features
    
    
class FeatureFusionMultiHeadAttention(nn.Module):
    def __init__(self, feature_size, num_heads):
        super(FeatureFusionMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_size = feature_size
        self.depth = feature_size // num_heads
        
        self.query = nn.Linear(feature_size, feature_size)
        self.key = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)
        
        self.linear = nn.Linear(feature_size, feature_size)
        self.norm = nn.LayerNorm(feature_size)

    def split_heads(self, x, batch_size):
        # 分割最后一个维度到 (num_heads, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # 重排为(batch_size, num_heads, seq_len * num_features, depth)

    def forward(self, x):
        batch_size, num_tensors, seq_len, num_features, feature_size = x.shape
        
        # 重塑输入以适应多头注意力
        x_reshaped = x.view(batch_size, num_tensors * seq_len * num_features, feature_size)
        
        # 线性变换并分割头
        query = self.split_heads(self.query(x_reshaped), batch_size)
        key = self.split_heads(self.key(x_reshaped), batch_size)
        value = self.split_heads(self.value(x_reshaped), batch_size)
        
        # 缩放点积注意力
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        scores = matmul_qk / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重
        output = torch.matmul(attention_weights, value)
        
        # 合并头
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, num_tensors, seq_len, num_features, feature_size)
        
        # 在num_tensors维度上求和
        output = output.sum(dim=1)
        
        # 通过最终线性层
        output = self.linear(output)
        
        # 添加层归一化
        output = self.norm(output)
        
        return output
    
     
class AttPredictionDecoder(nn.Module):
    def __init__(self, neighbors):
        super(AttPredictionDecoder, self).__init__()
        self._agents = neighbors + 1
        self.predictor = GMMPredictor()

    def forward(self, current_states, att_content):
        N = self._agents
        predictions, scores = self.predictor(att_content)
        predictions[..., :2] += current_states[:, :N, None, None, :2]
        return att_content, predictions, scores   
    
    
    