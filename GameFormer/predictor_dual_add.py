import torch
from .predictor_modules import *

torch.cuda.set_device(1)

class Encoder(nn.Module):
    def __init__(self, dim=256, layers=6, heads=8, dropout=0.1):
        super(Encoder, self).__init__()
        self._lane_len = 50
        self._lane_feature = 7
        self._route_len = 50
        self._route_feature = 3
        self.agent_encoder = AgentEncoder(agent_dim=11)   # LSTM
        self.ego_encoder = AgentEncoder(agent_dim=7)
        self.lanenet = LaneNet().cuda()
        self.routenet = RouteNet().cuda()
        self.m2m = M2M()

        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation='gelu', dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers)

    def forward(self, inputs):
        # agents
        ego = inputs['ego_agent_past']
        neighbors = inputs['neighbor_agents_past']
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1)

        # agent encoding
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)

        # vector maps
        map_lanes = inputs['map_lanes']
        route_lanes = inputs['route_lanes'] 

        # map encoding
        encoded_lane, map_mask = self.lanenet(map_lanes)
        encoded_route = self.routenet(route_lanes)
        encoded_map = self.m2m(encoded_lane, encoded_route)

        # attention fusion encoding
        input = torch.cat([encoded_actors, encoded_map], dim=1)
        mask = torch.cat([actors_mask, map_mask], dim=1)
        encoding = self.fusion_encoder(input, src_key_padding_mask=mask)

        # outputs
        encoder_outputs = {
            'actors': actors,
            'encoding': encoding,
            'mask': mask,
            'route_lanes': inputs['route_lanes']
        }

        return encoder_outputs


class Decoder(nn.Module):
    def __init__(self, neighbors=10, modalities=6, levels=3):
        super(Decoder, self).__init__()
        self.levels = levels
        future_encoder = FutureEncoder()

        # initial level
        self.initial_predictor = InitialPredictionDecoder(modalities, neighbors)

        # level-k reasoning
        self.interaction_stage = nn.ModuleList([InteractionDecoder(modalities, future_encoder) for _ in range(levels)])

        self.content_feature_extractor = ContentFeatureExtractor(256, 256)
        self.feature_fusion_attention = FeatureFusionAttention(feature_size=256, num_features=6)
        self.att_predictor = AttPredictionDecoder(neighbors)
        
    def forward(self, encoder_outputs):
        decoder_outputs = {}
        current_states = encoder_outputs['actors'][:, :, -1]
        encoding, mask = encoder_outputs['encoding'], encoder_outputs['mask']

        # level 0 decode
        last_content, last_level, last_score = self.initial_predictor(current_states, encoding, mask)
        decoder_outputs['level_0_interactions'] = last_level
        decoder_outputs['level_0_scores'] = last_score
        
        content_list = [last_content]
        # level k reasoning
        for k in range(1, self.levels+1):
            interaction_decoder = self.interaction_stage[k-1]
            last_content, last_level, last_score = interaction_decoder(current_states, last_level, last_score, last_content, encoding, mask)
            decoder_outputs[f'level_{k}_interactions'] = last_level
            decoder_outputs[f'level_{k}_scores'] = last_score
            content_list.append(last_content)
        
        # env_encoding = last_content[:, 0]
        
        # level k content feature extractor and attention
        content_dim = []
        for input in content_list:
            batch_size, num_v, modalities, feature_dim = input.shape
            out = self.content_feature_extractor(input.view(-1, feature_dim)).view(batch_size, num_v, modalities, -1)
            content_dim.append(out)
            
        # combined_feature_tensors = torch.stack((content_dim[0], content_dim[1], content_dim[2]), dim=1) 
        # attention_output = self.feature_fusion_attention(combined_feature_tensors)
        add_output = content_dim[0] + content_dim[1] + content_dim[2]
        att_content, predictions, scores = self.att_predictor(current_states, add_output)
        
        decoder_outputs['level_3_interactions'] = predictions
        decoder_outputs['level_3_scores'] = scores       
        
        return decoder_outputs, att_content[:, 0]


class NeuralPlanner(nn.Module):
    def __init__(self):
        super(NeuralPlanner, self).__init__()
        self._future_len = 80
        self.route_fusion = CrossTransformer()
        self.plan_decoder = nn.Sequential(nn.Linear(512, 256), nn.ELU(), nn.Dropout(0.1), nn.Linear(256, self._future_len*2))
        self.route_encoder = VectorMapEncoder(3, 50)

    def dynamics_layer(self, controls, initial_state):       
        dt = 0.1 # discrete time period [s]
        max_a = 5 # vehicle's accleration limits [m/s^2]
        max_d = 0.5 # vehicle's steering limits [rad]
        
        vel_init = torch.hypot(initial_state[..., 3], initial_state[..., 4])
        vel = vel_init[:, None] + torch.cumsum(controls[..., 0].clamp(-max_a, max_a) * dt, dim=-1)
        vel = torch.clamp(vel, min=0)

        yaw_rate = controls[..., 1].clamp(-max_d, max_d) * vel
        yaw = initial_state[:, None, 2] + torch.cumsum(yaw_rate * dt, dim=-1)
        pi = torch.tensor(math.pi)
        yaw = torch.fmod(yaw, 2 * pi)
        # yaw = torch.fmod(yaw, 2*torch.pi)

        vel_x = vel * torch.cos(yaw)
        vel_y = vel * torch.sin(yaw)

        x = initial_state[:, None, 0] + torch.cumsum(vel_x * dt, dim=-1)
        y = initial_state[:, None, 1] + torch.cumsum(vel_y * dt, dim=-1)

        return torch.stack((x, y, yaw), dim=-1)

    def forward(self, env_encoding, route_lanes, initial_state):
        route_lanes, mask = self.route_encoder(route_lanes)
        mask[:, 0] = False
        route_encoding = self.route_fusion(env_encoding, route_lanes, route_lanes, mask)
        env_route_encoding = torch.cat([env_encoding, route_encoding], dim=-1)
        # dim0是最大值value，dim1是索引
        env_route_encoding = torch.max(env_route_encoding, dim=1)[0] # max pooling over modalities
        control = self.plan_decoder(env_route_encoding)
        control = control.reshape(-1, self._future_len, 2)
        plan = self.dynamics_layer(control, initial_state)

        return plan
    
    
class GameFormer(nn.Module):
    def __init__(self, encoder_layers=6, decoder_levels=3, modalities=6, neighbors=10):
        super(GameFormer, self).__init__()
        self.encoder = Encoder(layers=encoder_layers)
        self.decoder = Decoder(neighbors, modalities, decoder_levels)
        self.planner = NeuralPlanner()

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        route_lanes = encoder_outputs['route_lanes']
        initial_state = encoder_outputs['actors'][:, 0, -1]
        decoder_outputs, env_encoding = self.decoder(encoder_outputs)
        ego_plan = self.planner(env_encoding, route_lanes, initial_state)

        return decoder_outputs, ego_plan
    
    
    
    
    
    
    