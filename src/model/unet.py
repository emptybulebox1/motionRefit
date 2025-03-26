import torch
import torch.nn as nn
import math


class Unet(nn.Module):
    def __init__(
            self,
            dim_model,
            num_heads,
            num_layers,
            dropout_p,
            dim_input,
            dim_output,
            free_p=0.1,
            text_emb=True,
            device='cuda',          
            **kwargs
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.text_emb = text_emb
        self.dim_input = dim_input
        self.device = device
        try:
            self.Disc = kwargs['Disc']
        except:
            self.Disc = False

        # layers
        self.free_p = free_p
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding_input = nn.Linear(dim_input, dim_model)
        self.embedding_original = nn.Linear(dim_input, dim_model)


        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_model*4,
                                                   dropout=dropout_p,
                                                   activation="gelu",
                                                   )

        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers,
                                                )
        if self.Disc:
            # for discriminator
            self.pred = nn.Sequential(nn.Linear(dim_output, dim_output),
                                  nn.SiLU(inplace=False),
                                  nn.Linear(dim_output, 1),
                                  nn.Sigmoid())
            
        self.out = nn.Linear(dim_model, dim_output)

        self.embed_timestep = TimestepEmbedder(self.dim_model, self.positional_encoder)
        
        if self.text_emb: 
            #for embedding progress indicator
            print("text embedding is enabled!")
            self.positional_encoder_pi = PositionalEncoding(
                dim_model=dim_model, dropout_p=dropout_p, max_len=5000
            )
            self.embed_prog_ind = ProgIndEmbedder(self.dim_model, self.positional_encoder_pi)

    def forward_disc(self, x, timesteps):
        t_emb = self.embed_timestep(timesteps) # t_emb refers to time embedding
        
        x, t_emb = x.permute(1, 0, 2), t_emb.permute(1, 0, 2)
        x = self.embedding_input(x) * math.sqrt(self.dim_model)
    
        x = torch.cat((t_emb, x), dim=0)
        x = self.positional_encoder(x)
        x = self.transformer(x)
        output = self.out(x)[1:]
        output = output.permute(1, 0, 2)
        output = output.mean(dim=1)
        output = self.pred(output)
        return output
    
    def forward_(self, x, timesteps, text_emb=None, prog_ind=None, joints_orig=None): 
        t_emb = self.embed_timestep(timesteps) # t_emb refers to time embedding
        if self.text_emb:
            text_emb = text_emb.unsqueeze(1) # batchsize, 1, 512
            assert text_emb.shape == (x.shape[0], 1, self.dim_model), \
                f'text_emb shape should be (batchsize, 1, {self.dim_model})'
        
        x, joints_orig, t_emb = x.permute(1, 0, 2), joints_orig.permute(1, 0, 2), t_emb.permute(1, 0, 2)
        x = self.embedding_input(x) * math.sqrt(self.dim_model)
        joints_orig = self.embedding_original(joints_orig) * math.sqrt(self.dim_model)
        x = (x + joints_orig) / 2.

        if not self.text_emb:
            x = torch.cat((t_emb, x), dim=0) # (seq_len+1), batchsize, dim_model
        else:
            text_emb = text_emb.permute(1, 0, 2)
            prog_ind = (prog_ind*100).round().to(torch.int64)
            prog_ind_emb = self.embed_prog_ind(prog_ind).permute(1, 0, 2)
            t_emb = (t_emb + text_emb/10.0 + prog_ind_emb) * math.sqrt(self.dim_model)
            x = torch.cat((t_emb, x), dim=0)

        x = self.positional_encoder(x)
        x = self.transformer(x)
        output = self.out(x)[1:]
        output = output.permute(1, 0, 2)
        return output

    def forward(self, x, timesteps, text_emb=None, prog_ind=None, joints_orig=None):
        if self.Disc:
            return self.forward_disc(x, timesteps)
        else:
            return self.forward_(x, timesteps, text_emb, prog_ind, joints_orig)


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).reshape(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(inplace=False),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pos_encoding[timesteps])#.permute(1, 0, 2)

# totally the same as TimeStepEmbedder
class ProgIndEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(inplace=False),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pos_encoding[timesteps])#.permute(1, 0, 2)
