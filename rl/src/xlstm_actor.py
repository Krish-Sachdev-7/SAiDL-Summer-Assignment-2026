"""xLSTM actor bonus code."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class sLSTMCell(nn.Module):
    """Scalar-memory xLSTM cell."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.x_proj = nn.Linear(input_dim, 4 * hidden_dim)
        self.h_proj = nn.Linear(hidden_dim, 4 * hidden_dim)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor,
                m_prev: torch.Tensor) -> tuple:
        gates = self.x_proj(x_t) + self.h_proj(h_prev)
        i_pre, f_pre, o_pre, g_pre = gates.chunk(4, dim=-1)

        log_i = F.logsigmoid(i_pre)
        log_f = F.logsigmoid(f_pre)
        m_t = torch.maximum(log_f + m_prev, log_i)

        i_hat = torch.exp(log_i - m_t)
        f_hat = torch.exp(log_f + m_prev - m_t)

        g_t = torch.tanh(g_pre)
        c_t = f_hat * c_prev + i_hat * g_t
        h_t = torch.sigmoid(o_pre) * torch.tanh(c_t)
        return h_t, c_t, m_t


class mLSTMCell(nn.Module):
    """Matrix-memory xLSTM cell."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.x_proj = nn.Linear(input_dim, 6 * hidden_dim)

    def forward(self, x_t: torch.Tensor, C_prev: torch.Tensor,
                m_prev: torch.Tensor) -> tuple:
        gates = self.x_proj(x_t)
        i_pre, f_pre, o_pre, q_pre, k_pre, v_pre = gates.chunk(6, dim=-1)

        log_i = F.logsigmoid(i_pre)
        log_f = F.logsigmoid(f_pre)
        m_t = torch.maximum(log_f + m_prev, log_i)

        i_hat = torch.exp(log_i - m_t)
        f_hat = torch.exp(log_f + m_prev - m_t)

        q_t = torch.tanh(q_pre)
        k_t = torch.tanh(k_pre)
        v_t = torch.tanh(v_pre)

        outer = torch.einsum("bi,bj->bij", v_t, k_t)
        C_t = f_hat.unsqueeze(-1) * C_prev + i_hat.unsqueeze(-1) * outer

        read = torch.einsum("bij,bj->bi", C_t, q_t)
        denom = torch.maximum(read.abs().amax(dim=-1, keepdim=True), torch.ones_like(read[:, :1]))
        h_t = torch.sigmoid(o_pre) * (read / denom)
        return h_t, C_t, m_t


class xLSTMActor(nn.Module):
    """xLSTM actor for TD3."""
    def __init__(self, obs_dim: int, act_dim: int, embed_dim: int,
                 n_slstm_layers: int, n_mlstm_layers: int, max_action: float = 1.0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.embed_dim = int(embed_dim)
        self.max_action = float(max_action)

        self.input_proj = nn.Linear(self.obs_dim + self.act_dim, self.embed_dim)
        self.slstm_layers = nn.ModuleList(
            [sLSTMCell(self.embed_dim, self.embed_dim) for _ in range(int(n_slstm_layers))]
        )
        self.mlstm_layers = nn.ModuleList(
            [mLSTMCell(self.embed_dim, self.embed_dim) for _ in range(int(n_mlstm_layers))]
        )
        self.out_proj = nn.Linear(self.embed_dim, self.act_dim)

    def forward(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        bsz, seq_len, _ = obs_seq.shape
        x_seq = self.input_proj(torch.cat([obs_seq, act_seq], dim=-1))

        slstm_states = [
            {
                "h": torch.zeros(bsz, self.embed_dim, device=x_seq.device, dtype=x_seq.dtype),
                "c": torch.zeros(bsz, self.embed_dim, device=x_seq.device, dtype=x_seq.dtype),
                "m": torch.zeros(bsz, self.embed_dim, device=x_seq.device, dtype=x_seq.dtype),
            }
            for _ in self.slstm_layers
        ]
        mlstm_states = [
            {
                "C": torch.zeros(bsz, self.embed_dim, self.embed_dim, device=x_seq.device, dtype=x_seq.dtype),
                "m": torch.zeros(bsz, self.embed_dim, device=x_seq.device, dtype=x_seq.dtype),
            }
            for _ in self.mlstm_layers
        ]

        out_t = None
        for t in range(seq_len):
            x_t = x_seq[:, t]
            for i, cell in enumerate(self.slstm_layers):
                st = slstm_states[i]
                h, c, m = cell(x_t, st["h"], st["c"], st["m"])
                st["h"], st["c"], st["m"] = h, c, m
                x_t = h

            for i, cell in enumerate(self.mlstm_layers):
                st = mlstm_states[i]
                h, C, m = cell(x_t, st["C"], st["m"])
                st["C"], st["m"] = C, m
                x_t = h

            out_t = x_t

        action = torch.tanh(self.out_proj(out_t)) * self.max_action
        return action
