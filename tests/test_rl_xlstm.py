import unittest

import torch

from rl.src.xlstm_actor import mLSTMCell, sLSTMCell, xLSTMActor


class xLSTMStabilityTests(unittest.TestCase):
    def test_slstm_and_mlstm_cells_remain_finite_under_large_inputs(self):
        torch.manual_seed(0)
        x = torch.randn(4, 6) * 50.0

        slstm = sLSTMCell(input_dim=6, hidden_dim=8)
        h = torch.zeros(4, 8)
        c = torch.zeros(4, 8)
        m = torch.zeros(4, 8)
        h_next, c_next, m_next = slstm(x, h, c, m)
        self.assertTrue(torch.isfinite(h_next).all())
        self.assertTrue(torch.isfinite(c_next).all())
        self.assertTrue(torch.isfinite(m_next).all())

        mlstm = mLSTMCell(input_dim=6, hidden_dim=8)
        C = torch.zeros(4, 8, 8)
        m = torch.zeros(4, 8)
        h_next, C_next, m_next = mlstm(x, C, m)
        self.assertTrue(torch.isfinite(h_next).all())
        self.assertTrue(torch.isfinite(C_next).all())
        self.assertTrue(torch.isfinite(m_next).all())

    def test_xlstm_actor_outputs_bounded_finite_actions(self):
        torch.manual_seed(1)
        actor = xLSTMActor(
            obs_dim=5,
            act_dim=3,
            embed_dim=16,
            n_slstm_layers=1,
            n_mlstm_layers=1,
            max_action=2.0,
        )
        obs_seq = torch.randn(2, 8, 5)
        act_seq = torch.randn(2, 8, 3).clamp(-1.0, 1.0)
        actions = actor(obs_seq, act_seq)

        self.assertEqual(actions.shape, (2, 3))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertLessEqual(float(actions.detach().abs().max()), 2.0)


if __name__ == "__main__":
    unittest.main()
