"""Small independent checks; run with python -m unittest discover -s lab3."""

import unittest

import torch

import lab3


class Lab3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.x = torch.randn(11, 13, dtype=torch.float64)

    def test_asymmetric_shift_and_circular_edges(self):
        a = torch.zeros(3, 5, dtype=torch.float64)
        a[0, 4] = 1.0  # Offset (-1, +2) from the kernel center.
        torch.testing.assert_close(lab3.forward(self.x, a), self.x.roll((-1, 2), (0, 1)))
        torch.testing.assert_close(lab3.adjoint(self.x, a), self.x.roll((1, -2), (0, 1)))

    def test_adjoint_and_gradient_against_autograd(self):
        a = torch.rand(3, 5, dtype=torch.float64)
        a /= a.sum()
        torch.testing.assert_close(lab3.adjoint(self.x, a), lab3.adjoint_autograd(self.x, a))
        z = self.x.clone().requires_grad_()
        y = torch.randn_like(z)
        automatic = torch.autograd.grad(lab3.cost(z, y, a, 0.02, 0.05), z)[0]
        torch.testing.assert_close(automatic, lab3.cost_grad(z.detach(), y, a, 0.02, 0.05))

    def test_prior_against_independent_unordered_pair_sum(self):
        x, sigma_x = self.x, 0.05
        pairs = ((x[:, 1:] - x[:, :-1]).square().sum()
                 + (x[1:, :] - x[:-1, :]).square().sum()) / 6.0
        pairs += ((x[1:, 1:] - x[:-1, :-1]).square().sum()
                  + (x[1:, :-1] - x[:-1, 1:]).square().sum()) / 12.0
        torch.testing.assert_close(lab3.prior_cost(x, sigma_x), pairs / (2 * sigma_x**2))

    def test_prior_constant_and_corner_boundary(self):
        torch.testing.assert_close(lab3.prior_grad(torch.ones_like(self.x), 1.0),
                                   torch.zeros_like(self.x), atol=1e-14, rtol=0)
        impulse = torch.zeros_like(self.x)
        impulse[0, 0] = 1.0
        expected = torch.zeros_like(self.x)
        expected[0, 0] = 5 / 12
        expected[1, 0] = expected[0, 1] = -1 / 6
        expected[1, 1] = -1 / 12
        torch.testing.assert_close(lab3.prior_grad(impulse, 1.0), expected)

    def test_descent_contract_and_stationarity(self):
        a = torch.ones(1, 1, dtype=torch.float64)
        original = self.x.clone()
        result, history = lab3.gradient_descent(self.x, a, 0.05, 0.05, 100)
        self.assertEqual(len(history), 101)
        self.assertEqual(history[0], float(lab3.cost(self.x, self.x, a, 0.05, 0.05)))
        self.assertAlmostEqual(history[-1], float(lab3.cost(result, self.x, a, 0.05, 0.05)))
        self.assertLess(float(lab3.cost_grad(result, self.x, a, 0.05, 0.05).abs().max()), 1e-10)
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(history, history[1:])))
        torch.testing.assert_close(original, self.x, rtol=0, atol=0)
        zero_result, zero_history = lab3.gradient_descent(self.x, a, 0.05, 0.05, 0)
        torch.testing.assert_close(zero_result, self.x)
        self.assertEqual(len(zero_history), 1)

    def test_diagnostics_match_required_interface(self):
        a = torch.ones(1, 1, dtype=torch.float64)
        result, history = lab3.gradient_descent(self.x, a, 0.05, 0.05, 4)
        traced, traced_cost, changes = lab3.gradient_descent_diagnostics(self.x, a, 0.05, 0.05, 4)
        torch.testing.assert_close(result, traced, atol=0, rtol=0)
        self.assertEqual(history, traced_cost)
        self.assertEqual(len(changes), 4)
        expected = self.x - lab3.cost_grad(self.x, self.x, a, 0.05, 0.05) / 1200
        self.assertAlmostEqual(changes[0], float(torch.linalg.vector_norm(expected - self.x)
                                               / torch.linalg.vector_norm(self.x)))


if __name__ == "__main__":
    unittest.main()
