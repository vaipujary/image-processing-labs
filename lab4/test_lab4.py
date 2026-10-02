"""Independent numerical checks of the lab equations, boundaries and optimizer."""
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab4.lab4 import (rho, rho_prime, neighbor_diffs, surrogate_weights,
                      true_cost, true_gradient, optimal_step_size, reconstruct,
                      forward, adjoint)


class Lab4Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.gen = torch.Generator().manual_seed(41)
        self.x = torch.randn((7, 9), generator=self.gen, dtype=torch.float64)
        self.y = torch.randn((7, 9), generator=self.gen, dtype=torch.float64)
        self.a = torch.tensor([[0., .1, .2], [.05, .3, .1], [.05, .1, .1]], dtype=torch.float64)
        self.params = (.02, .02, 2., 1.2, 1.)

    def test_potential_and_influence(self):
        d = torch.tensor([-1., -.3, -.01, -1e-12, 0., 1e-12, .01, .3, 1.],
                         dtype=torch.float64, requires_grad=True)
        actual = rho(d, .2, 2., 1.2, 1.)
        derivative = torch.autograd.grad(actual.sum(), d)[0]
        torch.testing.assert_close(derivative, rho_prime(d, .2, 2., 1.2, 1.), atol=1e-13, rtol=1e-13)
        self.assertTrue(torch.isfinite(derivative).all())
        nz = d.detach()[d.abs() > 1e-8]
        z = (nz.abs()/.2).pow(-.8)
        expected = nz.square()/(2*.2**2) * z/(1+z)
        torch.testing.assert_close(rho(nz,.2,2.,1.2,1.), expected)

    def test_neighbors_and_independent_pair_cost(self):
        x = torch.arange(12, dtype=torch.float64).reshape(3,4)
        d, valid = neighbor_diffs(x)
        self.assertEqual(d.shape, (8,3,4))
        self.assertEqual(valid.sum().item(), 58)
        self.assertEqual(d[0,1,1].item(), 5.)
        self.assertEqual(d[4,1,1].item(), -1.)
        self.assertTrue((d[~valid]==0).all())
        # Count every unordered pair exactly once using slices, independently
        # of the eight-plane routine; no pixel loop is needed.
        prior = (rho(self.x[:,1:]-self.x[:,:-1],.02,2.,1.2,1.).sum()/6
                 +rho(self.x[1:,:]-self.x[:-1,:],.02,2.,1.2,1.).sum()/6
                 +rho(self.x[1:,1:]-self.x[:-1,:-1],.02,2.,1.2,1.).sum()/12
                 +rho(self.x[1:,:-1]-self.x[:-1,1:],.02,2.,1.2,1.).sum()/12)
        identity = torch.ones((1,1),dtype=torch.float64)
        torch.testing.assert_close(true_cost(self.x,self.x,identity,*self.params),prior)

    def test_autograd_against_equation16(self):
        x = self.x.clone().requires_grad_(True)
        g = true_gradient(x,self.y,self.a,*self.params)
        d, valid = neighbor_diffs(x.detach())
        b = x.new_tensor([1/12,1/6,1/12,1/6,1/6,1/12,1/6,1/12])[:,None,None]
        ref = adjoint(forward(x.detach(),self.a)-self.y,self.a)/.02**2
        ref += (valid*b*rho_prime(d,.02,2.,1.2,1.)).sum(0)
        torch.testing.assert_close(g,ref,atol=2e-11,rtol=2e-12)

    def test_surrogate_zero_limit_and_hessian(self):
        zero = torch.zeros_like(self.x)
        w0 = surrogate_weights(zero,.02,2.,1.2,1.)
        self.assertAlmostEqual(w0[:,3,4].sum().item(),1/(2*.02**2))
        self.assertAlmostEqual(w0[:,0,0].sum().item(),(5/12)/(2*.02**2))
        weights = surrogate_weights(self.x,.02,2.,1.2,1.)
        x = self.x.clone().requires_grad_(True)
        d,_ = neighbor_diffs(x)
        Q = forward(x,self.a).square().sum()/(2*.02**2)+.5*(weights*d.square()).sum()
        qg = torch.autograd.grad(Q,x,create_graph=True)[0]
        v = self.y
        hv = torch.autograd.grad((qg*v).sum(),x)[0]
        vd,_ = neighbor_diffs(v)
        denominator = forward(v,self.a).square().sum()/.02**2+(weights*vd.square()).sum()
        torch.testing.assert_close((v*hv).sum(),denominator)
        alpha = optimal_step_size(self.x,v,self.a,*self.params)
        torch.testing.assert_close(alpha, v.square().sum()/denominator)

    def test_scalar_surrogate_bound(self):
        d = torch.linspace(-1,1,10001,dtype=torch.float64)
        for t in [0.,.02,.15]:
            t = d.new_tensor(t)
            coefficient = d.new_tensor(1/(2*.2**2)) if t==0 else rho_prime(t,.2,2.,1.2,1.)/(2*t)
            bound = coefficient*(d.square()-t.square())+rho(t,.2,2.,1.2,1.)
            self.assertGreaterEqual((bound-rho(d,.2,2.,1.2,1.)).min().item(),-1e-13)

    def test_reconstruction_and_stationary_case(self):
        for omega in [1.,1.9]:
            x,h = reconstruct(self.y,self.a,*self.params,12,omega)
            self.assertEqual(len(h),13)
            self.assertTrue(all(b<a for a,b in zip(h,h[1:])))
            self.assertTrue(torch.isfinite(x).all())
        zero = torch.zeros_like(self.y)
        x,h = reconstruct(zero,torch.ones((1,1),dtype=torch.float64),*self.params,3,1.)
        self.assertEqual(h,[0.]*4)
        torch.testing.assert_close(x,zero)


if __name__ == '__main__':
    unittest.main(verbosity=2)
