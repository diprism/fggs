import unittest

from fggs.indices import *
from fggs.semirings import RealSemiring
from itertools import permutations
from math import inf, nan
import torch

def nrand(*size) -> torch.Tensor:
    s = torch.Size((*size,))
    n = s.numel()
    return torch.zeros(s) if n == 0 else torch.arange(0.42-n, n, 2).view(s)

class TestEmbedding(unittest.TestCase):

    def setUp(self):
        self.k1  = EmbeddingVar(5)
        self.k1_ = EmbeddingVar(5)
        self.k2  = EmbeddingVar(2)
        self.k2_ = EmbeddingVar(2)
        self.k3  = EmbeddingVar(3)
        self.k3_ = EmbeddingVar(3)
        self.k4  = EmbeddingVar(6)

    def test_prime_factors(self):
        for e in [self.k2, self.k4, SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3 )),5)]:
            self.assertEqual([e], list(e.prime_factors({})))
        self.assertEqual([self.k2, self.k3],
                         list(ProductEmbedding((self.k2, self.k3)).prime_factors({})))
        self.assertEqual([self.k2, self.k3],
                         list(self.k4.prime_factors({self.k4: ProductEmbedding((self.k2, self.k3))})))

    def test_alpha(self):
        rename = {self.k2: self.k2_, self.k3: self.k3_}
        self.assertTrue (SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3 )),5).alpha(
                         SumEmbedding(4,ProductEmbedding((self.k2_,self.k3_)),5), rename))
        self.assertFalse(SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3 )),5).alpha(
                         SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3 )),5), rename))
        self.assertFalse(SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3 )),5).alpha(
                         SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3 )),5), {}))
        self.assertFalse(SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3 )),5).alpha(
                         SumEmbedding(4,ProductEmbedding((self.k3_,self.k2_)),5), rename))
        self.assertFalse(SumEmbedding(4,ProductEmbedding((self.k2_,self.k3 )),5).alpha(
                         SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3_)),5), rename))
        self.assertFalse(SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3_)),5).alpha(
                         SumEmbedding(4,ProductEmbedding((self.k2_,self.k3 )),5), rename))
        self.assertFalse(SumEmbedding(4,ProductEmbedding((self.k2_,self.k3_)),5).alpha(
                         SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3 )),5), rename))
        self.assertFalse(SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3 )),5).alpha(
                         SumEmbedding(3,ProductEmbedding((self.k2_,self.k3_)),5), rename))
        self.assertFalse(SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3 )),5).alpha(
                         SumEmbedding(4,ProductEmbedding((self.k2_,self.k3_)),6), rename))
        self.assertFalse(SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3 )),5).alpha(
                         SumEmbedding(4,self.k4                              ,5), rename))
        self.assertFalse(SumEmbedding(4,self.k4                              ,5).alpha(
                         SumEmbedding(4,ProductEmbedding((self.k2 ,self.k3 )),5), rename))
        self.assertFalse(SumEmbedding(1,self.k2,0).alpha(self.k3, rename))
        self.assertFalse(self.k3.alpha(SumEmbedding(1,self.k2,0), rename))
        self.assertFalse(ProductEmbedding((self.k2,self.k3)).alpha(SumEmbedding(1,self.k1,0), rename))
        self.assertFalse(SumEmbedding(1,self.k1,0).alpha(ProductEmbedding((self.k2,self.k3)), rename))

    def test_freshen(self):
        for e in [self.k1, SumEmbedding(4,ProductEmbedding((self.k2,self.k3)),5)]:
            rename = {}
            f = e.freshen(rename)
            self.assertTrue(frozenset(rename.keys()).isdisjoint(rename.values()) and
                            len(rename) == len(frozenset(rename.values())) and
                            e.alpha(f, rename))
            self.assertFalse(f.alpha(e, rename))
            self.assertFalse(e.alpha(f, {}))

    def test_unify_1(self):
        subst : Subst = {}
        self.assertTrue(SumEmbedding(1,self.k4                            ,2).unify(
                        SumEmbedding(1,ProductEmbedding((self.k2,self.k3)),2), subst))
        self.assertEqual(subst, {self.k4: ProductEmbedding((self.k2,self.k3))})
        self.assertTrue(SumEmbedding(1,self.k4                            ,2).unify(
                        SumEmbedding(1,ProductEmbedding((self.k2,self.k3)),2), subst))
        self.assertEqual(subst, {self.k4: ProductEmbedding((self.k2,self.k3))})

    def test_unify_2(self):
        subst : Subst = {}
        self.assertTrue(SumEmbedding(1,ProductEmbedding((self.k2,self.k3)),2).unify(
                        SumEmbedding(1,self.k4                            ,2), subst))
        self.assertEqual(subst, {self.k4: ProductEmbedding((self.k2,self.k3))})
        self.assertTrue(SumEmbedding(1,ProductEmbedding((self.k2,self.k3)),2).unify(
                        SumEmbedding(1,self.k4                            ,2), subst))
        self.assertEqual(subst, {self.k4: ProductEmbedding((self.k2,self.k3))})

    def test_unify_3(self):
        subst : Subst = {}
        self.assertFalse(SumEmbedding(1,self.k4,2).unify(
                         SumEmbedding(7,self.k2,0), subst))

    def test_unify_4(self):
        subst : Subst = {}
        self.assertTrue(SumEmbedding(1,ProductEmbedding((self.k2_,self.k3_)),2).unify(
                        SumEmbedding(1,ProductEmbedding((self.k2,self.k3)),2), subst))
        self.assertEqual(subst, {self.k2_: self.k2, self.k3_: self.k3})

class TestEmbeddedTensor(unittest.TestCase):

    def assertTEqual(self, input: Tensor, other: Tensor) -> None:
        self.assertTrue(torch.equal(input, other), (input, other))

    def assertTClose(self, input: Tensor, other: Tensor) -> None:
        self.assertTrue(torch.allclose(input, other, equal_nan=True), (input, other))

    def assertEEqual(self, input: EmbeddedTensor, other: EmbeddedTensor) -> None:
        self.assertTrue(input.equal(other))
        self.assertTrue(other.equal(input))
        self.assertTrue(input.allclose(other, atol=0.1000001, rtol=0))
        self.assertTrue(other.allclose(input, atol=0.1000001, rtol=0))

    def assertENotEqual(self, input: EmbeddedTensor, other: EmbeddedTensor) -> None:
        self.assertFalse(input.equal(other))
        self.assertFalse(other.equal(input))
        self.assertFalse(input.allclose(other, atol=0.1000001, rtol=0))
        self.assertFalse(other.allclose(input, atol=0.1000001, rtol=0))

    def setUp(self):
        self.k0  = EmbeddingVar(0)
        self.k0_ = EmbeddingVar(0)
        self.k1  = EmbeddingVar(5)
        self.k1_ = EmbeddingVar(5)
        self.k2  = EmbeddingVar(2)
        self.k2_ = EmbeddingVar(2)
        self.k3  = EmbeddingVar(3)
        self.k3_ = EmbeddingVar(3)
        self.k4  = EmbeddingVar(6)
        self.k4_ = EmbeddingVar(6)
        self.k5  = EmbeddingVar(5)
        self.k6  = EmbeddingVar(35)
        self.k7  = EmbeddingVar(7)
        self.k8  = EmbeddingVar(36)

    def test_diag(self):
        phys = nrand(5,2)
        virt = EmbeddedTensor(phys,
                              (self.k1,self.k2),
                              (self.k2,self.k1,self.k1))
        self.assertEqual(virt.numel(), 50)
        self.assertTEqual(virt.to_dense(),
                          phys.t().diag_embed(dim1=1, dim2=2))
        virt = EmbeddedTensor(phys,
                              (self.k1,self.k2),
                              (self.k2,self.k1,self.k1),
                              42)
        self.assertEqual(virt.numel(), 50)
        self.assertTEqual(virt.to_dense(),
                          ((1-torch.eye(5))*42).unsqueeze(0).expand(2,5,5)
                          + phys.t().diag_embed(dim1=1, dim2=2))

    def test_algebraic(self):
        ones = torch.ones(2,3)
        diag = EmbeddedTensor(ones,
                              (self.k2,self.k3),
                              (self.k2,self.k3,SumEmbedding(1,ProductEmbedding((self.k2,self.k3)),2)))
        self.assertEqual(diag.numel(), 54)
        self.assertTEqual(diag.to_dense(),
                          torch.tensor([[[0,1,0,0,0,0,0,0,0],
                                         [0,0,1,0,0,0,0,0,0],
                                         [0,0,0,1,0,0,0,0,0]],
                                        [[0,0,0,0,1,0,0,0,0],
                                         [0,0,0,0,0,1,0,0,0],
                                         [0,0,0,0,0,0,1,0,0.0]]]))
        diag = EmbeddedTensor(ones,
                              (self.k2,self.k3),
                              (self.k2,self.k3,SumEmbedding(1,ProductEmbedding((self.k2,self.k3)),2)),
                              7)
        self.assertEqual(diag.numel(), 54)
        self.assertTEqual(diag.to_dense(),
                          torch.tensor([[[7,1,7,7,7,7,7,7,7],
                                         [7,7,1,7,7,7,7,7,7],
                                         [7,7,7,1,7,7,7,7,7]],
                                        [[7,7,7,7,1,7,7,7,7],
                                         [7,7,7,7,7,1,7,7,7],
                                         [7,7,7,7,7,7,1,7,7.0]]]))

    def test_copy(self):
        tensors = [EmbeddedTensor(nrand(5,2), (self.k1,self.k2), (self.k2,self.k1,self.k1)),
                   EmbeddedTensor(nrand(2,5), (self.k2,self.k1), (self.k2,self.k1,self.k1)),
                   EmbeddedTensor(torch.ones(2,3), (self.k2,self.k3), (self.k2,self.k3,SumEmbedding(1,ProductEmbedding((self.k2,self.k3)),2))),
                   EmbeddedTensor(torch.ones(1,1).expand(2,3), (self.k2,self.k3), (self.k2,self.k3,SumEmbedding(1,ProductEmbedding((self.k2,self.k3)),2))),
                   EmbeddedTensor(nrand(5,2), (self.k1,self.k2), (self.k2,self.k1,self.k1), 42),
                   EmbeddedTensor(nrand(2,5), (self.k2,self.k1), (self.k2,self.k1,self.k1), 43),
                   EmbeddedTensor(torch.ones(2,3), (self.k2,self.k3), (self.k2,self.k3,SumEmbedding(1,ProductEmbedding((self.k2,self.k3)),2)), 44),
                   EmbeddedTensor(torch.ones(1,1).expand(2,3), (self.k2,self.k3), (self.k2,self.k3,SumEmbedding(1,ProductEmbedding((self.k2,self.k3)),2)), 45)]
        for t1 in tensors:
            for t2 in tensors:
                t1_ = t1.clone()
                self.assertTrue(t1.equal(t1_) and t1_.equal(t1))
                t2_ = t2.clone()
                self.assertTrue(t2.equal(t2_) and t2_.equal(t2))
                t1_.copy_(t2_)
                self.assertTrue(t1_.equal(t2_) and t2_.equal(t1_))
                self.assertTEqual(t1_.to_dense(), t2_.to_dense())

    def test_equal_default(self):
        self.assertFalse(EmbeddedTensor(torch.Tensor([-1,0,1]), (self.k3,), (self.k3, self.k3)).equal_default())
        self.assertTrue (EmbeddedTensor(torch.Tensor([ 0,0,0]), (self.k3,), (self.k3, self.k3)).equal_default())
        self.assertFalse(EmbeddedTensor(torch.Tensor([-1,0,1]), (self.k3,), (self.k3, self.k3)).allclose_default())
        self.assertFalse(EmbeddedTensor(torch.Tensor([-1,0,1]), (self.k3,), (self.k3, self.k3)).allclose_default(rtol=9))
        self.assertTrue (EmbeddedTensor(torch.Tensor([-1,0,1]), (self.k3,), (self.k3, self.k3)).allclose_default(atol=1))
        self.assertTrue (EmbeddedTensor(torch.Tensor([-1,0,1]), (self.k3,), (self.k3, self.k3)).allclose_default(atol=1, rtol=9))
        self.assertTrue (EmbeddedTensor(torch.Tensor([ 0,0,0]), (self.k3,), (self.k3, self.k3)).allclose_default())

    def test_equal(self):
        t1 = EmbeddedTensor(torch.diag(torch.Tensor([1,2,3])))
        t2 = EmbeddedTensor(torch.Tensor([1,2,3]), (self.k3,), (self.k3, self.k3))
        self.assertEEqual(t1, t2)
        t2 = EmbeddedTensor(torch.Tensor([1,2,3]), (self.k3,), (self.k3, SumEmbedding(0,self.k3,2)))
        self.assertENotEqual(t1, t2)
        t2 = EmbeddedTensor(torch.Tensor([1,2,4]), (self.k3,), (self.k3, self.k3))
        self.assertENotEqual(t1, t2)
        t1 = EmbeddedTensor(torch.Tensor([5,6,7]), (self.k3,), (SumEmbedding(0,ProductEmbedding(()),2), self.k3))
        t2 = EmbeddedTensor(torch.Tensor([5,6,7]), (self.k3,), (self.k3, SumEmbedding(0,ProductEmbedding(()),2)))
        self.assertENotEqual(t1, t2)
        t1 = EmbeddedTensor(torch.Tensor([0,6,0]), (self.k3,), (SumEmbedding(0,ProductEmbedding(()),2), self.k3))
        t2 = EmbeddedTensor(torch.Tensor([0,6,0]), (self.k3,), (self.k3, SumEmbedding(0,ProductEmbedding(()),2)))
        self.assertENotEqual(t1, t2)
        t1 = EmbeddedTensor(torch.Tensor([5,0,0]), (self.k3,), (SumEmbedding(0,ProductEmbedding(()),2), self.k3))
        t2 = EmbeddedTensor(torch.Tensor([5,0,0]), (self.k3,), (self.k3, SumEmbedding(0,ProductEmbedding(()),2)))
        self.assertEEqual(t1, t2)
        t1 = EmbeddedTensor(torch.Tensor([1,2,3]))
        t2 = EmbeddedTensor(torch.Tensor([1,2,3]), default=-1)
        self.assertEEqual(t1, t2)
        t1 = EmbeddedTensor(torch.Tensor([1,2,3]), (self.k3,), (self.k3, self.k3))
        t2 = EmbeddedTensor(torch.Tensor([1,2,3]), (self.k3,), (self.k3, self.k3), default=-1)
        self.assertENotEqual(t1, t2)
        t1 = EmbeddedTensor(torch.diag(torch.Tensor([2,3,4]))-1, default=-1)
        t2 = EmbeddedTensor(torch.Tensor([1,2,3]), (self.k3,), (self.k3, self.k3), -1)
        self.assertEEqual(t1, t2)
        t2 = EmbeddedTensor(torch.Tensor([1,2,3]), (self.k3,), (self.k3, self.k3))
        self.assertENotEqual(t1, t2)
        t2 = EmbeddedTensor(torch.Tensor([1,2,3]), (self.k3,), (self.k3, SumEmbedding(0,self.k3,2)), -1)
        self.assertENotEqual(t1, t2)
        t2 = EmbeddedTensor(torch.Tensor([1,2,4]), (self.k3,), (self.k3, self.k3), -1)
        self.assertENotEqual(t1, t2)
        t1 = EmbeddedTensor(torch.Tensor([5,6,7]), (self.k3,), (SumEmbedding(0,ProductEmbedding(()),2), self.k3), -1)
        t2 = EmbeddedTensor(torch.Tensor([5,6,7]), (self.k3,), (self.k3, SumEmbedding(0,ProductEmbedding(()),2)), -1)
        self.assertENotEqual(t1, t2)
        t1 = EmbeddedTensor(torch.Tensor([0,6,0]), (self.k3,), (SumEmbedding(0,ProductEmbedding(()),2), self.k3), -1)
        t2 = EmbeddedTensor(torch.Tensor([0,6,0]), (self.k3,), (self.k3, SumEmbedding(0,ProductEmbedding(()),2)), -1)
        self.assertENotEqual(t1, t2)
        t1 = EmbeddedTensor(torch.Tensor([5,0,0]), (self.k3,), (SumEmbedding(0,ProductEmbedding(()),2), self.k3), -1)
        t2 = EmbeddedTensor(torch.Tensor([5,0,0]), (self.k3,), (self.k3, SumEmbedding(0,ProductEmbedding(()),2)), -1)
        self.assertENotEqual(t1, t2)
        t1 = EmbeddedTensor(torch.Tensor([5,-1,-1]), (self.k3,), (SumEmbedding(0,ProductEmbedding(()),2), self.k3), -1)
        t2 = EmbeddedTensor(torch.Tensor([5,-1,-1]), (self.k3,), (self.k3, SumEmbedding(0,ProductEmbedding(()),2)), -1)
        self.assertEEqual(t1, t2)

    def test_allclose(self):
        t1 = EmbeddedTensor(torch.Tensor([0,1,2]), (self.k3,), (self.k3, self.k3), -0.1)
        t2 = EmbeddedTensor(torch.Tensor([0,1,2]), (self.k3,), (self.k3, self.k3), -0.1)
        self.assertTrue(t1.allclose(t2, atol=0, rtol=0.1))
        self.assertTrue(t2.allclose(t1, atol=0, rtol=0.1))
        t2 = EmbeddedTensor(torch.Tensor([0.1,1.1,2.1]), (self.k3,), (self.k3, self.k3), -0.1)
        self.assertTrue(t1.allclose(t2, atol=0.05, rtol=0.5))
        self.assertFalse(t2.allclose(t1, atol=0.05, rtol=0.5))
        t2 = EmbeddedTensor(torch.Tensor([0,1,2]), (self.k3,), (self.k3, self.k3), 0)
        self.assertFalse(t1.allclose(t2, atol=0.05, rtol=0.5))
        self.assertTrue(t2.allclose(t1, atol=0.05, rtol=0.5))

    def test_einsum(self):
        matrix = nrand(36)
        vector = nrand(7)
        semiring = RealSemiring(dtype=matrix.dtype, device=matrix.device)
        self.assertTClose(matrix[1:].reshape((5,7)).matmul(vector),
                          einsum([EmbeddedTensor(matrix, (self.k8,), (self.k8,)),
                                  EmbeddedTensor(vector, (self.k7,), (self.k7,)),
                                  # Here's a sum-type factor represented compactly:
                                  EmbeddedTensor(torch.tensor(1).unsqueeze_(0).expand([35]),
                                                 (self.k6,),
                                                 (SumEmbedding(1,self.k6,0),self.k6)),
                                  # Here's a product-type factor represented compactly:
                                  EmbeddedTensor(torch.tensor(1).unsqueeze_(0).unsqueeze_(0).expand([5,7]),
                                                 (self.k5,self.k7),
                                                 (ProductEmbedding((self.k5,self.k7)),self.k5,self.k7))],
                                 ("m", "i", "mf", "foi"),
                                 "o",
                                 semiring).to_dense())
        self.assertTEqual(torch.ones(()),
                          einsum([], [], [], semiring).to_dense())
        self.assertTEqual(torch.zeros((6)),
                          einsum([EmbeddedTensor(matrix.reshape((6,6)), (self.k4,self.k4_),
                                                 (self.k4,SumEmbedding(7,self.k4_,0))),
                                  EmbeddedTensor(vector, (self.k7,),
                                                 (SumEmbedding(0,self.k7,6),))],
                                 [["o","i"], ["i"]],
                                 ["o"],
                                 semiring).to_dense())

    def test_binary(self):
        tensors = [t for default in [-inf, -2, 0, 1, inf, nan]
                     for t in [EmbeddedTensor(nrand(5,2),
                                              (self.k1,self.k2),
                                              (self.k2,self.k1,self.k1),
                                              default),
                               EmbeddedTensor(nrand(5,5),
                                              (self.k1,self.k1_),
                                              (SumEmbedding(0,ProductEmbedding(()),1),self.k1_,self.k1),
                                              default),
                               EmbeddedTensor(torch.arange(0.0,500,10).reshape(5,2,5),
                                              (self.k1,self.k2,self.k1_),
                                              (self.k2,self.k1_,self.k1),
                                              default)]]
        for t1 in tensors:
            for t2 in tensors:
                self.assertTClose(t1.add(t2).to_dense(),
                                  t1.to_dense().add(t2.to_dense()))
                self.assertTClose(t2.add(t1).to_dense(),
                                  t2.to_dense().add(t1.to_dense()))
                self.assertTClose(t1.sub(t2).to_dense(),
                                  t1.to_dense().sub(t2.to_dense()))
                self.assertTClose(t2.sub(t1).to_dense(),
                                  t2.to_dense().sub(t1.to_dense()))
                self.assertTClose(t1.logaddexp(t2).to_dense(),
                                  t1.to_dense().logaddexp(t2.to_dense()))
                self.assertTClose(t2.logaddexp(t1).to_dense(),
                                  t2.to_dense().logaddexp(t1.to_dense()))

    def test_transpose(self):
        for t in [EmbeddedTensor(nrand(3,5), (self.k3,self.k5), (self.k3,self.k5), 42),
                  EmbeddedTensor(nrand(3,5), (self.k3,self.k5), (ProductEmbedding(()),self.k3,self.k5), 42),
                  EmbeddedTensor(nrand(3,5), (self.k3,self.k5), (self.k5,self.k5,self.k3), 42),
                  EmbeddedTensor(nrand(3  ), (self.k3,       ), (self.k3,self.k3), 42),
                  EmbeddedTensor(nrand(3  ), (self.k3,       ), (self.k3,), 42)]:
            for dim0 in range(t.ndim):
                for dim1 in range(t.ndim):
                    self.assertTEqual(t.transpose(dim0,dim1).to_dense(),
                                      t.to_dense().transpose(dim0,dim1))
            if t.ndim <= 2: self.assertTEqual(t.t().to_dense(), t.to_dense().t())
            self.assertTEqual(t.T.to_dense(), t.to_dense().T)
            self.assertTEqual(t.flatten().to_dense(), t.to_dense().flatten())

    def test_reshape(self):
        ki  = EmbeddingVar(1)
        ki_ = EmbeddingVar(1)
        for vembeds, s in [((self.k2, self.k3), (6,)),
                           ((self.k2, self.k3), (6,1)),
                           ((self.k2, self.k3), (6,1,1)),
                           ((self.k2, self.k3, ki), (6,)),
                           ((self.k2, self.k3, ki), (6,1)),
                           ((self.k2, self.k3, ki), (6,1,1)),
                           ((self.k2, ki_, self.k3, ki), (6,)),
                           ((self.k2, ki_, self.k3, ki), (6,1)),
                           ((self.k2, ki_, self.k3, ki), (6,1,1)),
                           ((self.k2, self.k3, ki, ki_), (6,)),
                           ((self.k2, self.k3, ki, ki_), (6,1)),
                           ((self.k2, self.k3, ki, ki_), (6,1,1)),
                           ((), ()),
                           ((), (1,)),
                           ((), (1,1)),
                           ((ki,), ()),
                           ((ki,), (1,)),
                           ((ki,), (1,1)),
                           ((ki, ki_), ()),
                           ((ki, ki_), (1,)),
                           ((ki, ki_), (1,1)),
                           ((ProductEmbedding((self.k2, self.k3)),), (6,)),
                           ((self.k2, ProductEmbedding((self.k3_, self.k2_)), self.k3), (6,6)),
                           ((self.k2, ProductEmbedding((self.k2_, self.k3_)), self.k3), (4,9)),
                           ((ProductEmbedding((self.k2, self.k3_, self.k2_)), self.k3), (6,6)),
                           ((self.k2, ProductEmbedding((self.k2_, self.k3_, self.k3))), (4,9)),
                           ((self.k2, self.k0, self.k0_, self.k3), (0,)),
                           ((self.k2, self.k0, self.k0_, self.k3), (2,0)),
                           ((self.k2, self.k0, self.k0_, self.k3), (0,3)),
                           ((self.k2, self.k0, self.k0_, self.k3), (0,0))]:
            pembeds = tuple(frozenset(k for e in vembeds for k in e.stride({})[1]))
            physical = nrand(*(k.numel() for k in pembeds))
            t1 = EmbeddedTensor(physical, pembeds, vembeds, default=42)
            t2 = t1.reshape(s)
            self.assertTEqual(t1.to_dense().reshape(s), t2.to_dense())

    def test_stack(self):
        ts = [EmbeddedTensor(nrand(2,3,6)),
              EmbeddedTensor(nrand(2,6,2),
                             (self.k2_, self.k4, self.k2),
                             (self.k2, SumEmbedding(1,self.k2_,0), self.k4)),
              EmbeddedTensor(nrand(2,6,2),
                             (self.k2, self.k4, self.k2_),
                             (self.k2, SumEmbedding(1,self.k2_,0), self.k4)),
              EmbeddedTensor(nrand(6,2),
                             (self.k4, self.k2),
                             (self.k2, SumEmbedding(0,ProductEmbedding(()),2), self.k4)),
              EmbeddedTensor(nrand(3,2),
                             (self.k3, self.k2),
                             (self.k2, self.k3, ProductEmbedding((self.k2, self.k3))))]
        for t in ts:
            for d in range(-t.ndim-1, t.ndim+1):
                self.assertTEqual(t.unsqueeze(d).to_dense(),
                                  t.to_dense().unsqueeze(d))
            for d in range(t.ndim):
                td = t.dim_to_dense(dim=d)
                self.assertTEqual(td.to_dense(), t.to_dense())
                self.assertTrue(isinstance(td.vembeds[d], EmbeddingVar))
                self.assertTrue(td is td.dim_to_dense(dim=d))
                self.assertTClose(t.log_softmax(d).to_dense(),
                                  t.to_dense().log_softmax(d))
            for p in permutations(range(t.ndim)):
                tp = t.permute(p)
                actual = list(tp) # test __iter__
                expect = list(tp.to_dense())
                self.assertEqual(len(actual), len(expect))
                for actual_elem, expect_elem in zip(actual, expect):
                    self.assertTEqual(actual_elem.to_dense(), expect_elem)
        for n in range(len(ts)):
            for tensors in permutations(ts, n+1):
                for dim in range(4):
                    self.assertTEqual(stack(tensors, dim).to_dense(),
                                      torch.stack(list(t.to_dense() for t in tensors), dim))

    def test_solve(self):
        a = EmbeddedTensor(nrand(7), (self.k7,),
                           (ProductEmbedding((SumEmbedding(1,ProductEmbedding(()),0), self.k7)),
                            ProductEmbedding((SumEmbedding(0,ProductEmbedding(()),1), self.k7))))
        for b in [EmbeddedTensor(nrand(3), (self.k3,),
                                 (ProductEmbedding((SumEmbedding(0,ProductEmbedding(()),1),
                                                    SumEmbedding(0,self.k3,4))),)),
                  EmbeddedTensor(nrand(3), (self.k3,),
                                 (ProductEmbedding((SumEmbedding(0,ProductEmbedding(()),1),
                                                    SumEmbedding(0,self.k3,4))),
                                  self.k3)),
                  EmbeddedTensor(nrand(3,5), (self.k3,self.k5),
                                 (ProductEmbedding((SumEmbedding(0,ProductEmbedding(()),1),
                                                    SumEmbedding(0,self.k3,4))),
                                  self.k5)),
                  EmbeddedTensor(nrand(3,5), (self.k3,self.k5),
                                 (ProductEmbedding((SumEmbedding(0,ProductEmbedding(()),1),
                                                    SumEmbedding(0,self.k3,4))),
                                  self.k5,
                                  self.k3))]:
            semiring = RealSemiring(dtype=b.physical.dtype, device=b.physical.device)
            self.assertTEqual(at_most_matrix(a.solve(b, semiring).to_dense()),
                              semiring.solve(a.to_dense(), at_most_matrix(b.to_dense())))

def at_most_matrix(t: Tensor) -> Tensor:
    return t.flatten(start_dim=1) if t.ndim >= 2 else t

if __name__ == "__main__":
    unittest.main()
