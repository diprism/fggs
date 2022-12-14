import unittest

from fggs.indices import *

class TestEmbeddedTensor(unittest.TestCase):

    def assertTEqual(self, input: Tensor, other: Tensor) -> bool:
        self.assertTrue(torch.equal(input, other))

    def assertTClose(self, input: Tensor, other: Tensor) -> bool:
        self.assertTrue(torch.allclose(input, other))

    def setUp(self):
        self.k1  = EmbeddingVar(5)
        self.k1_ = EmbeddingVar(5)
        self.k2  = EmbeddingVar(2)
        self.k2_ = EmbeddingVar(2)
        self.k3  = EmbeddingVar(3)
        self.k3_ = EmbeddingVar(3)
        self.k4  = EmbeddingVar(6)
        self.k5  = EmbeddingVar(5)
        self.k5_ = EmbeddingVar(5)
        self.k6  = EmbeddingVar(35)
        self.k7  = EmbeddingVar(7)
        self.k7_ = EmbeddingVar(7)
        self.k8  = EmbeddingVar(36)

    def test_diag(self):
        phys = torch.randn(5,2)
        virt = EmbeddedTensor(phys,
                              (self.k1,self.k2),
                              (self.k2,self.k1,self.k1))
        self.assertTEqual(virt.to_dense({}),
                          phys.t().diag_embed(dim1=1, dim2=2))

    def test_algebraic(self):
        ones = torch.ones(2,3)
        diag = EmbeddedTensor(ones,
                              (self.k2,self.k3),
                              (self.k2,self.k3,SumEmbedding(1,ProductEmbedding((self.k2,self.k3)),2)))
        self.assertTEqual(diag.to_dense({}),
                          torch.tensor([[[0,1,0,0,0,0,0,0,0],
                                         [0,0,1,0,0,0,0,0,0],
                                         [0,0,0,1,0,0,0,0,0]],
                                        [[0,0,0,0,1,0,0,0,0],
                                         [0,0,0,0,0,1,0,0,0],
                                         [0,0,0,0,0,0,1,0,0.0]]]))

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

    def test_einsum(self):
        matrix = torch.randn(36)
        vector = torch.randn(7)
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
                                                 (self.k5_,self.k7_),
                                                 (ProductEmbedding((self.k5_,self.k7_)),self.k5_,self.k7_))],
                                 [["maybe-f"], ["i"], ["maybe-f","f"], ["f","o","i"]],
                                 ["o"],
                                 semiring).to_dense({}))

    def test_add_dense(self):
        phys = torch.randn(5,2)
        virt = EmbeddedTensor(phys,
                              (self.k1,self.k2),
                              (self.k2,self.k1,self.k1))
        phys2 = torch.randn(5,5)
        virt2 = EmbeddedTensor(phys2,
                               (self.k1,self.k1_),
                               (SumEmbedding(0,ProductEmbedding(()),1),self.k1_,self.k1))
        self.assertTEqual(add(virt, virt2).to_dense({}),
                          torch.add(virt.to_dense({}), virt2.to_dense({})))
        self.assertTEqual(add(virt2, virt).to_dense({}),
                          torch.add(virt2.to_dense({}), virt.to_dense({})))

    def test_add_sparse(self):
        phys = torch.randn(5,2)
        virt = EmbeddedTensor(phys,
                              (self.k1,self.k2),
                              (self.k2,self.k1,self.k1))
        phys3 = torch.arange(0.0,500,10).reshape(5,2,5)
        virt3 = EmbeddedTensor(phys3,
                               (self.k1,self.k2,self.k1_),
                               (self.k2,self.k1_,self.k1))
        self.assertTEqual(add(virt, virt3).to_dense({}),
                          torch.add(virt.to_dense({}), virt3.to_dense({})))
        self.assertTEqual(add(virt3, virt).to_dense({}),
                          torch.add(virt3.to_dense({}), virt.to_dense({})))

if __name__ == "__main__":
    unittest.main()
