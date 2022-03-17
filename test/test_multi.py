from fggs.multi import *
from fggs.semirings import *
import unittest, torch, random

class TestMultiTensor(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        random.seed(0)
        
        self.matrices = []
        self.vectors = []
        
        m = 3
        self.shapes = shapes = {i:torch.Size((random.randrange(1, 4),)) for i in range(m)}
        n = sum(shape.numel() for shape in shapes.values())
        c = 0
        self.offsets = offsets = {}
        for i in range(m):
            offsets[i] = c
            c += shapes[i].numel()
        semiring = RealSemiring()

        for l in range(10):
            a_sparse = MultiTensor((shapes, shapes), semiring)
            a_dense = torch.zeros(n, n)
            for k in range(m**2//2):
                i = random.randrange(0, m)
                j = random.randrange(0, m)
                a_sparse[i,j] = torch.rand(shapes[i]+shapes[j])
                a_dense[offsets[i]:offsets[i]+shapes[i].numel(),
                        offsets[j]:offsets[j]+shapes[j].numel()] = a_sparse[i,j]

            a_norm = torch.linalg.norm(a_dense)
            for a in a_sparse.values():
                a /= 2*a_norm
            a_dense /= 2*a_norm
            self.matrices.append((a_sparse, a_dense))

        for l in range(10):
            b_sparse = MultiTensor((shapes,), semiring)
            b_dense = torch.zeros(n)
            for k in range(m//2):
                i = random.randrange(0, m)
                b_sparse[i] = torch.rand(shapes[i])
                b_dense[offsets[i]:offsets[i]+shapes[i].numel()] = b_sparse[i]
            self.vectors.append((b_sparse, b_dense))

    def compare_vectors(self, sparse, dense):
        shapes, offsets = self.shapes, self.offsets
        for i in shapes:
            if i in sparse:
                if not torch.allclose(sparse[i], dense[offsets[i]:offsets[i]+shapes[i].numel()]):
                    return False
            else:
                if not torch.allclose(torch.zeros(shapes[i]), dense[offsets[i]:offsets[i]+shapes[i].numel()]):
                    return False
        return True

    def test_multi_mv(self):
        for a_sparse, a_dense in self.matrices:
            for b_sparse, b_dense in self.vectors:
                c_sparse = multi_mv(a_sparse, b_sparse)
                c_dense = torch.mv(a_dense, b_dense)
                self.assertTrue(self.compare_vectors(c_sparse, c_dense), (c_sparse, c_dense))
                
                c_sparse = multi_mv(a_sparse, b_sparse, transpose=True)
                c_dense = torch.mv(a_dense.T, b_dense)
                self.assertTrue(self.compare_vectors(c_sparse, c_dense), (c_sparse, c_dense))

    def test_multi_solve(self):
        for a_sparse, a_dense in self.matrices:
            for b_sparse, b_dense in self.vectors:
                x_sparse = multi_solve(a_sparse, b_sparse)
                x_dense = torch.linalg.solve(torch.eye(*a_dense.shape)-a_dense, b_dense)
                self.assertTrue(self.compare_vectors(x_sparse, x_dense), (x_sparse, x_dense))

                x_sparse = multi_solve(a_sparse, b_sparse, transpose=True)
                x_dense = torch.linalg.solve(torch.eye(*a_dense.shape)-a_dense.T, b_dense)
                self.assertTrue(self.compare_vectors(x_sparse, x_dense), (x_sparse, x_dense))

            
if __name__ == '__main__':
    unittest.main()
