from fggs.semirings import *
import unittest
import itertools, math
import torch, torch_semiring_einsum

examples = [
    (RealSemiring(),    list(map(torch.tensor, [0., 1., 2.]))),
    (LogSemiring(),     list(map(torch.tensor, [-math.inf, 0., math.log(2.)]))),
    (ViterbiSemiring(), list(map(torch.tensor, [-math.inf, 0., 0.]))),
    (BoolSemiring(),    list(map(torch.tensor, [False, True, True]))),
]

class TestSemirings(unittest.TestCase):
    def assertAlmostEqual(self, x, y):
        if x.dtype is torch.bool:
            self.assertTrue(torch.all(x == y), (x, y))
        else:
            self.assertTrue(torch.allclose(x.nan_to_num(), y.nan_to_num()), (x, y))
    
    def test_from_int(self):
        for semiring, values in examples:
            with self.subTest(semiring=semiring.__class__.__name__):
                for i, x in enumerate(values):
                    self.assertAlmostEqual(semiring.from_int(i), x)
    
    def test_add_associative(self):
        for semiring, values in examples:
            with self.subTest(semiring=semiring.__class__.__name__):
                for x, y, z in itertools.product(values, repeat=3):
                    self.assertAlmostEqual(semiring.add(x, semiring.add(y, z)),
                                           semiring.add(semiring.add(x, y), z))

    def test_add_identity(self):
        for semiring, values in examples:
            with self.subTest(semiring=semiring.__class__.__name__):
                for x in values:
                    self.assertAlmostEqual(semiring.add(x, values[0]), x)
                    self.assertAlmostEqual(semiring.add(values[0], x), x)

    def test_add_commutative(self):
        for semiring, values in examples:
            with self.subTest(semiring=semiring.__class__.__name__):
                for x, y in itertools.product(values, repeat=2):
                    self.assertAlmostEqual(semiring.add(x, y),
                                           semiring.add(y, x))
                
    def test_mul_associative(self):
        for semiring, values in examples:
            with self.subTest(semiring=semiring.__class__.__name__):
                for x, y, z in itertools.product(values, repeat=3):
                    self.assertAlmostEqual(semiring.mul(x, semiring.mul(y, z)),
                                           semiring.mul(semiring.mul(x, y), z))

    def test_mul_zero(self):
        for semiring, values in examples:
            with self.subTest(semiring=semiring.__class__.__name__):
                for x in values:
                    self.assertAlmostEqual(semiring.mul(x, values[0]), values[0])
                    self.assertAlmostEqual(semiring.mul(values[0], x), values[0])

    def test_mul_identity(self):
        for semiring, values in examples:
            with self.subTest(semiring=semiring.__class__.__name__):
                for x in values:
                    self.assertAlmostEqual(semiring.mul(x, values[1]), x)
                    self.assertAlmostEqual(semiring.mul(values[1], x), x)

    def test_mul_commutative(self):
        for semiring, values in examples:
            with self.subTest(semiring=semiring.__class__.__name__):
                for x, y in itertools.product(values, repeat=2):
                    self.assertAlmostEqual(semiring.mul(x, y),
                                           semiring.mul(y, x))
                
    def test_distributive(self):
        for semiring, values in examples:
            with self.subTest(semiring=semiring.__class__.__name__):
                for x, y, z in itertools.product(values, repeat=3):
                    self.assertAlmostEqual(semiring.mul(x, semiring.add(y, z)),
                                           semiring.add(semiring.mul(x, y), semiring.mul(x, z)))
                    self.assertAlmostEqual(semiring.mul(semiring.add(x, y), z),
                                           semiring.add(semiring.mul(x, z), semiring.mul(y, z)))

    def test_star(self):
        for semiring, values in examples:
            with self.subTest(semiring=semiring.__class__.__name__):
                for x in values:
                    self.assertAlmostEqual(semiring.star(x),
                                           semiring.add(values[1], semiring.mul(x, semiring.star(x))))
                    self.assertAlmostEqual(semiring.star(x),
                                           semiring.add(values[1], semiring.mul(semiring.star(x), x)))
                    
    def test_sub(self):
        for semiring, values in examples:
            with self.subTest(semiring=semiring.__class__.__name__):
                for x, y in itertools.product(values, repeat=2):
                    if x <= y:
                        self.assertAlmostEqual(semiring.add(semiring.sub(y, x), x), y)
                    else:
                        self.assertAlmostEqual(semiring.add(semiring.sub(y, x), x), x)

    def test_matmul(self):
        e = torch_semiring_einsum.compile_equation('ij,jk->ik')
        torch.manual_seed(0)
        a = torch.randint(0, 10, (10, 10))
        b = torch.randint(0, 10, (10, 10))
        c = a @ b
        for semiring, _ in examples:
            with self.subTest(semiring=semiring.__class__.__name__):
                self.assertAlmostEqual(
                    semiring.einsum(e, semiring.from_int(a), semiring.from_int(b)),
                    semiring.from_int(c)
                )

    def test_solve(self):
        torch.manual_seed(0)
        a = torch.rand(10, 10)
        b = torch.rand(10)
        a /= 2*torch.norm(a) # ensure that x exists
        x = torch.linalg.inv(torch.eye(*a.shape)-a) @ b
        
        with self.subTest(semiring='RealSemiring'):
            self.assertAlmostEqual(RealSemiring().solve(a, b), x)
        with self.subTest(semiring='LogSemiring'):
            self.assertAlmostEqual(LogSemiring().solve(a.log(), b.log()), x.log())
        # No tests for ViterbiSemiring or BoolSemiring yet
                            
if __name__ == '__main__':
    unittest.main()
