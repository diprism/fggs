import unittest
import torch
import torch_semiring_einsum
from fggs.equation import *

class TestEquation(unittest.TestCase):

    def assertTEqual(self, input: torch.Tensor, other: torch.Tensor) -> None:
        self.assertTrue(torch.equal(input, other), (input, other))

    def test_reduce_equation(self):
        eq = 'ij,ji->ij'
        compiled_eq = torch_semiring_einsum.compile_equation(eq)

        si = 2
        sj = 3
        shapes_a = [[], [1], [sj], [1, 1], [si, 1], [1, sj], [si, sj]]
        shapes_b = [[], [1], [si], [1, 1], [sj, 1], [1, si], [sj, si]]

        ten_a = [
            torch.tensor(0.1),
            torch.tensor([0.1]),
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([[0.1]]),
            torch.tensor([[0.1], [0.2]]),
            torch.tensor([[0.1, 0.2, 0.3]]),
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        ]

        ten_b = [
            torch.tensor(1),
            torch.tensor([1]),
            torch.tensor([1, 10]),
            torch.tensor([[1]]),
            torch.tensor([[1], [10], [100]]),
            torch.tensor([[1, 10]]),
            torch.tensor([[1, 10], [2, 20], [3, 30]])
        ]

        expected_equations = [
            # []
            [
                ",->",
                ",->",
                ",a->a",
                ",->",
                ",a->a",
                ",a->a",
                ",ab->ba",
            ],
            # [1]
            [
                ",->",
                ",->",
                ",a->a",
                ",->",
                ",a->a",
                ",a->a",
                ",ab->ba",
            ],
            # [sj]
            [
                "a,->a",
                "a,->a",
                "a,b->ba",
                "a,->a",
                "a,a->a",
                "a,b->ba",
                "a,ab->ba",
            ],
            # [1,1]
            [
                ",->",
                ",->",
                ",a->a",
                ",->",
                ",a->a",
                ",a->a",
                ",ab->ba",
            ],
            # [si,1]
            [
                "a,->a",
                "a,->a",
                "a,a->a",
                "a,->a",
                "a,b->ab",
                "a,a->a",
                "a,ba->ab",  # None,
            ],
            # [1,sj]
            [
                "a,->a",
                "a,->a",
                "a,b->ba",
                "a,->a",
                "a,a->a",
                "a,b->ba",
                "a,ab->ba",
            ],
            # [si,sj]
            [
                "ab,->ab",
                "ab,->ab",
                "ab,a->ab",
                "ab,->ab",
                "ab,b->ab",
                "ab,a->ab",
                "ab,ba->ab",  # None,
            ]
        ]

        expected_output_shape = [2, 3]

        expected_unsqueeze = [
            # []
            [
                [0, 1],
                [0, 1],
                [1],
                [0, 1],
                [0],
                [1],
                [],
            ],
            # [1]
            [
                [0, 1],
                [0, 1],
                [1],
                [0, 1],
                [0],
                [1],
                [],
            ],
            # [sj]
            [
                [0],
                [0],
                [],
                [0],
                [0],
                [],
                [],
            ],
            # [1,1]
            [
                [0, 1],
                [0, 1],
                [1],
                [0, 1],
                [0],
                [1],
                [],
            ],
            # [si,1]
            [
                [1],
                [1],
                [1],
                [1],
                [],
                [1],
                [],
            ],
            # [1,sj]
            [
                [0],
                [0],
                [],
                [0],
                [0],
                [],
                [],
            ],
            # [si,sj]
            [
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ]
        ]

        for i, a in enumerate(ten_a):
            a = a.clone()
            for j, b in enumerate(ten_b):
                b = b.clone()
                ea = a.expand(si, sj)
                eb = b.expand(sj, si)
                eeq = expected_equations[i][j]
                eun = expected_unsqueeze[i][j]
                ref_ceq = torch_semiring_einsum.compile_equation(eeq)
                (shrunk_tensors, reduced_eq, unsqueeze_index,
                 output_shape) = reduce_equation(compiled_eq, (ea, eb))

                self.assertEqual(reduced_eq.input_variables, ref_ceq.input_variables, (a, b))
                self.assertEqual(reduced_eq.output_variables, ref_ceq.output_variables)
                self.assertEqual(output_shape, expected_output_shape)
                for st, dims in zip(shrunk_tensors, reduced_eq.input_variables):
                    self.assertEqual(st.dim(), len(dims))
                self.assertEqual(unsqueeze_index, eun)

                expected_out = torch_semiring_einsum.einsum(
                    compiled_eq,
                    ea,
                    eb,
                    block_size=torch_semiring_einsum.AUTOMATIC_BLOCK_SIZE)
                actual_out = torch_semiring_einsum.einsum(
                    reduced_eq,
                    *shrunk_tensors,
                    block_size=torch_semiring_einsum.AUTOMATIC_BLOCK_SIZE)
                actual_out = post_einsum(actual_out, unsqueeze_index, output_shape)
                self.assertTEqual(actual_out, expected_out)

if __name__ == "__main__":
    unittest.main()
