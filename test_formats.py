import unittest
import formats
import json

class TestJson(unittest.TestCase):
    def test_roundtrip(self):
        j = json.loads(hmm_json)
        g = formats.json_to_fgg(j)
        j_check = formats.fgg_to_json(g)
        self.maxDiff = 10000
        self.assertEqual(j, j_check)
                
hmm_json = """
{
    "domains": {
        "T": {
            "class": "finite",
            "values": ["DT", "NN", "VBD", "IN", "BOS", "EOS"]
        },
        "W": {
            "class": "finite",
            "values": ["the", "cat", "sat", "on", "mat"]
        }
    },
    "factors": {
        "transition": {
            "function": "categorical",
            "type": ["T", "T"],
            "weights": [[0, 1, 0, 0, 0, 0],
                        [0, 0.25, 0.25, 0.25, 0, 0.25],
                        [0.3, 0, 0, 0.3, 0, 0.4],
                        [1, 0, 0, 0, 0, 0],
                        [0.8, 0, 0, 0.2, 0, 0],
                        [0, 0, 0, 0, 0, 0]]
        },
        "emission": {
            "function": "categorical",
            "type": ["T", "W"],
            "weights": [[1, 0, 0, 0, 0],
                        [0, 0.5, 0, 0, 0.5],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]
        },
        "is_bos": {
            "function": "categorical",
            "type": ["T"],
            "weights": [0, 0, 0, 0, 1, 0]
        },
        "is_eos": {
            "function": "categorical",
            "type": ["T"],
            "weights": [0, 0, 0, 0, 0, 1]
        }
    },
    "nonterminals": {
        "S": { "type" : [] },
        "X": { "type" : ["T"] }
    },
    "start": "S",
    "rules": [
        {
            "lhs": "S",
            "rhs": {
                "nodes": ["T"],
                "edges": [
                    {
                        "attachments": [0],
                        "label": "is_bos"
                    },
                    {
                        "attachments": [0],
                        "label": "X"
                    }
                ],
                "externals": []
            }
        },
        {
            "lhs": "X",
            "rhs": {
                "nodes": ["T", "T", "W"],
                "edges": [
                    {
                        "attachments": [0, 1],
                        "label": "transition"
                    },
                    {
                        "attachments": [1, 2],
                        "label": "emission"
                    },
                    {
                        "attachments": [1],
                        "label": "X"
                    }
                ],
                "externals": [0]
            }
        },
        {
            "lhs": "X",
            "rhs": {
                "nodes": ["T", "T"],
                "edges": [
                    {
                        "attachments": [0, 1],
                        "label": "transition"
                    },
                    {
                        "attachments": [1],
                        "label": "is_eos"
                    }
                ],
                "externals": [0]
            }
        }
    ]
}
"""

if __name__ == "__main__":
    unittest.main()
    
