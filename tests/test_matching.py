import unittest

from matching.title_similarity import title_similarity
from matching.author_similarity import author_similarity


class TestMatching(unittest.TestCase):

    def test_title_exact(self):

        score = title_similarity(
            "Attention Is All You Need",
            "Attention Is All You Need",
        )

        self.assertGreaterEqual(
            score,
            0.99,
        )

    def test_authors_short_names(self):

        score = author_similarity(
            [
                "Vaswani, A.",
                "Shazeer, N.",
                "Parmar, N.",
            ],
            [
                "Ashish Vaswani",
                "Noam Shazeer",
                "Niki Parmar",
            ],
        )

        self.assertGreater(
            score,
            0.80,
        )

    def test_authors_resnet(self):

        score = author_similarity(
            [
                "He, K.",
                "Zhang, X.",
                "Ren, S.",
                "Sun, J.",
            ],
            [
                "Kaiming He",
                "Xiangyu Zhang",
                "Shaoqing Ren",
                "Jian Sun",
            ],
        )

        self.assertGreater(
            score,
            0.80,
        )


if __name__ == "__main__":
    unittest.main()