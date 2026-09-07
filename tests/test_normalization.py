import unittest

from utils.normalization import normalize_doi, normalize_title, normalize_venue


class TestNormalization(unittest.TestCase):
    def test_title(self):
        self.assertEqual(
            normalize_title("Attention Is All You Need!"),
            "attention is all you need",
        )

    def test_doi(self):
        self.assertEqual(
            normalize_doi("https://doi.org/10.1038/nature14539."),
            "10.1038/nature14539",
        )

    def test_venue_alias(self):
        self.assertEqual(
            normalize_venue("NeurIPS"),
            normalize_venue("Advances in Neural Information Processing Systems"),
        )


if __name__ == "__main__":
    unittest.main()
