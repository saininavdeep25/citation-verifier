import unittest

from parser.citation_parser import parse_citation


class TestParser(unittest.TestCase):

    def test_attention_is_all_you_need(self):

        c = parse_citation(
            "Vaswani, A., Shazeer, N., Parmar, N., "
            "Uszkoreit, J., Jones, L., Gomez, A. N., "
            "Kaiser, L., & Polosukhin, I. (2017). "
            "Attention Is All You Need. "
            "Advances in Neural Information Processing Systems, 30."
        )

        self.assertEqual(
            c.year,
            2017,
        )

        self.assertEqual(
            c.title,
            "Attention Is All You Need",
        )

        self.assertEqual(
            c.venue,
            "Advances in Neural Information Processing Systems",
        )

        self.assertEqual(
            len(c.authors),
            8,
        )

        self.assertEqual(
            c.authors[0],
            "Vaswani, A.",
        )

    def test_resnet(self):

        c = parse_citation(
            "He, K., Zhang, X., Ren, S., & Sun, J. "
            "(2016). Deep Residual Learning for Image "
            "Recognition. Proceedings of the IEEE Conference "
            "on Computer Vision and Pattern Recognition, "
            "770-778. "
            "https://doi.org/10.1109/CVPR.2016.90"
        )

        self.assertEqual(
            c.year,
            2016,
        )

        self.assertEqual(
            c.title,
            "Deep Residual Learning for Image Recognition",
        )

        self.assertEqual(
            c.doi,
            "10.1109/cvpr.2016.90",
        )

        self.assertEqual(
            len(c.authors),
            4,
        )

    def test_bert(self):

        c = parse_citation(
            "Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. "
            "(2019). BERT: Pre-training of Deep Bidirectional "
            "Transformers for Language Understanding. "
            "Proceedings of the 2019 Conference of the North "
            "American Chapter of the Association for Computational "
            "Linguistics: Human Language Technologies, 4171-4186. "
            "https://doi.org/10.18653/v1/N19-1423"
        )

        self.assertEqual(
            c.year,
            2019,
        )

        self.assertEqual(
            c.title,
            "BERT: Pre-training of Deep Bidirectional Transformers "
            "for Language Understanding",
        )

        self.assertEqual(
            c.doi,
            "10.18653/v1/n19-1423",
        )

        self.assertEqual(
            len(c.authors),
            4,
        )

    def test_apa_doi(self):

        c = parse_citation(
            "LeCun, Y., Bengio, Y., Hinton, G. (2015). "
            "Deep learning. Nature, 521, 436-444. "
            "doi:10.1038/nature14539"
        )

        self.assertEqual(
            c.doi,
            "10.1038/nature14539",
        )

    def test_no_false_issue_from_de_novo_title(self):

        c = parse_citation(
            "Sohn, J., & Nam, J. (2016). The present and future of de novo "
            "whole-genome assembly. Briefings in Bioinformatics."
        )

        self.assertIsNone(c.issue)


if __name__ == "__main__":
    unittest.main()
