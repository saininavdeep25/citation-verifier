import unittest

from matching.author_similarity import author_similarity
from matching.title_similarity import (
    title_similarity,
    title_match_details,
)
from matching.venue_similarity import venue_similarity
from utils.normalization import (
    normalize_doi,
    normalize_title,
    normalize_author_name,
)


class TestAuthorSimilarity(unittest.TestCase):

    def test_authors_resnet(self):
        citation_authors = [
            "He, K.",
            "Zhang, X.",
            "Ren, S.",
            "Sun, J.",
        ]

        database_authors = [
            "Kaiming He",
            "Xiangyu Zhang",
            "Shaoqing Ren",
            "Jian Sun",
        ]

        score = author_similarity(
            citation_authors,
            database_authors,
        )

        self.assertGreaterEqual(
            score,
            0.90,
        )

    def test_authors_short_names(self):
        citation_authors = [
            "Vaswani, A.",
            "Shazeer, N.",
            "Parmar, N.",
        ]

        database_authors = [
            "Ashish Vaswani",
            "Noam Shazeer",
            "Niki Parmar",
        ]

        score = author_similarity(
            citation_authors,
            database_authors,
        )

        self.assertGreaterEqual(
            score,
            0.90,
        )


class TestTitleSimilarity(unittest.TestCase):

    def test_title_exact(self):
        title = "Deep Residual Learning for Image Recognition"

        self.assertEqual(
            title_similarity(
                title,
                title,
            ),
            1.0,
        )

    def test_title_case_difference(self):
        supplied = (
            "Deep Residual Learning for Image Recognition"
        )

        database = (
            "DEEP RESIDUAL LEARNING FOR IMAGE RECOGNITION"
        )

        self.assertEqual(
            title_similarity(
                supplied,
                database,
            ),
            1.0,
        )

    def test_html_italics(self):
        supplied = (
            "The present and future of <i>de novo</i> "
            "whole-genome assembly"
        )

        database = (
            "The present and future of de novo whole-genome assembly"
        )

        details = title_match_details(
            supplied,
            database,
        )

        self.assertEqual(
            details["normalized_similarity"],
            1.0,
        )

        self.assertEqual(
            details["match_type"],
            "FORMATTING_NORMALIZED",
        )

    def test_html_emphasis(self):
        supplied = (
            "The <em>present</em> and future of de novo "
            "whole-genome assembly"
        )

        database = (
            "The present and future of de novo whole-genome assembly"
        )

        self.assertEqual(
            title_similarity(
                supplied,
                database,
            ),
            1.0,
        )

    def test_html_bold(self):
        supplied = (
            "The <b>present</b> and future of de novo "
            "whole-genome assembly"
        )

        database = (
            "The present and future of de novo whole-genome assembly"
        )

        self.assertEqual(
            title_similarity(
                supplied,
                database,
            ),
            1.0,
        )

    def test_html_entity(self):
        supplied = "Learning &amp; Representation"
        database = "Learning & Representation"

        self.assertEqual(
            title_similarity(
                supplied,
                database,
            ),
            1.0,
        )

    def test_ampersand_and(self):
        supplied = "Learning & Representation"
        database = "Learning and Representation"

        self.assertEqual(
            title_similarity(
                supplied,
                database,
            ),
            1.0,
        )

    def test_unicode_dash(self):
        supplied = (
            "The present and future of de novo "
            "whole–genome assembly"
        )

        database = (
            "The present and future of de novo "
            "whole-genome assembly"
        )

        self.assertEqual(
            title_similarity(
                supplied,
                database,
            ),
            1.0,
        )

    def test_extra_whitespace(self):
        supplied = (
            "Deep   Residual   Learning for "
            "Image Recognition"
        )

        database = (
            "Deep Residual Learning for Image Recognition"
        )

        self.assertEqual(
            title_similarity(
                supplied,
                database,
            ),
            1.0,
        )

    def test_latex_italics(self):
        supplied = (
            r"The present and future of \textit{de novo} "
            "whole-genome assembly"
        )

        database = (
            "The present and future of de novo whole-genome assembly"
        )

        self.assertEqual(
            title_similarity(
                supplied,
                database,
            ),
            1.0,
        )

    def test_real_title_difference(self):
        supplied = (
            "The present and future of de novo "
            "whole-genome assembly"
        )

        database = (
            "The present and future of genome assembly"
        )

        details = title_match_details(
            supplied,
            database,
        )

        self.assertEqual(
            details["match_type"],
            "SUBSTANTIVE_DIFFERENCE",
        )

        self.assertLess(
            details["normalized_similarity"],
            0.90,
        )

    def test_missing_word(self):
        supplied = (
            "Deep Residual Learning for Image Recognition"
        )

        database = (
            "Deep Learning for Image Recognition"
        )

        details = title_match_details(
            supplied,
            database,
        )

        self.assertEqual(
            details["match_type"],
            "SUBSTANTIVE_DIFFERENCE",
        )

    def test_completely_different_title(self):
        supplied = "Deep Residual Learning for Image Recognition"

        database = (
            "Attention Is All You Need"
        )

        details = title_match_details(
            supplied,
            database,
        )

        self.assertEqual(
            details["match_type"],
            "SUBSTANTIVE_DIFFERENCE",
        )


class TestDOINormalization(unittest.TestCase):

    def test_doi(self):
        doi1 = "10.1109/CVPR.2016.90"
        doi2 = "https://doi.org/10.1109/CVPR.2016.90"

        self.assertEqual(
            normalize_doi(doi1),
            normalize_doi(doi2),
        )

    def test_doi_case(self):
        doi1 = "10.1109/CVPR.2016.90"
        doi2 = "10.1109/cvpr.2016.90"

        self.assertEqual(
            normalize_doi(doi1),
            normalize_doi(doi2),
        )


class TestVenueSimilarity(unittest.TestCase):

    def test_venue_alias(self):
        supplied = (
            "Advances in Neural Information Processing Systems"
        )

        database = "NeurIPS"

        score = venue_similarity(
            supplied,
            database,
        )

        self.assertGreaterEqual(
            score,
            0.90,
        )


class TestNormalization(unittest.TestCase):

    def test_title_normalization(self):
        supplied = (
            "The Present and Future of <i>DE NOVO</i> "
            "Whole-Genome Assembly"
        )

        database = (
            "the present and future of de novo "
            "whole-genome assembly"
        )

        self.assertEqual(
            normalize_title(supplied),
            normalize_title(database),
        )

    def test_author_normalization(self):
        self.assertEqual(
            normalize_author_name("  Kaiming He. "),
            "kaiming he",
        )


if __name__ == "__main__":
    unittest.main()