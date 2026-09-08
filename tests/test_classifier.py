import unittest

import verification.classifier as classifier
from models.schemas import CandidatePaper, CitationMetadata


class TestClassifierThresholds(unittest.TestCase):

    def test_uncertain_threshold_is_configurable(self):
        citation = CitationMetadata(title="Input title")
        candidate = CandidatePaper(
            source="test",
            source_id="test-1",
            title="Different title",
            authors=[],
            year=None,
            venue=None,
            doi=None,
            volume=None,
            issue=None,
            pages=None,
            url=None,
            retrieval_method="test",
            candidate_score=0.65,
        )
        original_threshold = classifier.UNCERTAIN_THRESHOLD

        try:
            classifier.UNCERTAIN_THRESHOLD = 0.70
            result = classifier.classify(citation, candidate)
        finally:
            classifier.UNCERTAIN_THRESHOLD = original_threshold

        self.assertEqual(result["status"], "LIKELY_HALLUCINATED")


if __name__ == "__main__":
    unittest.main()
