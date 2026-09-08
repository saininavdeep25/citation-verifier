import unittest
from unittest.mock import Mock

import requests

from sources.crossref import CrossrefSource
from sources.openalex import OpenAlexSource


class TestSourceRetries(unittest.TestCase):
    def test_not_found_is_not_retried(self):
        for source_class in (OpenAlexSource, CrossrefSource):
            with self.subTest(source=source_class.name):
                source = source_class()
                response = requests.Response()
                response.status_code = 404
                response.url = "https://example.test/works/missing"
                source.session.get = Mock(return_value=response)

                self.assertIsNone(source._get("/works/missing"))
                self.assertEqual(source.session.get.call_count, 1)
                self.assertEqual(source.last_status_code, 404)


if __name__ == "__main__":
    unittest.main()
