from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts.twitter import process_exports


class TwitterIdentityConfigurationTests(unittest.TestCase):
    def test_configured_identity_is_case_insensitive_and_strips_at_prefix(self) -> None:
        with patch.dict(os.environ, {"WRITING_RAG_TWITTER_SELF_SCREEN_NAME": "@DemoAuthor"}):
            tweet = process_exports.normalize_tweet({"id": "1", "screen_name": "demoauthor"})
        self.assertIsNotNone(tweet)
        self.assertEqual(tweet["speaker"], "self")

    def test_missing_identity_stops_twitter_import_before_reading_private_exports(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WRITING_RAG_TWITTER_SELF_SCREEN_NAME", None)
            with self.assertRaisesRegex(SystemExit, "WRITING_RAG_TWITTER_SELF_SCREEN_NAME is required"):
                process_exports.main()


if __name__ == "__main__":
    unittest.main()
