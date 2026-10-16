import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate import GenerationRuntime  # noqa: E402


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> str:
        self.calls.append((messages, kwargs))
        return "<formatted>" + messages[0]["content"]

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class GenerationRuntimeTests(unittest.TestCase):
    def runtime_with(self, processor: object) -> GenerationRuntime:
        runtime = object.__new__(GenerationRuntime)
        runtime.backend = "mlx-lm"
        runtime.processor = processor
        return runtime

    def test_mlx_lm_prompt_is_formatted_once_with_thinking_disabled(self) -> None:
        tokenizer = RecordingTokenizer()
        runtime = self.runtime_with(tokenizer)

        formatted = runtime.format_prompt("write from evidence")

        self.assertEqual(formatted, "<formatted>write from evidence")
        self.assertEqual(len(tokenizer.calls), 1)
        messages, kwargs = tokenizer.calls[0]
        self.assertEqual(messages, [{"role": "user", "content": "write from evidence"}])
        self.assertEqual(kwargs["tokenize"], False)
        self.assertEqual(kwargs["add_generation_prompt"], True)
        self.assertEqual(kwargs["enable_thinking"], False)

    def test_missing_mlx_lm_chat_template_fails_closed(self) -> None:
        runtime = self.runtime_with(object())

        with self.assertRaisesRegex(RuntimeError, "apply_chat_template"):
            runtime.format_prompt("unformatted prompt")

    def test_count_tokens_counts_the_formatted_prompt(self) -> None:
        tokenizer = RecordingTokenizer()
        runtime = self.runtime_with(tokenizer)

        self.assertEqual(runtime.count_tokens("one two three"), 3)
        self.assertEqual(tokenizer.calls[0][1]["add_generation_prompt"], True)


if __name__ == "__main__":
    unittest.main()
