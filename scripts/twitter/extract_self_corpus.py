import json
from pathlib import Path

try:
    from pipeline_common import count_tokens as pipeline_count_tokens
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline_common import count_tokens as pipeline_count_tokens


TWEETS_FILE = Path("data/twitter/normalized/tweets.json")
TWEET_GRAPH_FILE = Path("data/twitter/normalized/tweet_graph.json")
SELF_CORPUS_FILE = Path("data/twitter/curated/self_corpus.json")
SELF_THREADS_FILE = Path("data/twitter/curated/self_threads.json")
SELF_QUOTE_THREADS_FILE = Path("data/twitter/curated/self_quote_threads.json")
SELF_TWEETS_FILE = Path("data/twitter/curated/self_tweets.json")
SELF_CORPUS_REPORT_FILE = Path("data/twitter/curated/self_corpus_report.json")

SOURCE = "twitter"
MODE = "compressed_public_reasoning"


def count_tokens(text: str) -> int:
    return pipeline_count_tokens(text)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def combine_text(tweets: list[dict]) -> str:
    return "\n\n".join(tweet["content"] for tweet in tweets if tweet["content"])


def build_self_thread_paths(tweets: list[dict], graph: dict) -> list[list[str]]:
    tweet_by_id = {tweet["message_id"]: tweet for tweet in tweets}
    self_to_self = graph["indexes"]["self_to_self_replies"]
    all_self_children = {
        child_id
        for child_ids in self_to_self.values()
        for child_id in child_ids
    }
    root_ids = [
        tweet["message_id"]
        for tweet in tweets
        if tweet["speaker"] == "self"
        and tweet["message_id"] in self_to_self
        and tweet["message_id"] not in all_self_children
    ]

    paths = []

    def walk(message_id: str, path: list[str]) -> None:
        children = [
            child_id
            for child_id in self_to_self.get(message_id, [])
            if tweet_by_id.get(child_id, {}).get("speaker") == "self"
        ]
        if not children:
            if len(path) > 1:
                paths.append(path)
            return

        for child_id in children:
            if child_id in path:
                continue
            walk(child_id, path + [child_id])

    for root_id in root_ids:
        walk(root_id, [root_id])

    return sorted(
        paths,
        key=lambda path: (
            tweet_by_id[path[0]]["timestamp_ms"],
            len(path),
            path[0],
        ),
    )


def build_thread_units(tweets: list[dict], graph: dict) -> tuple[list[dict], set[str]]:
    tweet_by_id = {tweet["message_id"]: tweet for tweet in tweets}
    thread_paths = build_self_thread_paths(tweets, graph)
    used_tweet_ids = set()
    units = []

    for path in thread_paths:
        path_tweets = [tweet_by_id[message_id] for message_id in path]
        text = combine_text(path_tweets)
        used_tweet_ids.update(path)
        root = path_tweets[0]
        leaf = path_tweets[-1]
        units.append(
            {
                "corpus_id": f"twitter_self_thread_{len(units) + 1:06d}",
                "unit_type": "self_thread",
                "source": SOURCE,
                "mode": MODE,
                "conversation_id": root["conversation_id"],
                "root_tweet_id": root["tweet_id"],
                "root_message_id": root["message_id"],
                "leaf_tweet_id": leaf["tweet_id"],
                "leaf_message_id": leaf["message_id"],
                "tweet_ids": [tweet["tweet_id"] for tweet in path_tweets],
                "message_ids": path,
                "tweet_count": len(path_tweets),
                "start_timestamp_ms": root["timestamp_ms"],
                "end_timestamp_ms": leaf["timestamp_ms"],
                "token_count": count_tokens(text),
                "combined_text": text,
            }
        )

    return units, used_tweet_ids


def build_self_quote_units(tweets: list[dict], used_in_threads: set[str]) -> tuple[list[dict], set[str]]:
    tweet_by_id = {tweet["message_id"]: tweet for tweet in tweets}
    units = []
    used_tweet_ids = set()

    self_quotes = [
        tweet
        for tweet in tweets
        if tweet["speaker"] == "self"
        and tweet.get("quoted_message_id") in tweet_by_id
        and tweet_by_id[tweet["quoted_message_id"]]["speaker"] == "self"
    ]

    for quote_tweet in self_quotes:
        quoted_tweet = tweet_by_id[quote_tweet["quoted_message_id"]]
        message_ids = [quoted_tweet["message_id"], quote_tweet["message_id"]]

        direct_replies = [
            tweet
            for tweet in tweets
            if tweet["speaker"] == "self"
            and tweet.get("in_reply_to_message_id") == quote_tweet["message_id"]
        ]
        for reply in sorted(direct_replies, key=lambda tweet: (tweet["timestamp_ms"], tweet["message_id"])):
            if reply["message_id"] not in message_ids:
                message_ids.append(reply["message_id"])

        path_tweets = [tweet_by_id[message_id] for message_id in message_ids]
        text = combine_text(path_tweets)
        used_tweet_ids.update(message_ids)
        units.append(
            {
                "corpus_id": f"twitter_self_quote_thread_{len(units) + 1:06d}",
                "unit_type": "self_quote_thread",
                "source": SOURCE,
                "mode": MODE,
                "conversation_id": quote_tweet["conversation_id"],
                "quoted_tweet_id": quoted_tweet["tweet_id"],
                "quoted_message_id": quoted_tweet["message_id"],
                "quote_tweet_id": quote_tweet["tweet_id"],
                "quote_message_id": quote_tweet["message_id"],
                "tweet_ids": [tweet["tweet_id"] for tweet in path_tweets],
                "message_ids": message_ids,
                "tweet_count": len(path_tweets),
                "start_timestamp_ms": min(tweet["timestamp_ms"] for tweet in path_tweets),
                "end_timestamp_ms": max(tweet["timestamp_ms"] for tweet in path_tweets),
                "token_count": count_tokens(text),
                "combined_text": text,
                "already_in_self_reply_thread": any(message_id in used_in_threads for message_id in message_ids),
            }
        )

    return sorted(units, key=lambda unit: (unit["start_timestamp_ms"], unit["corpus_id"])), used_tweet_ids


def build_single_tweet_units(tweets: list[dict], used_in_threads: set[str]) -> list[dict]:
    self_tweets = [
        tweet
        for tweet in tweets
        if tweet["speaker"] == "self"
        and tweet["message_id"] not in used_in_threads
    ]
    units = []

    for tweet in self_tweets:
        text = tweet["content"]
        units.append(
            {
                "corpus_id": f"twitter_self_tweet_{len(units) + 1:06d}",
                "unit_type": "single_self_tweet",
                "source": SOURCE,
                "mode": MODE,
                "conversation_id": tweet["conversation_id"],
                "tweet_id": tweet["tweet_id"],
                "message_id": tweet["message_id"],
                "timestamp_ms": tweet["timestamp_ms"],
                "token_count": count_tokens(text),
                "content": text,
            }
        )

    return sorted(units, key=lambda unit: (unit["timestamp_ms"], unit["tweet_id"]))


def build_report(thread_units: list[dict], quote_units: list[dict], single_units: list[dict]) -> dict:
    thread_token_counts = [unit["token_count"] for unit in thread_units]
    quote_token_counts = [unit["token_count"] for unit in quote_units]
    single_token_counts = [unit["token_count"] for unit in single_units]
    long_threads = sorted(
        thread_units,
        key=lambda unit: (unit["tweet_count"], unit["token_count"]),
        reverse=True,
    )[:25]

    return {
        "source": SOURCE,
        "corpus_class": "separate_twitter_corpus",
        "total_units": len(thread_units) + len(quote_units) + len(single_units),
        "self_thread_units": len(thread_units),
        "self_quote_thread_units": len(quote_units),
        "single_self_tweet_units": len(single_units),
        "thread_tokens": sum(thread_token_counts),
        "self_quote_thread_tokens": sum(quote_token_counts),
        "single_tweet_tokens": sum(single_token_counts),
        "total_tokens": sum(thread_token_counts) + sum(quote_token_counts) + sum(single_token_counts),
        "max_thread_tweet_count": max((unit["tweet_count"] for unit in thread_units), default=0),
        "max_self_quote_thread_tweet_count": max((unit["tweet_count"] for unit in quote_units), default=0),
        "max_thread_tokens": max(thread_token_counts, default=0),
        "max_self_quote_thread_tokens": max(quote_token_counts, default=0),
        "max_single_tweet_tokens": max(single_token_counts, default=0),
        "top_threads": [
            {
                "corpus_id": unit["corpus_id"],
                "tweet_count": unit["tweet_count"],
                "token_count": unit["token_count"],
                "root_tweet_id": unit["root_tweet_id"],
                "preview": unit["combined_text"][:300],
            }
            for unit in long_threads
        ],
    }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    tweets = load_json(TWEETS_FILE)
    graph = load_json(TWEET_GRAPH_FILE)

    thread_units, used_tweet_ids = build_thread_units(tweets, graph)
    quote_units, quote_used_tweet_ids = build_self_quote_units(tweets, used_tweet_ids)
    single_units = build_single_tweet_units(tweets, used_tweet_ids | quote_used_tweet_ids)
    corpus = sorted(
        thread_units + quote_units + single_units,
        key=lambda unit: (
            unit.get("start_timestamp_ms", unit.get("timestamp_ms", 0)),
            unit["corpus_id"],
        ),
    )
    report = build_report(thread_units, quote_units, single_units)

    write_json(SELF_THREADS_FILE, thread_units)
    write_json(SELF_QUOTE_THREADS_FILE, quote_units)
    write_json(SELF_TWEETS_FILE, single_units)
    write_json(SELF_CORPUS_FILE, corpus)
    write_json(SELF_CORPUS_REPORT_FILE, report)

    print(f"Self thread units: {len(thread_units)}")
    print(f"Self quote thread units: {len(quote_units)}")
    print(f"Single self tweet units: {len(single_units)}")
    print(f"Total Twitter corpus units: {len(corpus)}")
    print(f"Total tokens: {report['total_tokens']}")
    print(f"Max thread tweet count: {report['max_thread_tweet_count']}")
    print(f"Max thread tokens: {report['max_thread_tokens']}")


if __name__ == "__main__":
    main()
