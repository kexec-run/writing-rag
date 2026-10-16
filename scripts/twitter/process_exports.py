import json
import os
from collections import Counter, defaultdict, deque
from datetime import datetime
from itertools import pairwise
from pathlib import Path

RAW_DIR = Path("data/raw/twitter")
LIGHT_FILE = RAW_DIR / "twitter_no_metadata.json"
METADATA_FILE = RAW_DIR / "twitter_metadata.json"

TWEETS_FILE = Path("data/twitter/normalized/tweets.json")
TWEET_GRAPH_FILE = Path("data/twitter/normalized/tweet_graph.json")
CONVERSATION_PATHS_FILE = Path("data/twitter/normalized/conversation_paths.json")
THREAD_STATS_FILE = Path("data/twitter/reports/thread_stats.json")

SOURCE = "twitter"


def configured_self_screen_name() -> str | None:
    value = os.environ.get("WRITING_RAG_TWITTER_SELF_SCREEN_NAME", "").strip()
    return value.lstrip("@").casefold() or None


def load_json_array(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON array in {path}")
    return data


def timestamp_ms(created_at: str | None) -> int:
    if not created_at:
        return 0

    for fmt in ("%Y-%m-%d %H:%M:%S %z", "%a %b %d %H:%M:%S %z %Y"):
        try:
            return int(datetime.strptime(created_at, fmt).timestamp() * 1000)  # noqa: DTZ007 — both accepted formats include %z
        except ValueError:
            continue

    return 0


def message_id(tweet_id: str | int | None) -> str | None:
    if tweet_id is None:
        return None
    return f"twitter_{tweet_id}"


def legacy_from_metadata(row: dict | None) -> dict:
    if not row:
        return {}
    metadata = row.get("metadata") or {}
    legacy = metadata.get("legacy") or {}
    return legacy if isinstance(legacy, dict) else {}


def nested_quoted_legacy(row: dict | None) -> dict | None:
    if not row:
        return None
    metadata = row.get("metadata") or {}
    quoted = metadata.get("quoted_status_result") or {}
    result = quoted.get("result") or {}
    legacy = result.get("legacy")
    return legacy if isinstance(legacy, dict) else None


def normalize_tweet(row: dict, metadata_row: dict | None = None, node_origin: str = "top_level") -> dict | None:
    tweet_id = str(row.get("id") or row.get("id_str") or row.get("rest_id") or "")
    if not tweet_id:
        return None

    legacy = legacy_from_metadata(metadata_row)
    screen_name = row.get("screen_name") or legacy.get("screen_name")
    self_screen_name = configured_self_screen_name()
    speaker = "self" if self_screen_name and screen_name and screen_name.casefold() == self_screen_name else "other"

    conversation_id = (
        legacy.get("conversation_id_str")
        or row.get("conversation_id")
        or row.get("conversation_id_str")
        or tweet_id
    )
    in_reply_to_tweet_id = (
        legacy.get("in_reply_to_status_id_str")
        or row.get("in_reply_to")
        or row.get("in_reply_to_status_id_str")
    )
    in_reply_to_screen_name = legacy.get("in_reply_to_screen_name") or row.get("in_reply_to_screen_name")
    quoted_tweet_id = (
        legacy.get("quoted_status_id_str")
        or row.get("quoted_status_id")
        or ((row.get("quoted_status") or {}).get("id") if isinstance(row.get("quoted_status"), dict) else None)
    )
    retweeted_tweet_id = (
        (row.get("retweeted_status") or {}).get("id")
        if isinstance(row.get("retweeted_status"), dict)
        else row.get("retweeted_status")
    )

    content = row.get("full_text") or legacy.get("full_text") or ""
    created_at = row.get("created_at") or legacy.get("created_at")
    media = row.get("media") or []

    return {
        "tweet_id": tweet_id,
        "message_id": message_id(tweet_id),
        "source": SOURCE,
        "speaker": speaker,
        "screen_name": screen_name,
        "sender": screen_name,
        "sender_name": row.get("name"),
        "timestamp_ms": timestamp_ms(created_at),
        "created_at": created_at,
        "content": content,
        "conversation_id": str(conversation_id) if conversation_id is not None else None,
        "in_reply_to_tweet_id": str(in_reply_to_tweet_id) if in_reply_to_tweet_id else None,
        "in_reply_to_message_id": message_id(in_reply_to_tweet_id),
        "in_reply_to_screen_name": in_reply_to_screen_name,
        "quoted_tweet_id": str(quoted_tweet_id) if quoted_tweet_id else None,
        "quoted_message_id": message_id(quoted_tweet_id),
        "retweeted_tweet_id": str(retweeted_tweet_id) if retweeted_tweet_id else None,
        "retweeted_message_id": message_id(retweeted_tweet_id),
        "is_reply": bool(in_reply_to_tweet_id),
        "is_quote": bool(quoted_tweet_id),
        "is_retweet": bool(retweeted_tweet_id),
        "url": row.get("url"),
        "metrics": {
            "reply_count": row.get("reply_count") or legacy.get("reply_count") or 0,
            "retweet_count": row.get("retweet_count") or legacy.get("retweet_count") or 0,
            "quote_count": row.get("quote_count") or legacy.get("quote_count") or 0,
            "favorite_count": row.get("favorite_count") or legacy.get("favorite_count") or 0,
            "bookmark_count": row.get("bookmark_count") or legacy.get("bookmark_count") or 0,
            "views_count": row.get("views_count") or 0,
        },
        "has_media": bool(media),
        "media": media,
        "node_origin": node_origin,
    }


def normalize_nested_quoted_tweet(parent_row: dict, metadata_row: dict | None) -> dict | None:
    quoted_status = parent_row.get("quoted_status")
    if isinstance(quoted_status, dict):
        return normalize_tweet(quoted_status, None, node_origin="nested_quoted_status")

    legacy = nested_quoted_legacy(metadata_row)
    if not legacy:
        return None

    tweet_id = legacy.get("id_str")
    if not tweet_id:
        return None

    metadata = metadata_row.get("metadata") if metadata_row else {}
    quoted = metadata.get("quoted_status_result") or {}
    result = quoted.get("result") or {}
    user_result = (((result.get("core") or {}).get("user_results") or {}).get("result") or {})
    user_core = user_result.get("core") or {}
    screen_name = user_core.get("screen_name")

    row = {
        "id": tweet_id,
        "screen_name": screen_name,
        "name": user_result.get("legacy", {}).get("name") if isinstance(user_result.get("legacy"), dict) else None,
        "created_at": legacy.get("created_at"),
        "full_text": legacy.get("full_text"),
        "reply_count": legacy.get("reply_count"),
        "retweet_count": legacy.get("retweet_count"),
        "quote_count": legacy.get("quote_count"),
        "favorite_count": legacy.get("favorite_count"),
        "bookmark_count": legacy.get("bookmark_count"),
    }
    return normalize_tweet(row, {"metadata": {"legacy": legacy}}, node_origin="metadata_quoted_status")


def sort_tweets(tweets: list[dict]) -> list[dict]:
    return sorted(tweets, key=lambda tweet: (tweet["timestamp_ms"], tweet["tweet_id"]))


def build_tweets(light_rows: list[dict], metadata_rows: list[dict]) -> list[dict]:
    metadata_by_id = {str(row.get("id")): row for row in metadata_rows if row.get("id")}
    tweets_by_message_id = {}

    for row in light_rows:
        tweet_id = str(row.get("id"))
        metadata_row = metadata_by_id.get(tweet_id)
        tweet = normalize_tweet(row, metadata_row, node_origin="top_level")
        if tweet:
            tweets_by_message_id[tweet["message_id"]] = tweet

        quoted = normalize_nested_quoted_tweet(row, metadata_row)
        if quoted and quoted["message_id"] not in tweets_by_message_id:
            tweets_by_message_id[quoted["message_id"]] = quoted

        retweeted_status = row.get("retweeted_status")
        if isinstance(retweeted_status, dict):
            retweeted = normalize_tweet(retweeted_status, None, node_origin="nested_retweeted_status")
            if retweeted and retweeted["message_id"] not in tweets_by_message_id:
                tweets_by_message_id[retweeted["message_id"]] = retweeted

    return sort_tweets(list(tweets_by_message_id.values()))


def add_edge(
    edges: list[dict],
    source_id: str,
    target_id: str | None,
    relation: str,
    known_node_ids: set[str],
) -> None:
    if not target_id:
        return
    edges.append(
        {
            "from": source_id,
            "to": target_id,
            "relation": relation,
            "target_exists": target_id in known_node_ids,
        }
    )


def sorted_ids(ids: list[str], tweet_by_message_id: dict[str, dict]) -> list[str]:
    return sorted(
        set(ids),
        key=lambda message_id_value: (
            tweet_by_message_id.get(message_id_value, {}).get("timestamp_ms", 0),
            message_id_value,
        ),
    )


def build_tweet_graph(tweets: list[dict]) -> dict:
    tweet_by_message_id = {tweet["message_id"]: tweet for tweet in tweets}
    known_node_ids = set(tweet_by_message_id)
    edges = []

    replies_by_parent = defaultdict(list)
    quotes_by_target = defaultdict(list)
    retweets_by_target = defaultdict(list)
    conversation_index = defaultdict(list)
    self_to_other_replies = defaultdict(list)
    other_to_self_replies = defaultdict(list)
    self_to_self_replies = defaultdict(list)

    for tweet in tweets:
        tweet_id_value = tweet["message_id"]
        conversation_index[tweet["conversation_id"]].append(tweet_id_value)

        parent_id = tweet["in_reply_to_message_id"]
        if parent_id:
            add_edge(edges, tweet_id_value, parent_id, "replies_to", known_node_ids)
            replies_by_parent[parent_id].append(tweet_id_value)

            parent = tweet_by_message_id.get(parent_id)
            if parent:
                if parent["speaker"] == "self" and tweet["speaker"] == "other":
                    self_to_other_replies[parent_id].append(tweet_id_value)
                elif parent["speaker"] == "other" and tweet["speaker"] == "self":
                    other_to_self_replies[parent_id].append(tweet_id_value)
                elif parent["speaker"] == "self" and tweet["speaker"] == "self":
                    self_to_self_replies[parent_id].append(tweet_id_value)

        quoted_id = tweet["quoted_message_id"]
        if quoted_id:
            add_edge(edges, tweet_id_value, quoted_id, "quotes", known_node_ids)
            quotes_by_target[quoted_id].append(tweet_id_value)

        retweeted_id = tweet["retweeted_message_id"]
        if retweeted_id:
            add_edge(edges, tweet_id_value, retweeted_id, "retweets", known_node_ids)
            retweets_by_target[retweeted_id].append(tweet_id_value)

    indexes = {
        "replies_by_parent": {
            parent_id: sorted_ids(child_ids, tweet_by_message_id)
            for parent_id, child_ids in sorted(replies_by_parent.items())
        },
        "self_to_other_replies": {
            parent_id: sorted_ids(child_ids, tweet_by_message_id)
            for parent_id, child_ids in sorted(self_to_other_replies.items())
        },
        "other_to_self_replies": {
            parent_id: sorted_ids(child_ids, tweet_by_message_id)
            for parent_id, child_ids in sorted(other_to_self_replies.items())
        },
        "self_to_self_replies": {
            parent_id: sorted_ids(child_ids, tweet_by_message_id)
            for parent_id, child_ids in sorted(self_to_self_replies.items())
        },
        "quote_to_quote_tweets": {
            quoted_id: sorted_ids(tweet_ids, tweet_by_message_id)
            for quoted_id, tweet_ids in sorted(quotes_by_target.items())
        },
        "retweet_to_retweets": {
            retweeted_id: sorted_ids(tweet_ids, tweet_by_message_id)
            for retweeted_id, tweet_ids in sorted(retweets_by_target.items())
        },
        "conversation_id_to_tweet_ids": {
            conversation_id: sorted_ids(tweet_ids, tweet_by_message_id)
            for conversation_id, tweet_ids in sorted(conversation_index.items())
        },
    }

    return {
        "nodes": [tweet["message_id"] for tweet in tweets],
        "edges": sorted(edges, key=lambda edge: (edge["relation"], edge["to"] or "", edge["from"])),
        "indexes": indexes,
    }


def build_conversation_paths(tweets: list[dict], graph: dict) -> list[dict]:
    tweet_by_message_id = {tweet["message_id"]: tweet for tweet in tweets}
    replies_by_parent = graph["indexes"]["replies_by_parent"]
    quotes_by_target = graph["indexes"]["quote_to_quote_tweets"]
    children_by_parent = {
        parent_id: sorted_ids(
            list(replies_by_parent.get(parent_id, []))
            + list(quotes_by_target.get(parent_id, [])),
            tweet_by_message_id,
        )
        for parent_id in set(replies_by_parent) | set(quotes_by_target)
    }
    paths = []

    root_ids = [
        tweet["message_id"]
        for tweet in tweets
        if not any(
            parent_id in tweet_by_message_id
            for parent_id in (
                tweet["in_reply_to_message_id"],
                tweet["quoted_message_id"],
            )
            if parent_id
        )
    ]

    for root_id in sorted_ids(root_ids, tweet_by_message_id):
        queue = deque([(root_id, [root_id])])
        emitted_single = False
        while queue:
            current_id, path = queue.popleft()
            children = children_by_parent.get(current_id, [])

            if not children:
                root = tweet_by_message_id[root_id]
                leaf = tweet_by_message_id[current_id]
                path_tweets = [tweet_by_message_id[item] for item in path]
                contains_self = any(tweet["speaker"] == "self" for tweet in path_tweets)
                contains_other = any(tweet["speaker"] == "other" for tweet in path_tweets)
                if contains_self and contains_other:
                    path_type = "mixed_interaction_path"
                elif contains_self:
                    path_type = "self_only_path"
                else:
                    path_type = "other_only_path"

                paths.append(
                    {
                        "path_id": f"twitter_path_{len(paths) + 1:06d}",
                        "conversation_id": root["conversation_id"],
                        "root_tweet_id": root["message_id"],
                        "leaf_tweet_id": leaf["message_id"],
                        "message_ids": path,
                        "path_depth": len(path),
                        "contains_self": contains_self,
                        "contains_other": contains_other,
                        "path_type": path_type,
                        "relations": [
                            "quotes"
                            if tweet_by_message_id[child].get("quoted_message_id") == parent
                            else "replies_to"
                            for parent, child in pairwise(path)
                        ],
                    }
                )
                emitted_single = True
                continue

            for child_id in children:
                if child_id in path:
                    continue
                queue.append((child_id, path + [child_id]))

        if not emitted_single and root_id not in children_by_parent:
            root = tweet_by_message_id[root_id]
            paths.append(
                {
                    "path_id": f"twitter_path_{len(paths) + 1:06d}",
                    "conversation_id": root["conversation_id"],
                    "root_tweet_id": root["message_id"],
                    "leaf_tweet_id": root["message_id"],
                    "message_ids": [root_id],
                    "path_depth": 1,
                    "contains_self": root["speaker"] == "self",
                    "contains_other": root["speaker"] == "other",
                    "path_type": "self_only_path" if root["speaker"] == "self" else "other_only_path",
                    "relations": [],
                }
            )

    return sorted(paths, key=lambda path: (tweet_by_message_id[path["root_tweet_id"]]["timestamp_ms"], path["path_id"]))


def build_thread_stats(tweets: list[dict], graph: dict, paths: list[dict]) -> dict:
    tweet_by_message_id = {tweet["message_id"]: tweet for tweet in tweets}
    edges_by_relation = Counter(edge["relation"] for edge in graph["edges"])
    external_edges_by_relation = Counter(
        edge["relation"] for edge in graph["edges"] if not edge["target_exists"]
    )
    replies_by_parent = graph["indexes"]["replies_by_parent"]
    self_to_other_replies = graph["indexes"]["self_to_other_replies"]

    top_self_tweets_by_other_replies = []
    for parent_id, child_ids in sorted(
        self_to_other_replies.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )[:20]:
        tweet = tweet_by_message_id.get(parent_id)
        top_self_tweets_by_other_replies.append(
            {
                "tweet_id": tweet["tweet_id"] if tweet else parent_id.replace("twitter_", ""),
                "message_id": parent_id,
                "other_reply_count": len(child_ids),
                "content_preview": (tweet.get("content", "")[:240] if tweet else ""),
            }
        )

    return {
        "total_tweets": len(tweets),
        "self_tweets": sum(1 for tweet in tweets if tweet["speaker"] == "self"),
        "other_tweets": sum(1 for tweet in tweets if tweet["speaker"] == "other"),
        "reply_edges": edges_by_relation["replies_to"],
        "quote_edges": edges_by_relation["quotes"],
        "retweet_edges": edges_by_relation["retweets"],
        "external_edge_targets": sum(1 for edge in graph["edges"] if not edge["target_exists"]),
        "external_edge_targets_by_relation": dict(sorted(external_edges_by_relation.items())),
        "conversation_count": len(graph["indexes"]["conversation_id_to_tweet_ids"]),
        "conversation_path_count": len(paths),
        "max_path_depth": max((path["path_depth"] for path in paths), default=0),
        "average_path_depth": (
            sum(path["path_depth"] for path in paths) / len(paths)
            if paths
            else 0
        ),
        "tweets_with_replies": len(replies_by_parent),
        "tweets_with_multiple_replies": sum(1 for child_ids in replies_by_parent.values() if len(child_ids) > 1),
        "self_tweets_with_other_replies": len(self_to_other_replies),
        "self_tweets_with_multiple_other_replies": sum(
            1 for child_ids in self_to_other_replies.values() if len(child_ids) > 1
        ),
        "max_other_replies_to_self_tweet": max((len(child_ids) for child_ids in self_to_other_replies.values()), default=0),
        "top_self_tweets_by_other_replies": top_self_tweets_by_other_replies,
        "node_origins": dict(Counter(tweet["node_origin"] for tweet in tweets)),
    }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    if configured_self_screen_name() is None:
        raise SystemExit("WRITING_RAG_TWITTER_SELF_SCREEN_NAME is required to normalize Twitter exports")
    light_rows = load_json_array(LIGHT_FILE)
    metadata_rows = load_json_array(METADATA_FILE)

    tweets = build_tweets(light_rows, metadata_rows)
    graph = build_tweet_graph(tweets)
    paths = build_conversation_paths(tweets, graph)
    stats = build_thread_stats(tweets, graph, paths)

    write_json(TWEETS_FILE, tweets)
    write_json(TWEET_GRAPH_FILE, graph)
    write_json(CONVERSATION_PATHS_FILE, paths)
    write_json(THREAD_STATS_FILE, stats)

    print(f"Normalized tweets: {len(tweets)}")
    print(f"Self tweets: {stats['self_tweets']}")
    print(f"Other tweets: {stats['other_tweets']}")
    print(f"Reply edges: {stats['reply_edges']}")
    print(f"Quote edges: {stats['quote_edges']}")
    print(f"Retweet edges: {stats['retweet_edges']}")
    print(f"Conversation paths: {len(paths)}")
    print(f"Max path depth: {stats['max_path_depth']}")


if __name__ == "__main__":
    main()
