# -*- coding: utf-8 -*-
import re


MAX_WORDS = 200
TITLE_BOOST_WORDS = 4

STOP_WORDS = {
    "company",
    "name",
    "city",
    "state",
}


def clean_text(text, max_words=MAX_WORDS, title_boost_words=TITLE_BOOST_WORDS):
    """Normalize resume text for both training and prediction."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./-]+", " ", text)
    words = [word for word in text.split() if word not in STOP_WORDS]
    words = words[:max_words]
    if title_boost_words:
        words = words[:title_boost_words] + words
    return " ".join(words)


def parse_fasttext_line(line):
    line = line.strip()
    if not line:
        return None, ""

    parts = line.split(maxsplit=1)
    label = parts[0]
    text = parts[1] if len(parts) > 1 else ""
    return label, text


def clean_fasttext_line(line):
    label, text = parse_fasttext_line(line)
    if not label:
        return ""
    return f"{label} {clean_text(text)}\n"
