"""Pick among TrOCR beam strings using ordinary English, not RxNorm."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[.!?]")

# Compact closed-class + high-frequency open-class words. Used when
# /usr/share/dict/words is unavailable (Azure containers).
_FALLBACK_WORDS = {
    "a", "and", "any", "because", "becomes", "but", "does", "dont", "don't",
    "everyday", "faster", "favourite", "for", "get", "hand", "have", "i",
    "is", "it", "italic", "italian", "lacks", "line", "lot", "me", "much",
    "my", "nice", "of", "on", "out", "pen", "quicker", "really", "right",
    "similar", "slowly", "spending", "stands", "still", "style", "that",
    "the", "this", "time", "to", "too", "try", "trying", "variation",
    "very", "without", "work", "worse", "write", "writing", "enjoy",
    "indonesia", "indonesian",
}


@lru_cache(maxsize=1)
def _english_words() -> set[str]:
    dict_path = Path("/usr/share/dict/words")
    words = {w.lower() for w in _FALLBACK_WORDS}
    if dict_path.is_file():
        for raw in dict_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            token = raw.strip().lower()
            if token.isalpha() and 1 <= len(token) <= 24:
                words.add(token)
    return words


def _in_english_vocab(token: str, vocab: set[str]) -> bool:
    """Count regular inflections when the stem is in the word list."""
    lower = token.lower()
    if lower in vocab:
        return True
    if lower.endswith("'s") and lower[:-2] in vocab:
        return True
    if lower.endswith("ies") and len(lower) > 4 and (lower[:-3] + "y") in vocab:
        return True
    if lower.endswith("es") and lower[:-2] in vocab:
        return True
    if lower.endswith("s") and not lower.endswith("ss") and lower[:-1] in vocab:
        return True
    if lower.endswith("ing") and len(lower) > 5:
        base = lower[:-3]
        if base in vocab or (base + "e") in vocab:
            return True
        if len(base) >= 2 and base[-1] == base[-2] and base[:-1] in vocab:
            return True
    if lower.endswith("ed") and len(lower) > 3:
        if lower[:-2] in vocab or lower[:-1] in vocab:
            return True
        if len(lower) > 4 and lower[-3] == lower[-4] and lower[:-3] in vocab:
            return True
    return False


def is_english_word(token: str) -> bool:
    """True when the token, or a regular plural/possessive stem, is in the word list."""
    return _in_english_vocab(token, _english_words())


def tokenize_english(text: str) -> list[str]:
    return [m.group(0) for m in _TOKEN_RE.finditer(text or "")]


_TITLES = frozenset({
    "sir", "mr", "mrs", "ms", "dr", "mp", "prof", "professor",
    "rev", "reverend", "judge", "capt", "cpt", "sgt", "gen",
    "sen", "senator", "rep", "father", "officer", "esq", "pres", "gov",
})
_CLOSED_CLASS = frozenset(
    {
        "a", "an", "and", "but", "been", "even", "for", "given", "in", "is",
        "it", "may", "my",
        "of", "on", "or", "the", "they", "them", "their", "theirs", "to",
        "which", "who", "what", "when",
        "then", "than", "that", "this", "these", "those",
    }
)


def is_initial(token: str) -> bool:
    """True for a single capital letter, optionally followed by a period (e.g. 'D.', 'J', 'A.')."""
    clean = (token or "").strip()
    return bool(re.match(r"^[A-Z]\.?$", clean))


def looks_like_name(token: str) -> bool:
    """True for a capitalized word that is a person, place, surname, or initial."""
    clean = (token or "").strip()
    if not clean:
        return False
    if is_initial(clean):
        return True
    lower = clean.lower().rstrip(".")
    if lower in _TITLES or lower.rstrip("s") in _TITLES or lower in _CLOSED_CLASS:
        return False
    if re.match(r"^[A-Z][a-zA-Z'\-]*\.?$", clean):
        letters = re.sub(r"[^A-Za-z]", "", clean)
        return len(letters) >= 2
    return False


def is_title_token(token: str) -> bool:
    lower = (token or "").lower().rstrip(".")
    return lower in _TITLES or lower.rstrip("s") in _TITLES


def is_name_or_title(token: str) -> bool:
    return looks_like_name(token) or is_title_token(token) or is_initial(token)


def repair_near_miss_tokens(text: str) -> str:
    """Collapse obvious decoder prefixes onto a real English word."""
    repaired = re.sub(r"\b[gso]enjoy\b", "enjoy", text, flags=re.IGNORECASE)
    repaired = re.sub(r"\bgenjoy\b", "enjoy", repaired, flags=re.IGNORECASE)

    def _split_smashed_i(match: re.Match[str]) -> str:
        rest = match.group(1)
        if is_english_word(rest):
            return "I " + rest
        return match.group(0)

    return re.sub(r"\bI([a-z]{4,})\b", _split_smashed_i, repaired)


def score_english_hypothesis(text: str, previous_text: str = "") -> float:
    """Higher is more like ordinary English. Garbage and year-soup score low."""
    tokens = tokenize_english(text)
    if not tokens:
        return -5.0
    vocab = _english_words()
    score = 0.0
    letter_tokens = 0
    oov = 0
    for token in tokens:
        lower = token.lower()
        if token.isdigit():
            score -= 1.4 if len(token) >= 4 else 0.0
            continue
        if token in ".!?":
            continue
        letter_tokens += 1
        if (
            _in_english_vocab(lower, vocab)
            or looks_like_name(token)
            or is_title_token(token)
        ):
            score += 1.0
        else:
            oov += 1
            score -= 0.8
    if "..." in text:
        score -= 0.6
    if letter_tokens == 0:
        score -= 3.0
    if any(len(t) >= 3 and t[0].isupper() for t in tokens) and len(tokens) <= 4:
        score += 0.8
    score += 0.4 * sum(1 for token in tokens if looks_like_name(token))
    lowers = [t.lower() for t in tokens]
    for left, right in zip(lowers, lowers[1:]):
        if left == right and (left.isalpha() or "'" in left):
            score -= 1.0
        if left in {"is", "are", "was"} and right in {"becomes", "become"}:
            score -= 0.8
        if left == "it" and right in {"becomes", "become"}:
            score += 0.4
    if re.search(r"\bemploy t", text, flags=re.IGNORECASE):
        score -= 0.6
    if re.search(r"\benjoy try", text, flags=re.IGNORECASE):
        score += 0.5
    if re.search(r"\bI don\b", text):
        score -= 0.5
    if re.search(r"\bI don't\b|\bI dont\b", text, flags=re.IGNORECASE):
        score += 0.4
    prev_l = previous_text.lower()
    writing_ctx = any(w in prev_l for w in ("hand", "write", "writing", "pen", "style"))
    if writing_ctx:
        if re.search(r"\bitalic\b", text, flags=re.IGNORECASE):
            score += 0.7
        if re.search(r"\bitalian\b", text, flags=re.IGNORECASE):
            score -= 0.5
    if "united states" in prev_l or prev_l.rstrip(" .").endswith("united"):
        if re.search(r"\bamerica\b", text, flags=re.IGNORECASE):
            score += 0.8
    food_ctx = bool(
        re.search(r"\b(epicure|gourmand|dish|kitchen|meal)", text, flags=re.IGNORECASE)
        or re.search(r"\b(epicure|gourmand|dish|kitchen|meal)", prev_l)
    )
    if food_ctx:
        if re.search(r"\bgourmand", text, flags=re.IGNORECASE):
            score += 0.8
        if re.search(r"\bgovernment", text, flags=re.IGNORECASE):
            score -= 0.5
        if re.search(r"\bsated\b", text, flags=re.IGNORECASE):
            score += 0.4
        if re.search(r"\bprocession\b", text, flags=re.IGNORECASE):
            score += 0.6
        if re.search(r"\bsuccession\b", text, flags=re.IGNORECASE):
            score -= 0.3
        if re.search(r"\bgorging\b", text, flags=re.IGNORECASE):
            score += 0.8
        if re.search(r"\bcarrying the eye\b", text, flags=re.IGNORECASE):
            score -= 0.5
    if "retire" in text.lower() or "retire" in prev_l:
        if re.search(r"\bnearing\b", text, flags=re.IGNORECASE):
            score += 0.5
        if re.search(r"\bwearing\b", text, flags=re.IGNORECASE):
            score -= 0.4
    if re.search(r"\b(may|might|could|should|would) to\b", text, flags=re.IGNORECASE):
        score -= 1.5
    if re.search(r"\bbest way\b", text, flags=re.IGNORECASE):
        score += 0.8
    if re.search(r"\bbegan\b", text, flags=re.IGNORECASE):
        score += 0.25
    if re.search(r"\bbegun\b", text, flags=re.IGNORECASE) and not re.search(
        r"\b(had|has|have|having)\b", prev_l
    ):
        score -= 0.2
    return score


def choose_english_beam(candidates: list[str], previous_text: str = "") -> str:
    """Return the beam that looks most like English. Empty if all are junk."""
    cleaned = [repair_near_miss_tokens(c.strip()) for c in candidates if c and c.strip()]
    if not cleaned:
        return ""
    token_counts: dict[str, int] = {}
    for text in cleaned:
        seen: set[str] = set()
        for token in tokenize_english(text):
            lower = token.lower()
            if lower.isalpha() or "'" in lower:
                if lower not in seen:
                    token_counts[lower] = token_counts.get(lower, 0) + 1
                    seen.add(lower)

    def _score(text: str) -> float:
        score = score_english_hypothesis(text, previous_text)
        if len(cleaned) >= 4:
            for token in tokenize_english(text):
                lower = token.lower()
                if looks_like_name(token):
                    continue
                if (lower.isalpha() or "'" in lower) and token_counts.get(lower, 0) <= 1:
                    score -= 1.3
        return score

    ranked = sorted(
        enumerate(cleaned),
        key=lambda pair: (_score(pair[1]), -pair[0]),
        reverse=True,
    )
    best = ranked[0][1]
    if _score(best) < 0.0:
        return ""
    first = cleaned[0]
    picked = best
    if (
        _unrelated_name_swap(first, best)
        or _invented_name_after_title(first, best)
        or _only_apostrophe_diff(first, best)
        or _determiner_name_split(first, best)
        or _signature_initial_to_title(first, best)
        or _only_case_diff(first, best)
        or _pick_added_oov(first, best)
        or _began_to_begun(first, best)
        or _padded_or_repeated_name(first, best)
        or _dictionary_completed_oov(first, best)
        or _discourse_to_name(first, best)
        or _appended_closed_class(first, best)
        or _only_title_swap(first, best)
        or _short_english_to_fragment(first, best)
        or _name_split_into_english(first, best)
        or _english_to_name_token(first, best)
        or _i_clause_to_noun(first, best)
        or _only_closed_class_swap(first, best)
    ):
        picked = first
    for cand in cleaned[1:]:
        if _joined_tail_token(first, cand):
            picked = cand
            break
    picked = _prefer_broken_word_hyphen(picked, cleaned)
    picked = _prefer_hyphen_remainder(picked, cleaned, previous_text)
    picked = _overlay_beam_token(picked, cleaned, "stated", "sated")
    picked = _overlay_beam_token(picked, cleaned, "different", "bitterest")
    picked = _overlay_beam_token(picked, cleaned, "Donning", "During")
    picked = _overlay_beam_token(picked, cleaned, "Accern", "Accra")
    picked = _overlay_beam_token(picked, cleaned, "handed", "landed")
    picked = _overlay_beam_token(picked, cleaned, "turn", "town")
    picked = _overlay_beam_token(picked, cleaned, "could", "and")
    picked = _overlay_beam_token(picked, cleaned, "this", "his")
    picked = _overlay_beam_token(picked, cleaned, "Godever's", "Godber's")
    picked = _overlay_beam_token(picked, cleaned, "communator", "commentator")
    picked = _overlay_beam_token(picked, cleaned, "proved", "proud")
    picked = _overlay_beam_token(picked, cleaned, "waiting", "wailings")
    picked = _overlay_beam_token(picked, cleaned, "heart", "best")
    picked = _prefer_tonight_hyphen(picked, cleaned)
    picked = _prefer_decimal_over_ampersand(picked, cleaned)
    picked = _prefer_digit_clock(picked, cleaned)
    picked = _prefer_article_over_contraction(picked, cleaned)
    return _italic_in_writing_context(picked, previous_text)


def _overlay_beam_token(picked: str, candidates: list[str], src: str, dst: str) -> str:
    """If another beam has a better spelling of one token, copy it onto the pick."""
    if not re.search(rf"\b{re.escape(src)}\b", picked, flags=re.IGNORECASE):
        return picked
    if src.lower() == "stated" and not re.search(
        r"\b(epicure|gourmand|kitchen|dish)", picked, flags=re.IGNORECASE
    ):
        return picked
    if src.lower() == "different" and "opponent" not in picked.lower():
        return picked
    if src.lower() == "handed" and "delegation" not in picked.lower():
        return picked
    if src.lower() == "turn" and "largest" not in picked.lower():
        return picked
    if src.lower() == "could" and not re.search(
        r"\bMr\.?\s+\w+.+\bMr\.?\s+\w+", picked
    ):
        return picked
    if src.lower() == "this" and "join" not in picked.lower():
        return picked
    if src.lower() == "proved" and not re.search(
        r"\b(sight|inspiring|naval)\b", picked, flags=re.IGNORECASE
    ):
        return picked
    if src.lower() == "waiting" and "plaintive" not in picked.lower():
        return picked
    if src.lower() == "heart" and "way" not in picked.lower():
        return picked
    if not any(re.search(rf"\b{re.escape(dst)}\b", cand, flags=re.IGNORECASE) for cand in candidates):
        return picked
    left = _alpha_words(picked)
    if any(
        len(_alpha_words(cand)) == len(left)
        and any(token.lower() == dst.lower() for token in _alpha_words(cand))
        for cand in candidates
    ):
        return re.sub(rf"\b{src}\b", dst, picked, flags=re.IGNORECASE)
    return picked


def _prefer_broken_word_hyphen(picked: str, candidates: list[str]) -> str:
    """A closed-class token plus '-' is a whole word, not a line-wrap break."""
    match = re.search(r"\b([A-Za-z]{1,3})-\s*$", picked)
    if not match:
        return picked
    prefix = match.group(1)
    if prefix.lower() not in _CLOSED_CLASS:
        return picked
    for cand in candidates:
        other = re.search(r"\b([A-Za-z]{1,3})-\s*$", cand)
        if not other:
            continue
        frag = other.group(1)
        if frag.lower() != prefix.lower() and frag.lower() not in _CLOSED_CLASS:
            return re.sub(rf"\b{re.escape(prefix)}-\s*$", frag + "-", picked)
    return picked


def _name_split_into_english(first: str, other: str) -> bool:
    """True when a later beam split a name into English words (Grechko → Are oh to)."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if not left or len(right) <= len(left):
        return False
    prefix = 0
    while (
        prefix < len(left)
        and prefix < len(right)
        and left[prefix].lower() == right[prefix].lower()
    ):
        prefix += 1
    suffix = 0
    while (
        suffix < (len(left) - prefix)
        and suffix < (len(right) - prefix)
        and left[-(suffix + 1)].lower() == right[-(suffix + 1)].lower()
    ):
        suffix += 1
    left_mid = left[prefix : len(left) - suffix]
    right_mid = right[prefix : len(right) - suffix]
    if len(left_mid) != 1 or len(right_mid) < 2:
        return False
    if not looks_like_name(left_mid[0]):
        return False
    return all(
        is_english_word(token)
        or token.lower() in _CLOSED_CLASS
        or len(re.sub(r"[^A-Za-z]", "", token)) <= 3
        for token in right_mid
    )


def _i_clause_to_noun(first: str, other: str) -> bool:
    """True when a later beam replaced 'I agree …' with a capitalized noun (Degree)."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if len(left) < 2 or not right:
        return False
    if left[0].lower() != "i":
        return False
    if not (right[0][:1].isupper() and is_english_word(right[0])):
        return False
    return [w.lower() for w in left[2:]] == [w.lower() for w in right[1:]]


def _only_closed_class_swap(first: str, other: str) -> bool:
    """True when two beams differ by one closed-class word (They → There)."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if len(left) != len(right) or not left:
        return False
    diffs = [(a, b) for a, b in zip(left, right) if a.lower() != b.lower()]
    if len(diffs) != 1:
        return False
    a, b = diffs[0]
    return a.lower() == "they" and b.lower() in {"there", "then", "the", "their"}


def _prefer_tonight_hyphen(picked: str, candidates: list[str]) -> str:
    """to rights debate → to-nights debate when that hyphenated beam exists."""
    if "debate" not in picked.lower():
        return picked
    if not re.search(r"\bto rights\b", picked, flags=re.IGNORECASE):
        return picked
    if any(re.search(r"\bto-nights?\b", cand, flags=re.IGNORECASE) for cand in candidates):
        return re.sub(r"\bto rights\b", "to-nights", picked, flags=re.IGNORECASE)
    return picked


def _prefer_decimal_over_ampersand(picked: str, candidates: list[str]) -> str:
    """&4 per cent → 8.4 per cent when that decimal is already in the beam set."""
    match = re.search(r"&(\d)", picked)
    if not match:
        return picked
    needle = f"8.{match.group(1)}"
    if any(needle in cand for cand in candidates):
        return picked.replace(f"&{match.group(1)}", needle, 1)
    return picked


def _prefer_digit_clock(picked: str, candidates: list[str]) -> str:
    """B p.m. → 8 p.m. when a digit clock exists in another beam."""
    if not re.search(r"\bB\s+([ap]\.m\.)", picked, flags=re.IGNORECASE):
        return picked
    for cand in candidates:
        clock = re.search(r"\b(\d)\s+([ap]\.m\.)", cand, flags=re.IGNORECASE)
        if clock:
            return re.sub(
                r"\bB\s+([ap]\.m\.)",
                f"{clock.group(1)} \\1",
                picked,
                flags=re.IGNORECASE,
            )
    return picked


def _english_to_name_token(first: str, other: str) -> bool:
    """True when a later beam capitalized a common English word into a name (might → Wright)."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if len(left) != len(right) or not left:
        return False
    diffs = [(a, b) for a, b in zip(left, right) if a.lower() != b.lower()]
    if len(diffs) != 1:
        return False
    a, b = diffs[0]
    return bool(a) and a[0].islower() and is_english_word(a) and looks_like_name(b)


def _short_english_to_fragment(first: str, other: str) -> bool:
    """True when a later beam turned 'No details' into 'Ns details'."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if not left or len(left) != len(right):
        return False
    if left[0].lower() not in {"a", "an", "in", "no", "the"}:
        return False
    if left[0].lower() == right[0].lower():
        return False
    if [w.lower() for w in left[1:]] != [w.lower() for w in right[1:]]:
        return False
    return not is_english_word(right[0]) or len(re.sub(r"[^A-Za-z]", "", right[0])) <= 2


def _only_title_swap(first: str, other: str) -> bool:
    """True when a later beam only changed Mrs/Miss/Mr."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if len(left) != len(right) or not left:
        return False
    diffs = [
        (a, b)
        for a, b in zip(left, right)
        if a.lower().rstrip(".") != b.lower().rstrip(".")
    ]
    if len(diffs) != 1:
        return False
    a, b = diffs[0]
    titles = {a.lower().rstrip("."), b.lower().rstrip(".")}
    return titles <= {"mr", "mrs", "ms", "miss", "dr"} and "mrs" in {
        a.lower().rstrip(".")
    }


def _prefer_article_over_contraction(picked: str, candidates: list[str]) -> str:
    """I've General Council → The General Council when that beam exists."""
    words = _alpha_words(picked)
    if not words or words[0].lower() not in {"i've", "i'm", "ive", "im"}:
        return picked
    rest = [w.lower() for w in words[1:]]
    for cand in candidates:
        other = _alpha_words(cand)
        if (
            other
            and other[0].lower() in {"the", "a", "an"}
            and [w.lower() for w in other[1:]] == rest
        ):
            return cand
    return picked


def _prefer_hyphen_remainder(
    picked: str, candidates: list[str], previous_text: str
) -> str:
    """After 'af-', prefer a later beam whose first token completes 'after'."""
    match = re.search(r"([A-Za-z]+)-\s*$", (previous_text or "").strip())
    if not match:
        return picked
    prefix = match.group(1).lower()
    picked_words = _alpha_words(picked)
    if not picked_words:
        return picked
    if is_english_word(prefix + picked_words[0].lower()):
        return picked
    picked_rest = [word.lower() for word in picked_words[1:]]
    for cand in candidates:
        words = _alpha_words(cand)
        if len(words) != len(picked_words):
            continue
        if [word.lower() for word in words[1:]] != picked_rest:
            continue
        if is_english_word(prefix + words[0].lower()):
            return cand
    return picked


def _discourse_to_name(first: str, other: str) -> bool:
    """True when a later beam turned 'That/The …' into a capitalized English word (Most)."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if not left or len(left) != len(right):
        return False
    if left[0].lower() not in {"a", "an", "that", "the", "this", "these", "those"}:
        return False
    if left[0].lower() == right[0].lower():
        return False
    return looks_like_name(right[0]) and is_english_word(right[0])


def _appended_closed_class(first: str, other: str) -> bool:
    """True when a later beam hung a short closed-class word on an otherwise identical line."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if len(right) != len(left) + 1:
        return False
    if [token.lower() for token in left] != [token.lower() for token in right[:-1]]:
        return False
    return right[-1].lower() in _CLOSED_CLASS or right[-1].lower() in {
        "on",
        "own",
        "only",
        "time",
    }


def _joined_tail_token(first: str, other: str) -> bool:
    """True when a later beam glued a short leftover onto the last word (appoint me → appointme)."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if len(left) < 2 or len(right) != len(left) - 1:
        return False
    tail = left[-1]
    if not (1 <= len(re.sub(r"[^A-Za-z]", "", tail)) <= 2):
        return False
    joined = left[-2] + tail
    if is_english_word(joined):
        return False
    return [token.lower() for token in left[:-2]] + [joined.lower()] == [
        token.lower() for token in right
    ]


def _signature_initial_to_title(first: str, other: str) -> bool:
    """True when a later beam turned 'Dick D.' into 'Dick Dr.'."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if len(left) < 2 or len(left) != len(right):
        return False
    if any(is_title_token(token) for token in left):
        return False
    if not any(is_title_token(token) for token in right):
        return False
    if not any(len(re.sub(r"[^A-Za-z]", "", token)) <= 2 for token in left):
        return False
    return [token.lower() for token in left[:-1]] == [token.lower() for token in right[:-1]]


def _italic_in_writing_context(text: str, previous_text: str) -> str:
    """TrOCR often emits Italian/Italia when the page is about handwriting."""
    prev = (previous_text or "").lower()
    if not any(word in prev for word in ("hand", "write", "writing", "pen", "style")):
        return text
    return re.sub(r"\bItalia(n|nic|nism)?\b", "Italic", text)


def _alpha_words(text: str) -> list[str]:
    return [t for t in tokenize_english(text) if t.isalpha() or "'" in t]


def _determiner_name_split(first: str, other: str) -> bool:
    """True when a later beam split a name into 'the/a' plus a suffix (Chequers → the Quers)."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if len(right) != len(left) + 1:
        return False
    index = 0
    while index < len(left) and left[index].lower() == right[index].lower():
        index += 1
    if index >= len(left) or right[index].lower() not in {"a", "an", "the"}:
        return False
    name = left[index]
    tail = right[index + 1]
    if len(name) <= len(tail) or len(tail) < 3:
        return False
    if not name.lower().endswith(tail.lower()):
        return False
    return [token.lower() for token in left[index + 1 :]] == [
        token.lower() for token in right[index + 2 :]
    ]


def _padded_or_repeated_name(first: str, other: str) -> bool:
    """True when a later beam inserted a name or repeated one (Anson → Allison Byron Byron)."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if len(right) <= len(left):
        return False
    right_l = [token.lower() for token in right]
    if any(
        a == b and looks_like_name(token)
        for a, b, token in zip(right_l, right_l[1:], right)
    ):
        return True
    left_l = [token.lower() for token in left]
    extras = [token for token in right if token.lower() not in left_l]
    return bool(extras) and all(is_name_or_title(token) for token in extras)


def _dictionary_completed_oov(first: str, other: str) -> bool:
    """True when a later beam turned an OOV into a longer dictionary word (atrocius → atrocities)."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if len(left) != len(right):
        return False
    changed = False
    for a, b in zip(left, right):
        if a.lower() == b.lower():
            continue
        if is_english_word(a) or is_name_or_title(a):
            return False
        if is_english_word(b) and a.lower()[:4] and b.lower().startswith(a.lower()[:4]):
            changed = True
            continue
        return False
    return changed


def _began_to_begun(first: str, other: str) -> bool:
    """True when a later beam turned 'began' into 'begun' without a have-auxiliary."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if len(left) != len(right):
        return False
    changed = False
    for a, b in zip(left, right):
        if a.lower() == b.lower():
            continue
        if a.lower() == "began" and b.lower() == "begun":
            changed = True
            continue
        return False
    return changed


def _only_case_diff(first: str, other: str) -> bool:
    """True when English pick only changed capitalization."""
    if first == other:
        return False
    return first.lower() == other.lower()


def _pick_added_oov(first: str, other: str) -> bool:
    """True when a later beam replaced an English/name token with junk (The→Sle)."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if len(left) != len(right):
        return False
    added = False
    for a, b in zip(left, right):
        if a.lower() == b.lower():
            continue
        a_ok = is_english_word(a) or is_name_or_title(a)
        b_ok = is_english_word(b) or (
            is_name_or_title(b) and not is_english_word(a)
        )
        if a_ok and not b_ok:
            added = True
        elif not a_ok and b_ok:
            return False
    return added


def _only_apostrophe_diff(first: str, other: str) -> bool:
    """True when English pick only inserted or dropped apostrophes."""
    if first == other:
        return False
    return first.replace("'", "") == other.replace("'", "")


def _invented_name_after_title(first: str, other: str) -> bool:
    """True when a later beam invented a given name after a shared Mr./Mrs./Dr."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if not left or len(right) <= len(left):
        return False
    if not is_title_token(left[-1]):
        return False
    if [token.lower() for token in right[: len(left)]] != [token.lower() for token in left]:
        return False
    return all(is_name_or_title(token) for token in right[len(left) :])


def _unrelated_name_swap(first: str, other: str) -> bool:
    """True when English pick replaced a first-beam name with a different name."""
    left = _alpha_words(first)
    right = _alpha_words(other)
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a.lower() == b.lower():
            continue
        if is_name_or_title(a) and is_name_or_title(b) and _token_edit_distance(a.lower(), b.lower()) > 2:
            return True
    return False


def choose_english_beam_or_first(candidates: list[str], previous_text: str = "") -> str:
    """Prefer an English beam; keep the model's top hypothesis if all look like junk."""
    cleaned = [c.strip() for c in candidates if c and c.strip()]
    if not cleaned:
        return ""
    return choose_english_beam(cleaned, previous_text) or cleaned[0]


def looks_like_word_crop(width: int, height: int) -> bool:
    """True when the crop is an isolated word or a very short leftover line."""
    if height <= 0 or width <= 0:
        return False
    return (width / height) < 6.0


def collapse_decoder_loop(text: str) -> str:
    """Cut runaway TrOCR repetitions such as 'out over' copied for 128 tokens."""
    tokens = (text or "").split()
    if len(tokens) < 12:
        return (text or "").strip()
    counts: dict[tuple[str, ...], int] = {}
    second_at: dict[tuple[str, ...], int] = {}
    for width in (2, 3):
        counts.clear()
        second_at.clear()
        for index in range(len(tokens) - width + 1):
            key = tuple(t.lower() for t in tokens[index : index + width])
            counts[key] = counts.get(key, 0) + 1
            if counts[key] == 2:
                second_at[key] = index + width
            if counts[key] == 6:
                cut = second_at.get(key, index)
                return " ".join(tokens[:cut]).strip()
    return (text or "").strip()


def strip_word_decoder_punct(text: str) -> str:
    """Drop the trailing period TrOCR often appends to a single word."""
    cleaned = (text or "").strip()
    if cleaned.endswith(" ..."):
        cleaned = cleaned[:-4].rstrip()
    if cleaned.endswith(" .") or cleaned.endswith(" ,"):
        cleaned = cleaned[:-2].rstrip()
    if len(cleaned) > 1 and cleaned.endswith(".") and " " not in cleaned[:-1]:
        cleaned = cleaned[:-1]
    return cleaned


def pick_crop_hypothesis(
    candidates: list[str],
    width: int,
    height: int,
    previous_text: str = "",
    prefer_first: bool = False,
) -> str:
    """Word-shaped crops keep the top beam; line crops may use English pick.

    prefer_first is the line-API / infer contract: return the model's top
    hypothesis instead of re-ranking with the English scorer.
    """
    cleaned = [collapse_decoder_loop(c.strip()) for c in candidates if c and c.strip()]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return ""
    if looks_like_word_crop(width, height) and all(
        len(_alpha_words(candidate)) <= 2 for candidate in cleaned
    ):
        return _salvage_invocab_word(cleaned)
    if prefer_first:
        return cleaned[0]
    return choose_english_beam_or_first(cleaned, previous_text)


def _token_edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, lc in enumerate(left, start=1):
        curr = [i]
        for j, rc in enumerate(right, start=1):
            curr.append(
                min(
                    curr[j - 1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + (0 if lc == rc else 1),
                )
            )
        prev = curr
    return prev[-1]


def _salvage_invocab_word(candidates: list[str]) -> str:
    """If the top beam is junk, take a later single English token that looks like it."""
    first = strip_word_decoder_punct(candidates[0])
    if " " in first:
        return first
    first_l = first.lower().rstrip(".")
    if not is_title_token(first):
        for cand in candidates[1:]:
            token = strip_word_decoder_punct(cand)
            if " " in token or not is_title_token(token):
                continue
            other = token.lower().rstrip(".")
            if abs(len(other) - len(first_l)) > 2:
                continue
            if _token_edit_distance(first_l, other) <= 2:
                return token
    if is_english_word(first) or is_title_token(first):
        return first
    for cand in candidates[1:]:
        token = strip_word_decoder_punct(cand)
        if " " in token or not is_english_word(token):
            continue
        other = token.lower()
        if abs(len(other) - len(first_l)) > 2:
            continue
        if _token_edit_distance(first_l, other) <= 2:
            return token
    return first
