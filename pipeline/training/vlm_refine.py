"""Optional MLX vision-language second pass for lines TrOCR cannot finish."""

from __future__ import annotations

import base64
import os
import re
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from pipeline.training.english_beam import (
    is_english_word,
    is_initial,
    is_name_or_title,
    is_title_token,
    score_english_hypothesis,
    tokenize_english,
)

# First words TrOCR often swaps for a visually similar English word.
# VLM is allowed to replace these even when the rest of the line looks fine.
CONFUSABLE_FIRST_WORDS = frozenset(
    {
        "italian",
        "italia",
        "given",
        "then",
        "been",
        "employ",
        "energy",
        "open",
        "when",
        "even",
    }
)

try:
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm import load as mlx_load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config as mlx_load_config
    MLX_VLM_AVAILABLE = True
except Exception:
    mlx_generate = None
    mlx_load = None
    apply_chat_template = None
    mlx_load_config = None
    MLX_VLM_AVAILABLE = False

DEFAULT_VLM_ID = os.environ.get(
    "HTR_VLM_ID",
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
)


@lru_cache(maxsize=1)
def load_vlm(model_id: str = DEFAULT_VLM_ID) -> tuple[Any, Any]:
    if not MLX_VLM_AVAILABLE or mlx_load is None:
        raise RuntimeError("mlx-vlm is not installed")
    return mlx_load(model_id)


def azure_openai_configured() -> bool:
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"))


def vlm_refine_available() -> bool:
    return bool(MLX_VLM_AVAILABLE or azure_openai_configured())


def _line_prompt(hypothesis: str = "", previous_text: str = "") -> str:
    user = (
        "Transcribe this single line of English handwriting exactly as written. "
        "Pay extreme attention to proper nouns, names, initials, and signatures. "
        "Do NOT autocorrect unusual spellings, surnames, or uncommon names to standard words "
        "(e.g. preserve 'Jon' vs 'John', 'Dick D.' vs 'Dickie'). "
        "Keep the writer's exact spelling, capitalization, and punctuation. "
        "Output only the line text with no extra commentary."
    )
    if hypothesis.strip():
        user += f" A first-pass OCR guessed: {hypothesis.strip()!r}."
    if previous_text.strip():
        user += f" Previous lines context: {previous_text.strip()}"
    return user


def should_refine_with_vlm(text: str) -> bool:
    """True when TrOCR looks unfinished or starts with a known first-word miss."""
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    tokens = [token for token in tokenize_english(cleaned) if token.isalpha() or "'" in token]
    if not tokens:
        return True
    if tokens[0].lower() in CONFUSABLE_FIRST_WORDS:
        return True
    if re.search(r'[\"“”]', cleaned):
        return True
    return any(
        not is_english_word(token) and not is_name_or_title(token) for token in tokens
    )


def refine_line(
    image: Image.Image | str | Path,
    hypothesis: str = "",
    previous_text: str = "",
    model_id: str = DEFAULT_VLM_ID,
) -> str:
    """VLM transcription: MLX locally, Azure OpenAI gpt-4o when configured."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    rgb = image.convert("RGB")
    if MLX_VLM_AVAILABLE:
        try:
            return _refine_mlx(rgb, hypothesis, previous_text, model_id)
        except Exception:
            pass
    if azure_openai_configured():
        return _refine_azure_openai(rgb, hypothesis, previous_text)
    return ""


def _refine_mlx(
    rgb: Image.Image,
    hypothesis: str,
    previous_text: str,
    model_id: str,
) -> str:
    tmp = Path("/tmp/htr_vlm_line.png")
    rgb.save(tmp)
    model, processor = load_vlm(model_id)
    config = mlx_load_config(model_id)
    prompt = apply_chat_template(processor, config, _line_prompt(hypothesis, previous_text), num_images=1)
    result = mlx_generate(
        model,
        processor,
        prompt,
        image=str(tmp),
        max_tokens=96,
        temperature=0.0,
        verbose=False,
    )
    text = getattr(result, "text", None) or str(result)
    return _first_line(text)


def _refine_azure_openai(rgb: Image.Image, hypothesis: str, previous_text: str) -> str:
    buffer = BytesIO()
    rgb.save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-08-01-preview"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _line_prompt(hypothesis, previous_text)},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 160,
        "temperature": 0,
    }
    headers = {
        "api-key": os.environ["AZURE_OPENAI_API_KEY"],
        "Content-Type": "application/json",
    }
    response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    return _first_line(str(text or ""))


def _first_line(text: str) -> str:
    cleaned = (text or "").strip().strip('"').strip()
    lower = cleaned.lower()
    if (
        "no handwriting visible" in lower
        or "no text visible" in lower
        or "not visible" in lower
        or lower.startswith("i'm sorry")
        or lower.startswith("sorry")
        or "cannot see any handwriting" in lower
    ):
        return ""
    parts = [normalize_line_text(part) for part in cleaned.splitlines() if part.strip()]
    if not parts:
        return ""
    joined = [parts[0]]
    for part in parts[1:]:
        if part and part[0].islower():
            joined.append(part)
            continue
        break
    if len(joined) == 1:
        return joined[0]
    return " ".join(repair_line_continuations(joined))


def normalize_line_text(text: str) -> str:
    """Strip decoder junk that is not part of the handwriting."""
    cleaned = (text or "").replace("...", ".").replace(" …", ".")
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace(" .", ".")
    if cleaned.endswith(" ."):
        cleaned = cleaned[:-2] + "."
    cleaned = re.sub(r"(\w+'s)\s+s\b", r"\1", cleaned)
    cleaned = re.sub(r"(\w) 's\b", r"\1's", cleaned)
    if re.match(r"^r\.\s+[A-Z]", cleaned):
        cleaned = "M" + cleaned
    cleaned = re.sub(r"\s*[\"“”]+\s*$", "", cleaned)
    cleaned = _split_smashed_signature(cleaned)
    cleaned = re.sub(r"(\w{4,})\s+(?![aiAI]$)[A-Za-z]$", r"\1", cleaned)
    cleaned = re.sub(r"(\w{4,})\.\s+(?![aiAI]$)[A-Za-z]$", r"\1.", cleaned)
    cleaned = re.sub(
        r",\s+(Which|That|This|Who|What|Said)\b",
        lambda match: ", " + match.group(1).lower(),
        cleaned,
    )
    cleaned = re.sub(r"\bto Delay\b", "to delay", cleaned)
    cleaned = re.sub(r"\bThe Conference\b", "The conference", cleaned)
    cleaned = re.sub(r"\bHouse of chiefs\b", "House of Chiefs", cleaned)
    cleaned = re.sub(r"\bNorthern Rhodesian,", "Northern Rhodesia,", cleaned)
    cleaned = re.sub(r",\s+MPs\b", ", MP", cleaned)
    cleaned = re.sub(r"\bMPs for\b", "MP for", cleaned)
    cleaned = re.sub(r"\.\s+Most\.?$", ".", cleaned)
    cleaned = re.sub(r"([A-Z][a-z]+)\s+Said\b", r"\1 said", cleaned)
    cleaned = re.sub(r"^in ([A-Z]{3,})\b", r"In \1", cleaned)
    cleaned = re.sub(r"\bforeign Minister\b", "Foreign Minister", cleaned)
    return _period_before_trailing_signature(cleaned)


_PAGE_CHROME = {
    "navigation menu",
    "jump to navigation",
    "jump to search",
    "edit",
    "[edit]",
    "4th century union",
}


def is_page_chrome_line(text: str) -> bool:
    """True for leftover Wikipedia/UI strings or VLM refusal apologies."""
    cleaned = normalize_line_text(text or "")
    if not cleaned:
        return True
    lower = cleaned.lower().rstrip(".,!?;:")
    if (
        lower in _PAGE_CHROME
        or lower.startswith("4th century")
        or lower.startswith("i'm sorry")
        or lower.startswith("sorry")
        or "no handwriting" in lower
        or "no text visible" in lower
        or "cannot see any handwriting" in lower
    ):
        return True
    if "this article" in lower or "what links here" in lower:
        return True
    if re.search(r"\bi am i am\b", lower):
        return True
    if lower.startswith("#") and len(cleaned.split()) <= 4:
        return True
    return False


def _split_smashed_signature(text: str) -> str:
    """Turn 'DICKD.' into 'Dick D.' so the name/initial signature matcher can see it."""

    def _split(match: re.Match[str]) -> str:
        return match.group(1).title() + " " + match.group(2) + "."

    return re.sub(r"\b([A-Z]{4,})([A-Z])\.?$", _split, text)


def _period_before_trailing_signature(text: str) -> str:
    """Restore the sentence break when a signature was glued onto the last line."""
    tokens = (text or "").split()
    if len(tokens) < 4:
        return text
    tail = " ".join(tokens[-2:])
    if not looks_like_signature_line(tail):
        return text
    head = " ".join(tokens[:-2]).rstrip(".,")
    if head.endswith((".", "!", "?")):
        return text
    return head + ". " + tail


def looks_like_signature_line(text: str) -> bool:
    """Keep short name lines such as 'Dick D.' that follow a finished sentence."""
    tokens = [token for token in re.findall(r"[A-Za-z]+", text or "")]
    # One capitalized word ("Spot.") is a wrap leftover, not a signed name.
    # A real signature has a short initial ("Dick D."), not "West German Government".
    if len(tokens) < 2 or len(tokens) > 4:
        return False
    if not all(token[0].isupper() for token in tokens):
        return False
    if not any(len(token) >= 3 for token in tokens):
        return False
    if any(is_title_token(token) for token in tokens):
        return False
    return any(len(token) <= 2 for token in tokens)


def is_page_crumb_line(text: str, previous: str = "") -> bool:
    """True for leftover strokes decoded as 'I - I' after a real sentence."""
    cleaned = normalize_line_text(text or "")
    if not cleaned or is_page_chrome_line(cleaned):
        return True
    if looks_like_signature_line(cleaned):
        return False
    tokens = cleaned.split()
    if len(tokens) > 3:
        return False
    compact = re.sub(r"[^A-Za-z]", "", cleaned)
    if len(compact) <= 2 and len(tokens) >= 2:
        return True
    if len(tokens) == 1 and len(compact) <= 2:
        if compact.lower() in {
            "a",
            "i",
            "an",
            "at",
            "he",
            "in",
            "it",
            "me",
            "my",
            "no",
            "of",
            "on",
            "or",
            "so",
            "to",
            "we",
        } or is_title_token(cleaned):
            return False
        return True
    previous_clean = normalize_line_text(previous or "")
    if previous_clean.endswith((".", "!", "?")) and score_english_hypothesis(cleaned) < 0.5:
        if not any(len(re.sub(r"[^A-Za-z]", "", token)) >= 4 for token in tokens):
            return True
    return False


def is_stray_page_tail(text: str, page_so_far: str) -> bool:
    """Drop a leftover crop after the page has already finished a sentence."""
    prior = " ".join((page_so_far or "").split())
    if not prior:
        return False
    words = [w for w in (text or "").split() if re.search(r"[A-Za-z]", w)]
    if not (1 <= len(words) <= 4):
        return False
    if looks_like_signature_line(text):
        if re.search(r"\b(me|sincerely|yours|regards)\b", prior, flags=re.IGNORECASE):
            return False
        return len(prior.split()) >= 8
    if re.search(r"\b(Mr|Mrs|Ms|Dr|Sir)\.$", prior):
        return False
    if prior[-1] not in ".!?":
        return False
    if len(words) > 3:
        return False
    content = [re.sub(r"[^A-Za-z]", "", w) for w in words]
    content = [w for w in content if len(w) >= 4]
    prior_words = set(re.findall(r"[A-Za-z]{4,}", prior.lower()))
    if not content:
        return True
    return not any(w.lower() in prior_words for w in content)


def is_unrelated_page_tail(text: str, page_so_far: str) -> bool:
    """True for a short title-case leftover that shares no word with the page."""
    if looks_like_signature_line(text):
        return False
    tokens = [token for token in (text or "").split() if re.search(r"[A-Za-z]", token)]
    if not (2 <= len(tokens) <= 4):
        return False
    names = [re.sub(r"[^A-Za-z]", "", token) for token in tokens]
    names = [name for name in names if len(name) >= 3]
    if not names or not all(name[0].isupper() for name in names):
        return False
    prior = set(re.findall(r"[A-Za-z]{4,}", page_so_far.lower()))
    if len(prior) < 8:
        return False
    words = [word.lower() for word in re.findall(r"[A-Za-z]{4,}", text)]
    if not words:
        return False
    return not any(word in prior for word in words)


def filter_page_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if not is_page_chrome_line(line)]


_NO_SENTENCE_BREAK_AFTER = {
    "a",
    "an",
    "and",
    "another",
    "at",
    "but",
    "by",
    "dr",
    "east",
    "for",
    "from",
    "in",
    "asking",
    "bear",
    "into",
    "led",
    "mr",
    "mrs",
    "ms",
    "north",
    "of",
    "on",
    "onto",
    "or",
    "south",
    "than",
    "that",
    "the",
    "these",
    "this",
    "those",
    "to",
    "west",
    "with",
}


def _last_alpha_word(text: str) -> str:
    tokens = (text or "").split()
    return re.sub(r"[^A-Za-z']", "", tokens[-1]) if tokens else ""


def _wrapped_proper_name(text: str, nxt: str) -> bool:
    """True when a capitalized last word continues onto the next line."""
    last = _last_alpha_word(text)
    return bool(last) and last[0].isupper() and bool(nxt) and nxt[0].isupper()


def _restore_house_of_wrap(text: str, nxt: str) -> str:
    """Capitalize the institution after a wrapped 'House of …' break."""
    if _last_alpha_word(text) != "House":
        return nxt
    match = re.match(r"^(of )([A-Za-z]+)(.*)$", nxt)
    if not match:
        return nxt
    title = match.group(2)[:1].upper() + match.group(2)[1:]
    return match.group(1) + title + match.group(3)


def repair_line_continuations(lines: list[str]) -> list[str]:
    """Fix line-break punctuation: drop mid-sentence periods, add sentence stops."""
    repaired = [normalize_line_text(line) for line in lines]
    for index, text in enumerate(repaired):
        nxt = repaired[index + 1] if index + 1 < len(repaired) else ""
        if not text or not nxt:
            continue
        nxt = _restore_house_of_wrap(text, nxt)
        repaired[index + 1] = nxt
        hyphen = re.search(r"([A-Za-z]+)-$", text)
        if hyphen and nxt:
            first = nxt.split()[0]
            frag = re.sub(r"[^A-Za-z']", "", first)
            if (
                frag
                and frag[0].isupper()
                and is_english_word(hyphen.group(1) + frag)
            ):
                nxt = frag.lower() + nxt[len(first) :]
                repaired[index + 1] = nxt
        last = _last_alpha_word(text)
        if text.endswith("."):
            last_key = last.lower()
            if last_key in {"mr", "mrs", "ms", "dr"}:
                continue
            if nxt[0].islower():
                repaired[index] = text[:-1].rstrip()
                continue
            if last_key in _NO_SENTENCE_BREAK_AFTER:
                repaired[index] = text[:-1].rstrip()
                continue
            if _wrapped_proper_name(text, nxt) and not looks_like_signature_line(nxt):
                repaired[index] = text[:-1].rstrip()
                continue
            nxt_word = _last_alpha_word(nxt)
            if (
                last
                and last[0].islower()
                and len(nxt.split()) == 1
                and nxt_word
                and nxt_word[0].isupper()
                and is_english_word(nxt_word)
                and not looks_like_signature_line(nxt)
            ):
                repaired[index] = text[:-1].rstrip()
                repaired[index + 1] = nxt_word.lower() + ("." if nxt.endswith(".") else "")
                continue
        if text[-1].isalnum() and nxt[0].isupper() and len(text.split()) >= 3:
            if looks_like_signature_line(nxt):
                repaired[index] = text + "."
                continue
            if len(nxt.split()) <= 2:
                nxt_word = _last_alpha_word(nxt)
                if (
                    last
                    and last[0].islower()
                    and len(nxt.split()) == 1
                    and nxt_word
                    and nxt_word[0].isupper()
                    and is_english_word(nxt_word)
                ):
                    repaired[index + 1] = nxt_word.lower() + ("." if nxt.endswith(".") else "")
                continue
            if last.lower() in _NO_SENTENCE_BREAK_AFTER:
                first = nxt.split()[0]
                word = re.sub(r"[^A-Za-z']", "", first)
                if word.lower() in {"this", "that", "the", "these", "those", "and", "but"}:
                    repaired[index + 1] = word.lower() + nxt[len(first) :]
                continue
            if _wrapped_proper_name(text, nxt):
                continue
            repaired[index] = text + "."
    if (
        repaired
        and repaired[-1]
        and repaired[-1][-1].isalnum()
        and not looks_like_signature_line(repaired[-1])
    ):
        repaired[-1] = repaired[-1] + "."
    if (
        repaired
        and re.match(r'^[\"“”]', repaired[0].lstrip())
        and not re.search(r'[\"“”]\s*$', repaired[-1])
    ):
        repaired[-1] = repaired[-1].rstrip() + ' "'
    return repaired


def _alpha_tokens(text: str) -> list[str]:
    return [t for t in tokenize_english(text) if t.isalpha() or "'" in t]


def _is_subsequence(needles: list[str], haystack: list[str]) -> bool:
    index = 0
    for token in haystack:
        if index < len(needles) and token == needles[index]:
            index += 1
    return index == len(needles)


def _levenshtein(left: str, right: str) -> int:
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
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if lc == rc else 1)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]


def _token_near_miss(left: str, right: str) -> bool:
    """True when two tokens look like the same handwritten word."""
    a, b = left.lower(), right.lower()
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 4 and longer.startswith(shorter):
        return abs(len(a) - len(b)) <= 4
    if abs(len(a) - len(b)) > 3:
        return False
    dist = _levenshtein(a, b)
    if dist >= min(len(a), len(b)):
        return False
    return dist <= max(1, int(0.5 * max(len(a), len(b))))


def _equal_length_near_miss(trocr_tokens: list[str], vlm_tokens: list[str]) -> bool:
    if len(trocr_tokens) != len(vlm_tokens) or not trocr_tokens:
        return False
    pairs = list(zip(trocr_tokens, vlm_tokens))
    if all(a.lower() == b.lower() for a, b in pairs):
        return False
    return all(_token_near_miss(a, b) for a, b in pairs)


def _prefer_visual_token(trocr_token: str, vlm_token: str) -> str:
    """Take the VLM spelling unless that would replace real English or a name with junk."""
    if is_english_word(trocr_token) and not is_english_word(vlm_token):
        return trocr_token
    if is_english_word(trocr_token) and is_english_word(vlm_token):
        left, right = trocr_token.lower(), vlm_token.lower()
        if left != right and (right.startswith(left) or left.startswith(right)):
            return trocr_token
        if left.endswith("ing") and right.endswith("ing") and left != right:
            return trocr_token
    if is_name_or_title(trocr_token):
        if is_initial(trocr_token) and not is_initial(vlm_token):
            return trocr_token
        if not is_name_or_title(vlm_token):
            return trocr_token
        if is_name_or_title(vlm_token) and trocr_token.lower() != vlm_token.lower():
            return trocr_token
    return vlm_token


def _merge_near_miss_line(trocr: str, vlm: str, trocr_tokens: list[str], vlm_tokens: list[str]) -> str:
    merged = [_prefer_visual_token(a, b) for a, b in zip(trocr_tokens, vlm_tokens)]
    merged_l = [token.lower() for token in merged]
    if merged_l == [token.lower() for token in vlm_tokens]:
        return vlm
    if merged_l == [token.lower() for token in trocr_tokens]:
        return trocr
    return normalize_line_text(" ".join(merged))


def _splice_near_miss(trocr_tokens: list[str], vlm_tokens: list[str]) -> list[str] | None:
    """Apply VLM spellings onto a one-token-longer TrOCR line."""
    if len(trocr_tokens) != len(vlm_tokens) + 1 or not vlm_tokens:
        return None
    for skip in range(len(trocr_tokens)):
        aligned = trocr_tokens[:skip] + trocr_tokens[skip + 1 :]
        if not all(_token_near_miss(a, b) for a, b in zip(aligned, vlm_tokens)):
            continue
        if all(a.lower() == b.lower() for a, b in zip(aligned, vlm_tokens)):
            continue
        merged: list[str] = []
        v_index = 0
        for index, token in enumerate(trocr_tokens):
            if index == skip:
                merged.append(token)
                continue
            merged.append(_prefer_visual_token(aligned[v_index], vlm_tokens[v_index]))
            v_index += 1
        return merged
    return None


def _vlm_closed_class_over_hyphen_break(
    t_toks: list[str], v_toks: list[str]
) -> bool:
    """True when the VLM turned a line-wrap fragment (af-) into a preposition (of)."""
    if len(t_toks) != len(v_toks) or len(t_toks) < 2:
        return False
    if [token.lower() for token in t_toks[:-1]] != [token.lower() for token in v_toks[:-1]]:
        return False
    left, right = t_toks[-1], v_toks[-1]
    if left.lower() == right.lower():
        return False
    return (
        1 <= len(left) <= 3
        and left.isalpha()
        and not is_english_word(left)
        and right.lower() in {"a", "an", "at", "in", "of", "on", "or", "to"}
    )


def _vlm_swapped_retirement_ing(t_toks: list[str], v_toks: list[str]) -> bool:
    """True when the VLM turned 'nearing … retirement' into 'wearing'."""
    joined = " ".join(t_toks + v_toks).lower()
    if "retire" not in joined:
        return False
    return any(
        a.lower() == "nearing" and b.lower() == "wearing"
        for a, b in zip(t_toks, v_toks)
    )


def _vlm_completed_last_word(t_toks: list[str], v_toks: list[str]) -> bool:
    """True when the VLM only lengthened the last token (appointme → appointment)."""
    if len(t_toks) != len(v_toks) or not t_toks:
        return False
    if [token.lower() for token in t_toks[:-1]] != [token.lower() for token in v_toks[:-1]]:
        return False
    last_t = t_toks[-1].lower()
    last_v = v_toks[-1].lower()
    return last_v.startswith(last_t) and len(last_v) > len(last_t)


def fuse_line(trocr_text: str, vlm_text: str) -> str:
    """Keep TrOCR unless the VLM recovers a short missing span or a known first-word miss."""
    trocr = normalize_line_text(trocr_text or "")
    vlm = _first_line(vlm_text)
    if not vlm:
        return trocr
    if not trocr:
        return vlm
    t_toks = _alpha_tokens(trocr)
    v_toks = _alpha_tokens(vlm)
    if not t_toks:
        return vlm
    if not v_toks:
        return trocr
    if _vlm_completed_last_word(t_toks, v_toks):
        return trocr
    if _vlm_swapped_retirement_ing(t_toks, v_toks):
        return trocr
    if _vlm_closed_class_over_hyphen_break(t_toks, v_toks):
        return trocr
    if max(len(t_toks), len(v_toks)) <= 3:
        if t_toks[0].lower() == v_toks[0].lower() and trocr != vlm:
            if len(t_toks) == len(v_toks):
                return _merge_near_miss_line(trocr, vlm, t_toks, v_toks)
            return vlm
        t_caps = sum(1 for t in t_toks if len(t) >= 3 and t[0].isupper())
        v_caps = sum(1 for t in v_toks if len(t) >= 3 and t[0].isupper())
        return trocr if t_caps >= v_caps else vlm
    t_lower = [t.lower() for t in t_toks]
    v_lower = [t.lower() for t in v_toks]
    extra = len(v_toks) - len(t_toks)
    if _equal_length_near_miss(t_toks, v_toks):
        return _merge_near_miss_line(trocr, vlm, t_toks, v_toks)
    spliced = _splice_near_miss(t_toks, v_toks)
    if spliced:
        return normalize_line_text(" ".join(spliced))
    if len(v_toks) + 2 < len(t_toks) and _is_subsequence(v_lower, t_lower):
        return trocr
    if extra <= 4 and extra > 0 and _is_subsequence(t_lower, v_lower):
        extras = [token for token in v_toks if token.lower() not in t_lower]
        if t_toks and is_title_token(t_toks[-1]) and extras and all(
            is_name_or_title(token) for token in extras
        ):
            return trocr
        return vlm
    if t_lower[0] == v_lower[0] and -1 <= extra <= 4:
        overlap = len(set(t_lower) & set(v_lower)) / max(len(set(t_lower)), 1)
        if overlap >= 0.5 and score_english_hypothesis(vlm) > score_english_hypothesis(trocr) + 0.3:
            kept_name = any(
                is_name_or_title(a)
                and a.lower() not in v_lower
                and not any(_token_near_miss(a, b) for b in v_toks)
                for a in t_toks
            )
            name_swaps = [
                (a, b)
                for a, b in zip(t_toks, v_toks)
                if a.lower() != b.lower() and _prefer_visual_token(a, b) == a
            ]
            if kept_name or (name_swaps and len(t_toks) == len(v_toks)):
                return trocr
            return vlm
    if abs(len(t_toks) - len(v_toks)) <= 2 and t_toks[0].lower() != v_toks[0].lower():
        trocr_first = t_toks[0].lower()
        rest_t = t_lower[1:]
        rest_v = v_lower[1:]
        overlap = len(set(rest_t) & set(rest_v))
        rest_match = (not rest_t) or overlap / max(1, len(rest_t)) >= 0.6
        if rest_match and trocr_first in CONFUSABLE_FIRST_WORDS:
            return vlm
        return trocr
    return trocr
