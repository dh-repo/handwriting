"""
Unit and integration tests for Proper Noun and Name Recognition Engine.
Covers:
1. is_initial and looks_like_name detection.
2. Suppression of LM autocorrect on proper nouns via _prefer_visual_token.
3. Grapheme-strict VLM prompting.
4. Customer entity gazetteer trie injection and lookups.
"""

import pytest
from pathlib import Path
from pipeline.training.english_beam import is_initial, looks_like_name, is_name_or_title, is_title_token
from pipeline.training.vlm_refine import _prefer_visual_token, _line_prompt
from pipeline.rescorer.trie import PrefixTrie


def test_initial_and_name_detection():
    # Single-letter initials
    assert is_initial("D.")
    assert is_initial("J.")
    assert is_initial("A")
    assert not is_initial("Dick")
    assert not is_initial("the")

    # Proper names & surnames
    assert looks_like_name("Dick")
    assert looks_like_name("Hasak")
    assert looks_like_name("Smith")
    assert looks_like_name("O'Connor")
    assert looks_like_name("D.")

    # Closed-class English words should not be names
    assert not looks_like_name("the")
    assert not looks_like_name("and")
    assert not looks_like_name("this")

    # Titles & honorifics
    assert is_title_token("Dr.")
    assert is_title_token("Prof.")
    assert is_title_token("Judge")
    assert is_name_or_title("Dr.")
    assert is_name_or_title("Dick")
    assert is_name_or_title("D.")


def test_prefer_visual_token_protects_proper_nouns():
    # 1. Initial must NEVER be expanded into an ordinary word
    assert _prefer_visual_token("D.", "Dickie") == "D."
    assert _prefer_visual_token("J.", "John") == "J."

    # 2. Distinct proper names must not be swapped
    assert _prefer_visual_token("Dick", "Dickie") == "Dick"
    assert _prefer_visual_token("Jon", "John") == "Jon"
    assert _prefer_visual_token("Hasak", "Hassle") == "Hasak"

    # 3. Common first-word confusions CAN still be refined if not a name
    assert _prefer_visual_token("open", "pen") == "pen"


def test_grapheme_strict_vlm_prompting():
    prompt = _line_prompt(hypothesis="Dick D.", previous_text="the work for me.")
    assert "proper nouns" in prompt.lower()
    assert "autocorrect" in prompt.lower()
    assert "Dick D." in prompt


def test_customer_gazetteer_trie_injection(tmp_path: Path):
    trie = PrefixTrie()
    gazetteer_path = Path("data/reference_handwriting/vocabularies/customer_names.json")
    assert gazetteer_path.exists(), "customer_names.json should exist"

    count = trie.load_customer_gazetteer(gazetteer_path)
    assert count >= 50

    # Verify exact lookups of customer entities
    dick_match = trie.search_exact("Dick D.")
    assert dick_match is not None
    assert dick_match.metadata["type"] == "customer_proper_noun"
    assert dick_match.weight >= 4.0

    hasak_match = trie.search_exact("Hasak")
    assert hasak_match is not None
    assert hasak_match.metadata["type"] == "customer_proper_noun"

    # Verify plain text roster loading
    custom_txt = tmp_path / "custom_roster.txt"
    custom_txt.write_text("Damian Hasak\nAlice Wonder\n# comment\nBob\n")
    txt_count = trie.load_customer_gazetteer(custom_txt)
    assert txt_count == 3
    assert trie.search_exact("Damian Hasak") is not None
