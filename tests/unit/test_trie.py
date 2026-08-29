"""
tests/unit/test_trie.py
Unit tests for PrefixTrie, TrieNode, TrieMatch, and FuzzyMatch.
"""

from __future__ import annotations

import time
from pathlib import Path
import pytest

from pipeline.rescorer.trie import (
    FuzzyMatch,
    MatchType,
    PrefixTrie,
    TrieMatch,
    TrieNode,
)


class TestPrefixTrieBasics:
    """Test fundamental Trie insert, search, and normalization operations."""

    def test_node_slots_and_defaults(self) -> None:
        """Verify TrieNode slots optimization and default values."""
        node = TrieNode()
        assert node.children == {}
        assert node.is_terminal is False
        assert node.canonical_word is None
        assert node.weight == 1.0
        assert node.metadata is None

    def test_insert_and_exact_search(self) -> None:
        """Verify insertion and O(L) exact search."""
        trie = PrefixTrie()
        trie.insert("Amoxicillin", metadata={"rxcui": "7052"}, weight=2.0)
        trie.insert("Ampicillin", metadata={"rxcui": "733"}, weight=2.0)

        assert len(trie) == 2
        assert "Amoxicillin" in trie
        assert "Ampicillin" in trie
        assert "Aspirin" not in trie

        m1 = trie.search_exact("Amoxicillin")
        assert m1 is not None
        assert m1.word == "Amoxicillin"
        assert m1.weight == 2.0
        assert m1.metadata == {"rxcui": "7052"}
        assert m1.match_type == MatchType.EXACT

    def test_case_insensitive_matching(self) -> None:
        """Verify case-insensitive search by default."""
        trie = PrefixTrie(case_sensitive=False)
        trie.insert("Celebrex")

        assert trie.search_exact("celebrex") is not None
        assert trie.search_exact("CELEBREX") is not None
        assert trie.search_exact("Celebrex") is not None
        assert trie.search_exact("celebrex").word == "Celebrex"

    def test_case_sensitive_option(self) -> None:
        """Verify case-sensitive matching when enabled."""
        trie = PrefixTrie(case_sensitive=True)
        trie.insert("TID")

        assert trie.search_exact("TID") is not None
        assert trie.search_exact("tid") is None

    def test_punctuation_stripping(self) -> None:
        """Verify punctuation normalization in keys."""
        trie = PrefixTrie(strip_punctuation=True)
        trie.insert("P.O.")

        assert trie.search_exact("PO") is not None
        assert trie.search_exact("p.o.") is not None
        assert trie.search_exact("P.O.").word == "P.O."

    def test_update_existing_node(self) -> None:
        """Verify duplicate insert updates weight and merges metadata."""
        trie = PrefixTrie()
        trie.insert("Lisinopril", metadata={"source": "rxnorm"}, weight=1.0)
        assert len(trie) == 1

        trie.insert("Lisinopril", metadata={"extra": "data"}, weight=2.5)
        assert len(trie) == 1

        m = trie.search_exact("Lisinopril")
        assert m is not None
        assert m.weight == 2.5
        assert m.metadata == {"source": "rxnorm", "extra": "data"}

    def test_batch_insert(self) -> None:
        """Verify insert_many handles various tuple and string formats."""
        trie = PrefixTrie()
        items = [
            "Aspirin",
            ("Ibuprofen", {"class": "NSAID"}),
            ("Acetaminophen", {"class": "Analgesic"}, 3.0),
        ]
        count = trie.insert_many(items)
        assert count == 3
        assert len(trie) == 3
        assert trie.search_exact("Acetaminophen").weight == 3.0
        assert trie.search_exact("Ibuprofen").metadata == {"class": "NSAID"}


class TestPrefixSearch:
    """Test prefix search and autocompletion ranking."""

    def test_prefix_search_ranking(self) -> None:
        """Verify prefix matching returns all branch matches sorted by weight."""
        trie = PrefixTrie()
        trie.insert("Amoxicillin", weight=3.0)
        trie.insert("Amoxil", weight=2.0)
        trie.insert("Ampicillin", weight=1.0)
        trie.insert("Aspirin", weight=2.5)

        # Prefix "Amox"
        matches = trie.search_prefix("Amox")
        assert len(matches) == 2
        assert matches[0].word == "Amoxicillin"
        assert matches[1].word == "Amoxil"

        # Prefix "Am"
        all_am = trie.search_prefix("Am")
        assert len(all_am) == 3
        assert [m.word for m in all_am] == ["Amoxicillin", "Amoxil", "Ampicillin"]

    def test_prefix_empty_or_missing(self) -> None:
        """Verify prefix search handles empty string and missing prefixes gracefully."""
        trie = PrefixTrie()
        trie.insert("Metformin")

        assert trie.search_prefix("") == []
        assert trie.search_prefix("Zzz") == []

    def test_prefix_max_results(self) -> None:
        """Verify max_results limit."""
        trie = PrefixTrie()
        for i in range(20):
            trie.insert(f"Hydroxyzine_{i:02d}", weight=float(i))

        res = trie.search_prefix("Hydro", max_results=5)
        assert len(res) == 5
        assert res[0].word == "Hydroxyzine_19"


class TestFuzzySearch:
    """Test branch-pruned Levenshtein and Damerau-Levenshtein fuzzy matching."""

    @pytest.fixture
    def populated_trie(self) -> PrefixTrie:
        trie = PrefixTrie()
        words = [
            ("Amoxicillin", 2.0),
            ("Ampicillin", 1.8),
            ("Hydroxyzine", 2.0),
            ("Hydralazine", 2.0),
            ("Celebrex", 2.0),
            ("Celexa", 2.0),
            ("Prednisone", 2.0),
            ("Prednisolone", 2.0),
            ("BID", 3.0),
            ("TID", 3.0),
            ("QID", 3.0),
            ("PO", 3.0),
        ]
        for w, wt in words:
            trie.insert(w, weight=wt)
        return trie

    def test_fuzzy_exact_match(self, populated_trie: PrefixTrie) -> None:
        """Verify distance 0 for exact match in fuzzy search."""
        matches = populated_trie.fuzzy_search("Amoxicillin", max_distance=2)
        assert len(matches) > 0
        assert matches[0].word == "Amoxicillin"
        assert matches[0].distance == 0
        assert matches[0].similarity == 1.0

    def test_fuzzy_single_substitution(self, populated_trie: PrefixTrie) -> None:
        """Verify single character typo resolution (D=1)."""
        matches = populated_trie.fuzzy_search("Amoxcillin", max_distance=2)
        assert len(matches) > 0
        assert matches[0].word == "Amoxicillin"
        assert matches[0].distance == 1

    def test_fuzzy_damerau_transposition(self, populated_trie: PrefixTrie) -> None:
        """Verify adjacent character transposition at distance 1."""
        matches = populated_trie.fuzzy_search("Aomxicillin", max_distance=2, enable_damerau=True)
        assert len(matches) > 0
        assert matches[0].word == "Amoxicillin"
        assert matches[0].distance == 1

    def test_fuzzy_distance_2(self, populated_trie: PrefixTrie) -> None:
        """Verify double error resolution (D=2)."""
        matches = populated_trie.fuzzy_search("Ampycilin", max_distance=2)
        assert len(matches) > 0
        assert matches[0].word == "Ampicillin"
        assert matches[0].distance == 2

    def test_length_adaptive_clamping(self, populated_trie: PrefixTrie) -> None:
        """Verify short tokens (e.g. 'PO', 'B1') are not over-matched at D=2."""
        # 'PO' has length 2; length-adaptive clamps max_distance to 1
        matches = populated_trie.fuzzy_search("PO", max_distance=2, length_adaptive=True)
        # Should not match 3-letter or 4-letter words with distance 2
        for m in matches:
            assert m.distance <= 1

    def test_fuzzy_search_latency(self, populated_trie: PrefixTrie) -> None:
        """Verify fuzzy search execution is well under 1ms per query."""
        t0 = time.perf_counter()
        n_iters = 500
        for _ in range(n_iters):
            _ = populated_trie.fuzzy_search("Amoxcillin", max_distance=2)
        avg_ms = (time.perf_counter() - t0) * 1000.0 / n_iters
        assert avg_ms < 1.0, f"Fuzzy search latency {avg_ms:.4f}ms exceeds 1.0ms SLA"


class TestVocabularyIngestion:
    """Test loading real RxNorm and Latin Sig JSON catalogs."""

    def test_load_vocabularies(self) -> None:
        """Verify loading actual project vocabularies."""
        trie = PrefixTrie()
        stats = trie.load_vocabularies("data/reference_handwriting/vocabularies")
        assert "rxnorm" in stats
        assert "latin_sigs" in stats
        assert stats["rxnorm"] >= 1800
        assert stats["latin_sigs"] == 60
        assert len(trie) >= 1800

        # Verify key clinical entities
        assert "Amoxicillin" in trie
        assert "Amoxil" in trie
        assert "BID" in trie
        assert "PO" in trie
        assert "QD" in trie

        # Stats check
        s = trie.get_stats()
        assert s["total_terms"] == len(trie)
        assert s["total_nodes"] > 5000
        assert s["max_depth"] > 10
        assert s["estimated_memory_kb"] < 5000.0  # < 5MB
