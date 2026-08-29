"""
pipeline/rescorer/beam_rescorer.py
Multi-Objective Autoregressive Beam Rescorer for Handwriting Recognition.

Combines:
1. OCR Optical Log Probabilities: S_OCR(Y)
2. Pharmaceutical Lexicon Prior & Fuzzy Search: S_Lexicon(Y)
3. Clinical Context & Dosage/Route Agreement: S_Context(Y)
4. Visual Handwriting Confusion Penalty: ConfusionPenalty(Y)

Scoring Formula:
S_final(Y) = S_OCR(Y) + lambda_1 * S_Lexicon(Y) + lambda_2 * S_Context(Y) - lambda_3 * ConfusionPenalty(Y)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipeline.rescorer.trie import FuzzyMatch, PrefixTrie, TrieMatch
from pipeline.rescorer.confusion_matrix import VisualConfusionMatrix

logger = logging.getLogger("handwriting.rescorer.beam_rescorer")


# ---------------------------------------------------------------------------
# Pydantic Schemas (PROJECT.md & Interface Contracts)
# ---------------------------------------------------------------------------

class BeamCandidate(BaseModel):
    """Represents an autoregressive beam search hypothesis with OCR log probability."""
    text: str = Field(..., description="Hypothesis transcription text")
    log_prob: float = Field(..., description="OCR acoustic/decoder log probability")

    model_config = ConfigDict(extra="ignore")

    @field_validator("log_prob")
    @classmethod
    def check_finite(cls, v: float) -> float:
        if math.isnan(v):
            return -1000.0
        return v


class ContextFeatures(BaseModel):
    """Clinical context features for prescription validation and disambiguation."""
    dosage: Optional[str] = Field(default=None, description="Dosage strength, e.g. '500mg'")
    route: Optional[str] = Field(default=None, description="Administration route, e.g. 'PO'")
    frequency: Optional[str] = Field(default=None, description="Frequency / Sig code, e.g. 'TID'")
    form: Optional[str] = Field(default=None, description="Dosage form, e.g. 'capsule', 'tablet'")
    raw_context: Optional[Dict[str, Any]] = Field(default=None, description="Raw metadata dictionary")

    model_config = ConfigDict(extra="ignore")


class RescorerResult(BaseModel):
    """Complete structured output from the multi-objective beam rescorer."""
    rescored_text: str = Field(..., description="Top re-ranked candidate text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Normalized rescorer confidence score")
    original_top_beam: str = Field(..., description="Raw top candidate prior to rescoring")
    delta_score: float = Field(..., description="Score improvement over original top candidate")
    rescore_applied: bool = Field(..., description="True if ranking changed or text was modified")
    matched_lexicon_term: Optional[str] = Field(default=None, description="Matched canonical RxNorm or Sig term")
    all_candidates: Optional[List[Dict[str, Any]]] = Field(default=None, description="Detailed breakdown per candidate")

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Clinical Context Regex Extractors & Normalizers
# ---------------------------------------------------------------------------

_DOSAGE_REGEX = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|ml|meq|units|iu|%)\b",
    re.IGNORECASE,
)
_ROUTE_REGEX = re.compile(
    r"\b(PO|IV|IM|SL|PR|TOP|SC|SubQ|Ophthalmic|Otic|Inhalation|transdermal|oral|vaginal)\b",
    re.IGNORECASE,
)
_SIG_REGEX = re.compile(
    r"\b(QD|BID|TID|QID|QHS|PRN|Q4-6H|Q4H|Q6H|Q8H|Q12H|Q24H|STAT|Daily|Every\s+day|WF|AC|PC)\b",
    re.IGNORECASE,
)
_FORM_REGEX = re.compile(
    r"\b(tab(?:let)?s?|cap(?:sule)?s?|sol(?:ution)?|susp(?:ension)?|syrup|ointment|cream|patch(?:es)?|drops?|inhaler|inj(?:ection)?s?)\b",
    re.IGNORECASE,
)

# Route and form compatibility rules
_INCOMPATIBLE_ROUTE_FORMS = [
    ("iv", "tablet"),
    ("iv", "capsule"),
    ("iv", "tab"),
    ("iv", "cap"),
    ("im", "tablet"),
    ("im", "capsule"),
    ("po", "injection"),
    ("po", "inj"),
    ("topical", "capsule"),
]


class BeamRescorer:
    """
    Multi-Objective Autoregressive Beam Rescorer.
    
    Re-ranks top-K OCR beam candidates by balancing:
    1. OCR log-likelihood (S_OCR)
    2. Pharmaceutical vocabulary agreement from RxNorm Trie (S_Lexicon)
    3. Clinical context dosage, route, and frequency compatibility (S_Context)
    4. Visual handwriting confusion cost penalty (ConfusionPenalty)
    """

    def __init__(
        self,
        trie: Optional[PrefixTrie] = None,
        confusion_matrix: Optional[VisualConfusionMatrix] = None,
        lambda_lexicon: float = 1.0,
        lambda_context: float = 0.8,
        lambda_confusion: float = 0.5,
        vocab_dir: Optional[Union[str, Path]] = None,
        max_safe_mg: float = 4000.0,
    ) -> None:
        self.lambda_lexicon = lambda_lexicon
        self.lambda_context = lambda_context
        self.lambda_confusion = lambda_confusion
        self.max_safe_mg = max_safe_mg

        # Initialize Trie
        if trie is not None:
            self.trie = trie
        else:
            self.trie = PrefixTrie()
            vdir = Path(vocab_dir) if vocab_dir else Path("data/reference_handwriting/vocabularies")
            if vdir.exists():
                self.trie.load_vocabularies(vdir)

        # Initialize Confusion Matrix
        if confusion_matrix is not None:
            self.confusion_matrix = confusion_matrix
        else:
            self.confusion_matrix = VisualConfusionMatrix(load_defaults=True)

        # Clinical Knowledge Base for LASA Disambiguation
        self._medications_db: Dict[str, Dict[str, Any]] = {}
        self._brand_to_generic: Dict[str, str] = {}
        self._latin_sigs_db: Dict[str, Dict[str, Any]] = {}
        self._fuzzy_cache: Dict[str, List[FuzzyMatch]] = {}
        self._build_knowledge_base(vocab_dir)

    def _build_knowledge_base(self, vocab_dir: Optional[Union[str, Path]]) -> None:
        """Cache structured medication and sig details for rapid context scoring."""
        vdir = Path(vocab_dir) if vocab_dir else Path("data/reference_handwriting/vocabularies")
        rx_path = vdir / "rxnorm_medications.json"
        if rx_path.exists():
            try:
                with open(rx_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for med in data.get("medications", []):
                    gname = med.get("generic_name", "").strip()
                    if gname:
                        key = gname.lower()
                        self._medications_db[key] = med
                        for brand in med.get("brand_names", []):
                            bkey = brand.strip().lower()
                            self._medications_db[bkey] = med
                            self._brand_to_generic[bkey] = gname
            except Exception as e:
                logger.warning(f"Failed to cache RxNorm medications DB: {e}")

        sig_path = vdir / "latin_sig_codes.json"
        if sig_path.exists():
            try:
                with open(sig_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for sig in data.get("codes", []):
                    code = sig.get("code", "").strip()
                    if code:
                        self._latin_sigs_db[code.lower()] = sig
            except Exception as e:
                logger.warning(f"Failed to cache Latin Sigs DB: {e}")

    def _extract_context_from_text(self, text: str) -> ContextFeatures:
        """Extract dosage, route, frequency, and form tokens from raw text string."""
        dosage = None
        m_dose = _DOSAGE_REGEX.search(text)
        if m_dose:
            dosage = f"{m_dose.group(1)}{m_dose.group(2).lower()}"

        route = None
        m_route = _ROUTE_REGEX.search(text)
        if m_route:
            route = m_route.group(1).upper()

        freq = None
        m_freq = _SIG_REGEX.search(text)
        if m_freq:
            freq = m_freq.group(1).upper()

        form = None
        m_form = _FORM_REGEX.search(text)
        if m_form:
            form = m_form.group(1).lower()

        return ContextFeatures(
            dosage=dosage,
            route=route,
            frequency=freq,
            form=form,
        )

    def _merge_context(
        self,
        text_ctx: ContextFeatures,
        explicit_ctx: Optional[Union[Dict[str, Any], ContextFeatures]],
    ) -> ContextFeatures:
        """Merge text-extracted context with explicit caller-provided context."""
        if explicit_ctx is None:
            return text_ctx

        if isinstance(explicit_ctx, ContextFeatures):
            exp_dict = explicit_ctx.model_dump()
        else:
            exp_dict = explicit_ctx

        dosage = exp_dict.get("dosage") or text_ctx.dosage
        route = exp_dict.get("route") or text_ctx.route
        freq = exp_dict.get("frequency") or text_ctx.frequency
        form = exp_dict.get("form") or text_ctx.form

        # Normalize dosage if given as dict or string
        if dosage and isinstance(dosage, str):
            dosage = dosage.strip().lower()
        if route and isinstance(route, str):
            route = route.strip().upper()
        if freq and isinstance(freq, str):
            freq = freq.strip().upper()
        if form and isinstance(form, str):
            form = form.strip().lower()

        return ContextFeatures(
            dosage=dosage,
            route=route,
            frequency=freq,
            form=form,
            raw_context=exp_dict,
        )

    def _get_fuzzy_matches(self, clean_tok: str) -> List[FuzzyMatch]:
        """Cached fuzzy search lookup."""
        tok_low = clean_tok.lower()
        if tok_low in self._fuzzy_cache:
            return self._fuzzy_cache[tok_low]

        if len(self._fuzzy_cache) > 20000:
            self._fuzzy_cache.clear()

        matches = self.trie.fuzzy_search(clean_tok, max_distance=2, max_results=3)
        self._fuzzy_cache[tok_low] = matches
        return matches

    def _score_lexicon(self, text: str) -> Tuple[float, Optional[str], Optional[Dict[str, Any]]]:
        """
        Compute pharmaceutical lexicon prior score S_Lexicon and find matched drug entity.
        Returns: (lexicon_score, matched_term, matched_metadata)
        """
        if not text or not text.strip():
            return 0.0, None, None

        tokens = text.strip().split()
        if not tokens:
            return 0.0, None, None

        best_score = 0.0
        best_term: Optional[str] = None
        best_meta: Optional[Dict[str, Any]] = None

        # 1. Try full phrase or multi-word prefixes (e.g. "Metoprolol succinate", "Bempedoic acid")
        for n_words in range(min(4, len(tokens)), 0, -1):
            phrase = " ".join(tokens[:n_words])
            exact_match = self.trie.search_exact(phrase)
            if exact_match:
                score = 2.5 if n_words > 1 else (2.0 if exact_match.weight >= 2.0 else 1.5)
                if score > best_score:
                    best_score = score
                    best_term = exact_match.word
                    best_meta = exact_match.metadata
                break

        # 2. Token-by-token exact and fuzzy lookup
        token_lex_sum = 0.0
        for idx, token in enumerate(tokens):
            clean_tok = re.sub(r"[^\w\-]", "", token)
            if not clean_tok or len(clean_tok) < 2:
                continue

            exact = self.trie.search_exact(clean_tok)
            if exact:
                weight = exact.weight
                tok_score = 2.0 if weight >= 2.0 else (1.5 if weight >= 1.5 else 1.0)
                token_lex_sum += tok_score
                if best_term is None or tok_score > best_score:
                    best_score = max(best_score, tok_score)
                    best_term = exact.word
                    best_meta = exact.metadata
            elif idx < 3 or (len(clean_tok) >= 4 and clean_tok[0].isupper()):
                # Fuzzy search in Trie (D <= 2) for candidate medication/sig positions
                fmatches = self._get_fuzzy_matches(clean_tok)
                if fmatches:
                    top_f = fmatches[0]
                    tok_score = 1.5 * top_f.similarity
                    token_lex_sum += tok_score
                    if best_term is None:
                        best_term = top_f.word
                        best_meta = top_f.metadata
                        best_score = max(best_score, tok_score)

        final_lex_score = max(best_score, min(token_lex_sum, 3.5))
        return final_lex_score, best_term, best_meta

    def _score_context(
        self,
        candidate_text: str,
        matched_term: Optional[str],
        context: ContextFeatures,
    ) -> float:
        """
        Compute clinical context compatibility score S_Context.
        Validates dosage strength, route, and frequency against matched medication metadata.
        """
        score = 0.0

        # Unrealistic dosage penalty
        if context.dosage:
            m_val = re.match(r"^(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g)?", context.dosage.lower())
            if m_val:
                val = float(m_val.group(1))
                unit = m_val.group(2) or "mg"
                if unit == "g":
                    val_mg = val * 1000.0
                elif unit in ("mcg", "ug"):
                    val_mg = val / 1000.0
                else:
                    val_mg = val

                if val_mg > 50000.0 or val_mg > (self.max_safe_mg * 5.0):
                    score -= 1.0

        # Conflicting route and form penalty
        if context.route and context.form:
            r_low = context.route.lower()
            f_low = context.form.lower()
            for bad_r, bad_f in _INCOMPATIBLE_ROUTE_FORMS:
                if bad_r in r_low and bad_f in f_low:
                    score -= 1.0
                    break

        # Check against medication knowledge base
        med_entry: Optional[Dict[str, Any]] = None
        if matched_term:
            med_entry = self._medications_db.get(matched_term.lower())
            if med_entry is None:
                # Try generic ref
                gen_name = self._brand_to_generic.get(matched_term.lower())
                if gen_name:
                    med_entry = self._medications_db.get(gen_name.lower())

        if med_entry is not None:
            # 1. Dosage strength validation
            if context.dosage:
                std_strengths = [s.lower().replace(" ", "") for s in med_entry.get("standard_strengths", [])]
                norm_dose = context.dosage.lower().replace(" ", "")
                dose_matched = False
                if norm_dose in std_strengths:
                    dose_matched = True
                else:
                    escaped_dose = re.escape(norm_dose)
                    boundary_pattern = re.compile(
                        rf"(?<![0-9a-zA-Z\.]){escaped_dose}(?![0-9a-zA-Z])",
                        re.IGNORECASE,
                    )
                    dose_matched = any(bool(boundary_pattern.search(s)) for s in std_strengths)

                if dose_matched:
                    score += 1.0
                else:
                    # Mismatch with standard strength catalog
                    score -= 0.4

            # 2. Route validation
            if context.route:
                std_routes = [r.upper() for r in med_entry.get("standard_routes", [])]
                if context.route.upper() in std_routes:
                    score += 0.5
                else:
                    score -= 0.3

            # 3. Frequency validation
            if context.frequency:
                std_freqs = [f.upper() for f in med_entry.get("standard_frequencies", [])]
                if context.frequency.upper() in std_freqs:
                    score += 0.5

        # Check Latin Sig validity if present
        if context.frequency:
            sig_entry = self._latin_sigs_db.get(context.frequency.lower())
            if sig_entry is not None:
                score += 0.3

        return score

    def _compute_confusion_penalty(
        self,
        candidate_text: str,
        original_top_beam: str,
    ) -> float:
        """
        Compute optical handwriting visual confusion penalty between original OCR top beam
        and the candidate text. Normalized to [0.0, 1.20].
        """
        if candidate_text == original_top_beam:
            return 0.0
        return self.confusion_matrix.compute_distance(original_top_beam, candidate_text, normalize=True)

    def rescore_detailed(
        self,
        hypotheses: Sequence[Union[Tuple[str, float], BeamCandidate, Dict[str, Any]]],
        context: Optional[Union[Dict[str, Any], ContextFeatures]] = None,
    ) -> RescorerResult:
        """
        Full detailed multi-objective beam rescoring execution.
        Returns complete RescorerResult dataclass.
        """
        if not hypotheses:
            return RescorerResult(
                rescored_text="",
                confidence=0.0,
                original_top_beam="",
                delta_score=0.0,
                rescore_applied=False,
            )

        # Normalize input hypotheses
        parsed_cands: List[BeamCandidate] = []
        for h in hypotheses:
            if isinstance(h, BeamCandidate):
                parsed_cands.append(h)
            elif isinstance(h, tuple) and len(h) >= 2:
                parsed_cands.append(BeamCandidate(text=str(h[0]), log_prob=float(h[1])))
            elif isinstance(h, dict):
                parsed_cands.append(
                    BeamCandidate(
                        text=str(h.get("text", "")),
                        log_prob=float(h.get("log_prob", h.get("confidence", -0.5))),
                    )
                )

        if not parsed_cands:
            return RescorerResult(
                rescored_text="",
                confidence=0.0,
                original_top_beam="",
                delta_score=0.0,
                rescore_applied=False,
            )

        # Sort by initial OCR log_prob to establish baseline top candidate
        sorted_initial = sorted(parsed_cands, key=lambda c: c.log_prob, reverse=True)
        original_top_beam = sorted_initial[0].text
        original_top_logprob = sorted_initial[0].log_prob

        # Fast path for K=1
        if len(parsed_cands) == 1:
            top_cand = parsed_cands[0]
            text_ctx = self._extract_context_from_text(top_cand.text)
            full_ctx = self._merge_context(text_ctx, context)
            s_lex, matched_term, _ = self._score_lexicon(top_cand.text)
            s_ctx = self._score_context(top_cand.text, matched_term, full_ctx)
            final_s = top_cand.log_prob + (self.lambda_lexicon * s_lex) + (self.lambda_context * s_ctx)
            conf = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, final_s))))
            return RescorerResult(
                rescored_text=top_cand.text,
                confidence=round(conf, 4),
                original_top_beam=top_cand.text,
                delta_score=0.0,
                rescore_applied=False,
                matched_lexicon_term=matched_term,
            )

        # Multi-candidate scoring
        scored_candidates: List[Dict[str, Any]] = []
        for cand in parsed_cands:
            text_ctx = self._extract_context_from_text(cand.text)
            full_ctx = self._merge_context(text_ctx, context)

            s_ocr = cand.log_prob
            s_lex, matched_term, _ = self._score_lexicon(cand.text)
            s_ctx = self._score_context(cand.text, matched_term, full_ctx)
            conf_penalty = self._compute_confusion_penalty(cand.text, original_top_beam)

            final_score = (
                s_ocr
                + (self.lambda_lexicon * s_lex)
                + (self.lambda_context * s_ctx)
                - (self.lambda_confusion * conf_penalty)
            )

            scored_candidates.append({
                "text": cand.text,
                "s_ocr": s_ocr,
                "s_lex": s_lex,
                "s_ctx": s_ctx,
                "conf_penalty": conf_penalty,
                "final_score": final_score,
                "matched_term": matched_term,
            })

        # Rank candidates by final_score descending
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        top_rescored = scored_candidates[0]

        # Calculate softmax confidence over final scores
        max_s = top_rescored["final_score"]
        exps = [math.exp(max(-20.0, min(20.0, c["final_score"] - max_s))) for c in scored_candidates]
        sum_exps = sum(exps)
        conf = (exps[0] / sum_exps) if sum_exps > 0 else 0.5
        conf = max(0.01, min(0.99, conf))

        # Compute delta score against initial baseline
        baseline_final_score = next(
            (c["final_score"] for c in scored_candidates if c["text"] == original_top_beam),
            scored_candidates[0]["final_score"],
        )
        delta = top_rescored["final_score"] - baseline_final_score
        applied = (top_rescored["text"] != original_top_beam) or (delta > 0.01)

        return RescorerResult(
            rescored_text=top_rescored["text"],
            confidence=round(conf, 4),
            original_top_beam=original_top_beam,
            delta_score=round(delta, 4),
            rescore_applied=applied,
            matched_lexicon_term=top_rescored["matched_term"],
            all_candidates=scored_candidates,
        )

    def rescore(
        self,
        hypotheses: Sequence[Union[Tuple[str, float], BeamCandidate, Dict[str, Any]]],
        context: Optional[Union[Dict[str, Any], ContextFeatures]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Re-rank beam hypotheses. Returns list of (text, final_score) tuples sorted descending.
        """
        res = self.rescore_detailed(hypotheses, context=context)
        if not res.all_candidates:
            return [(h.text, h.log_prob) if isinstance(h, BeamCandidate) else (str(h[0]), float(h[1])) for h in hypotheses]
        return [(c["text"], c["final_score"]) for c in res.all_candidates]

    def rescore_top1(
        self,
        hypotheses: Sequence[Union[Tuple[str, float], BeamCandidate, Dict[str, Any]]],
        context: Optional[Union[Dict[str, Any], ContextFeatures]] = None,
    ) -> Tuple[str, float]:
        """
        Return the top re-ranked hypothesis tuple: (rescored_text, confidence).
        """
        res = self.rescore_detailed(hypotheses, context=context)
        return res.rescored_text, res.confidence
