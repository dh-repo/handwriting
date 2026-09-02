from pipeline.training.vlm_refine import (
    _first_line,
    azure_openai_configured,
    filter_page_lines,
    fuse_line,
    is_page_chrome_line,
    is_page_crumb_line,
    is_unrelated_page_tail,
    looks_like_signature_line,
    normalize_line_text,
    repair_line_continuations,
    should_refine_with_vlm,
    vlm_refine_available,
)


def test_fuse_prefers_vlm_pen_over_trocr_given():
    assert fuse_line("given for this style but still", "pen for this style but still") == (
        "pen for this style but still"
    )


def test_fuse_rejects_vlm_that_swallowed_the_next_line():
    trocr = "I don't really have the right"
    vlm = "I don't really have the right you for this stuff, but still enjoy trying"
    assert fuse_line(trocr, vlm) == trocr


def test_fuse_recovers_dropped_middle_word():
    assert fuse_line("I don't really the right", "I don't really have the right") == (
        "I don't really have the right"
    )


def test_fuse_recovers_right_side_of_line():
    assert fuse_line("enjoy trying.", "enjoy trying. Very slowly!") == (
        "enjoy trying. Very slowly!"
    )


def test_fuse_keeps_nearing_before_retirement():
    assert "nearing" in fuse_line(
        "of the problems of men and women nearing",
        "of the problems of men and women wearing or already in retirement",
    )
    assert "nearing" in fuse_line(
        "of the problems of men and women nearing",
        "of the problems of men and women wearing",
    )


def test_fuse_keeps_hyphen_break_over_preposition():
    assert fuse_line("Only a few hours af-", "only a few hours of") == (
        "Only a few hours af-"
    )


def test_fuse_does_not_complete_cropped_last_word():
    assert fuse_line(
        "Review results in follow-up appointme",
        "Review results in follow-up appointment",
    ).endswith("appointme")


def test_fuse_takes_clearly_better_english_when_first_word_matches():
    assert fuse_line("much gricker, but but", "much quicker, but lacks any line.") == (
        "much quicker, but lacks any line."
    )
    assert fuse_line(
        "time on it because the the",
        "time on it because the pen",
    ) == "time on it because the pen"


def test_fuse_keeps_trocr_when_first_word_matches():
    assert fuse_line(
        "variation . The faster I try to write ...",
        "variation. The faster I try it would",
    ) == "variation. The faster I try to write."


def test_fuse_keeps_signature_over_one_word_garbage():
    assert fuse_line("Dick N.", "Dieh") == "Dick N."


def test_fuse_trusts_vlm_for_same_name_initial():
    assert fuse_line("Dick N.", "Dick D.") == "Dick D."


def test_repair_drops_period_before_lowercase_continuation():
    lines = repair_line_continuations(
        [
            "out without spending too much.",
            "time on it because the pen",
            "the worse it becomes.",
            "Italic is still my favourite to",
        ]
    )
    assert lines[0] == "out without spending too much"
    assert lines[2] == "the worse it becomes."


def test_fuse_rejects_vlm_prefix_truncation():
    trocr = "I don't think he will storm the charts with this one, but it's a good start."
    assert fuse_line(trocr, "I don't think") == trocr


def test_fuse_joins_wrapped_vlm_continuation():
    trocr = "I don't think he will storm the charts with this one, but it's"
    vlm = "I don't think he will storm the charts with this one, but it's\na good start."
    assert "good start" in fuse_line(trocr, vlm)


def test_first_line_joins_lowercase_wrap_only():
    assert "good start" in _first_line(
        "I don't think he will storm the charts\nwith this one, but it's a good start."
    )
    assert _first_line("the worse it becomes\nItalic is still my favourite to") == (
        "the worse it becomes"
    )


def test_repair_does_not_break_after_led():
    lines = repair_line_continuations(
        [
            "Mr. Brown, passionate and warm-hearted, led",
            "Labour's attack on the higher health charges.",
        ]
    )
    assert not lines[0].endswith(".")
    assert lines[1].startswith("Labour")


def test_repair_does_not_break_after_bear():
    lines = repair_line_continuations(
        [
            "rose to say that the Chancellor would bear",
            "Mr. Wilson's offer in mind.",
        ]
    )
    assert not lines[0].endswith(".")
    assert lines[1].startswith("Mr.")


def test_repair_lowercases_hyphen_wrap_that_makes_an_english_word():
    lines = repair_line_continuations(
        [
            "the Foreign Minister of Indo-",
            "Nesia arrived in Belgrade as the guest of the Yugoslav",
        ]
    )
    assert lines[1].startswith("nesia")


def test_repair_does_not_break_after_asking():
    lines = repair_line_continuations(
        [
            "Boun Oum was considering asking",
            "Prince Souvanna Phouma to join his Government.",
        ]
    )
    assert not lines[0].endswith(".")
    assert lines[1].startswith("Prince")


def test_repair_closes_opening_quote():
    lines = repair_line_continuations(
        [
            '" That cannot continue without either',
            "development being limited or an",
            "adjustment being made in financing.",
        ]
    )
    assert lines[-1].endswith('"')


def test_repair_adds_period_before_new_capital_sentence():
    lines = repair_line_continuations(
        [
            "the worse it becomes",
            "Italic is still my favourite to",
        ]
    )
    assert lines[0] == "the worse it becomes."


def test_repair_adds_period_on_final_unfinished_line():
    lines = repair_line_continuations(
        [
            "Germany, Mr. Adenauer is in a tough",
            "spot",
        ]
    )
    assert lines[-1] == "spot."
    lines = repair_line_continuations(
        [
            "the London talks on the Protectorate's",
            "future",
        ]
    )
    assert lines[-1] == "future."
    lines = repair_line_continuations(
        [
            "does a lot of the work for me",
            "Dick D.",
        ]
    )
    assert lines[0] == "does a lot of the work for me."


def test_repair_does_not_break_wrapped_proper_names():
    lines = repair_line_continuations(
        [
            "And, since this is election year in West",
            "Germany, Dr. Adenauer is in a tough spot.",
        ]
    )
    assert lines[0] == "And, since this is election year in West"
    assert "West. Germany" not in " ".join(lines)


def test_repair_drops_false_period_on_wrapped_names():
    lines = repair_line_continuations(
        [
            "and he is to be backed by Mr. Will.",
            "Griffiths, MP for Manchester Exchange.",
            "getting an elected majority in Northern.",
            "Rhodesia, but the Colonial Secretary, Mr. Iain.",
            "Macleod, is insistent on a policy of change.",
            "The Senate Banking.",
            "Committee, which is headed by another.",
            "Southern Senator - William Robertson, of",
        ]
    )
    joined = " ".join(lines)
    assert "Will. Griffiths" not in joined
    assert "Northern. Rhodesia" not in joined
    assert "Iain. Macleod" not in joined
    assert "Banking. Committee" not in joined
    assert "another. Southern" not in joined


def test_repair_rejoins_wrapped_last_english_word():
    lines = repair_line_continuations(
        [
            "Germany, Mr. Adenauer is in a tough.",
            "Spot.",
        ]
    )
    assert lines == ["Germany, Mr. Adenauer is in a tough", "spot."]
    lines = repair_line_continuations(
        [
            "Germany, Mr. Adenauer is in a tough",
            "Spot.",
        ]
    )
    assert lines == ["Germany, Mr. Adenauer is in a tough", "spot."]


def test_spot_is_not_a_signature():
    assert looks_like_signature_line("Dick D.")
    assert not looks_like_signature_line("Spot.")
    assert not looks_like_signature_line("West German Government.")
    assert not looks_like_signature_line("Mr. Her")


def test_normalize_splits_smashed_signature():
    assert normalize_line_text("DICKD.") == "Dick D."
    assert (
        normalize_line_text(
            "because the pen does a lot of the work for me DICKD."
        )
        == "because the pen does a lot of the work for me. Dick D."
    )
    assert normalize_line_text('telephones "') == "telephones"
    assert normalize_line_text("the Protectorate 's") == "the Protectorate's"
    assert normalize_line_text("discuss Weaver's appointment. S") == (
        "discuss Weaver's appointment."
    )
    assert (
        normalize_line_text("Labour's Colonial spokesman, Said Sir Roy had")
        == "Labour's Colonial spokesman, said Sir Roy had"
    )
    assert "with. Mr." not in normalize_line_text(
        "his chief a report on his talks with Mr. Her"
    )
    assert (
        normalize_line_text("Griffiths, MPs for Manchester Exchange.")
        == "Griffiths, MP for Manchester Exchange."
    )
    assert (
        normalize_line_text("The Conference will meet")
        == "The conference will meet"
    )
    assert (
        normalize_line_text("no right to Delay progress")
        == "no right to delay progress"
    )
    assert (
        normalize_line_text("a proposed House of chiefs")
        == "a proposed House of Chiefs"
    )
    assert (
        normalize_line_text("in Northern Rhodesian, but the Colonial Secretary,")
        == "in Northern Rhodesia, but the Colonial Secretary,"
    )
    assert (
        normalize_line_text("African nationalists. Most.")
        == "African nationalists."
    )
    assert (
        normalize_line_text("Chancellor Adenauer Said the Vienna talks")
        == "Chancellor Adenauer said the Vienna talks"
    )
    assert normalize_line_text("in DUESSELDORF, Chancellor") == (
        "In DUESSELDORF, Chancellor"
    )
    assert (
        normalize_line_text("the Yugoslav foreign Minister.")
        == "the Yugoslav Foreign Minister."
    )


def test_repair_capitalizes_wrapped_house_of():
    lines = repair_line_continuations(
        [
            "to discuss the function of a proposed House",
            "of chiefs",
        ]
    )
    assert lines[-1] == "of Chiefs."


def test_normalize_drops_trailing_stray_letter():
    lines = repair_line_continuations(["discusses Weaver's appointment S"])
    assert lines[0] == "discusses Weaver's appointment."
    assert (
        normalize_line_text("careful cursive handwriting practice i")
        == "careful cursive handwriting practice i"
    )


def test_repair_lowercases_wrapped_closed_class():
    lines = repair_line_continuations(
        [
            "States officials quickly pointed out that",
            "This is money due to me and anyway",
        ]
    )
    assert lines[1].startswith("this is money")
    assert "that. This" not in " ".join(lines)


def test_normalize_restores_cropped_mr():
    lines = repair_line_continuations(["r. James Callaghan, Labour's Colonial spokesman,"])
    assert lines[0].startswith("Mr. James Callaghan")


def test_normalize_drops_stray_possessive_s():
    lines = repair_line_continuations(
        [
            "the London talks on the Protectorate's s",
            "future.",
        ]
    )
    assert "Protectorate's s" not in " ".join(lines)
    assert "Protectorate's future" in " ".join(lines)


def test_repair_keeps_title_abbreviation_period():
    lines = repair_line_continuations(
        [
            "his chief a report on his talks with Mr.",
            "Macmillan at Chequers",
        ]
    )
    assert lines[0] == "his chief a report on his talks with Mr."


def test_fuse_keeps_trocr_variation_over_vlm_transpiration():
    assert fuse_line(
        "variation. The faster I try to write.",
        "transpiration. The faster I try to write",
    ) == "variation. The faster I try to write."


def test_fuse_keeps_italic():
    assert fuse_line(
        "Italian is still my favourite to",
        "Italic is still my favourite to",
    ) == "Italic is still my favourite to"


def test_fuse_takes_equal_length_visual_near_miss():
    assert fuse_line(
        "of the Commons and in its own Conversations",
        "of the Commons and in its own Convocations",
    ) == "of the Commons and in its own Convocations"
    assert fuse_line(
        "the foreign mechanics, which whom the King's",
        "the foreign merchants, which whom the king's",
    ) == "the foreign merchants, which whom the king's"
    assert fuse_line(
        "excess of the role for native merchants.",
        "excess of the rate for native merchants.",
    ) == "excess of the rate for native merchants."
    assert fuse_line(
        "printed equivalent contributions, and second,",
        "granted equivalent contributions, and second,",
    ) == "granted equivalent contributions, and second,"


def test_fuse_keeps_sir_over_lets():
    assert fuse_line("Yesterday Sir Roy's", "yesterday let's Roy's") == "Yesterday Sir Roy's"


def test_skip_vlm_when_trocr_is_already_english_or_names():
    assert should_refine_with_vlm("")
    assert should_refine_with_vlm("given for this style but still")
    assert should_refine_with_vlm("this is money due to menca anyway.")
    assert should_refine_with_vlm('chief aide, Mr. Julius Greenfield, telephones "')
    assert not should_refine_with_vlm("Yesterday Sir Roy's")
    assert not should_refine_with_vlm("States officials quickly point out that")
    assert not should_refine_with_vlm("this is money due to Mecca anyway.")


def test_fuse_keeps_point_over_pointed():
    assert fuse_line(
        "States officials quickly point out that",
        "States officials quickly pointed out that",
    ) == "States officials quickly point out that"


def test_fuse_does_not_invent_given_name_after_title():
    assert fuse_line(
        "his chief a report on his talks with Mr.",
        "his chief a report on his talks with Mr. Her",
    ) == "his chief a report on his talks with Mr."


def test_fuse_keeps_america_over_me_and():
    assert "America" in fuse_line(
        "this is money due to America anyway.",
        "this is money due to me and anyway",
    )
    assert "Mecca" in fuse_line(
        "this is money due to Mecca anyway.",
        "this is money due to me and anyway",
    )


def test_fuse_keeps_named_person_over_unrelated_name():
    assert "Kennedy" in fuse_line(
        "President Kennedy's rejection of it is a painful blow to the West German Government.",
        "President University's rejection of it is a painful blow to the West German government.",
    )


def test_fuse_keeps_invocab_trocr_over_oov_vlm_near_miss():
    assert fuse_line(
        "an unending supply of cheap wine",
        "an limending supply of cheap wine",
    ) == "an unending supply of cheap wine"


def test_fuse_splices_near_miss_and_keeps_extra_trocr_token():
    assert fuse_line(
        "sanction at a rare roughly fifty per cent in",
        "sanction at a rate roughly fifty per cent,",
    ) == "sanction at a rate roughly fifty per cent in"


def test_fuse_keeps_trocr_when_vlm_is_exact_shorter_copy():
    assert fuse_line(
        "I don't think he will storm the charts with this one, but it's a good start.",
        "I don't think",
    ) == "I don't think he will storm the charts with this one, but it's a good start."


def test_fuse_rejects_short_unrelated_first_word():
    assert fuse_line(
        "at \" colleagues of merchants, and whose",
        "of colleagues of merchants\", and whole",
    ) == "at \" colleagues of merchants, and whose"


def test_crumb_drops_hyphen_salad_but_keeps_signature():
    assert is_page_crumb_line("I - I", "of Chiefs.")
    assert not is_page_crumb_line("Spot.", "Mr. Adenauer is in a tough.")
    assert not is_page_crumb_line("Dick D.", "does a lot of the work for me.")
    assert not is_page_crumb_line("Very slowly!", "enjoy trying.")
    assert is_page_crumb_line("n", "the quick brown fox jumps over the U")
    assert is_page_crumb_line("v", "patient reported recurring headaches sir")
    assert not is_page_crumb_line("I", "Careful cursive handwriting practice.")
    assert not is_page_crumb_line("Mr.")
    assert not is_page_crumb_line("on", "federalism in Europe, \" he went")


def test_unrelated_tail_drops_century_union():
    prior = (
        "the quick brown fox jumps over the U careful cursive handwriting "
        "practice. Every stroke and curve conveys person signed with "
        "precision and steady flow."
    )
    assert is_unrelated_page_tail("4th Century Union.", prior)
    assert not is_unrelated_page_tail("Dick D.", prior)
    assert not is_unrelated_page_tail(
        "Every stroke and curve conveys person", prior
    )
    assert not is_unrelated_page_tail(
        "Signed with precision and steady flow.", prior
    )


def test_filter_drops_navigation_and_hash_leftovers():
    assert is_page_chrome_line("Navigation menu")
    assert is_page_chrome_line("Navigation menu.")
    assert is_page_chrome_line("4th Century Union.")
    assert is_page_chrome_line("4th Century.")
    assert is_page_chrome_line("What links here this article has also been.")
    assert is_page_chrome_line("THE I am I am put them.")
    assert is_page_chrome_line("# U.V.O.")
    kept = filter_page_lines(
        [
            "United States officials quickly point out that this is money due to America anyway.",
            "Navigation menu",
            "# U.V.O.",
        ]
    )
    assert kept == [
        "United States officials quickly point out that this is money due to America anyway."
    ]


def test_azure_openai_configured_reads_env(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    assert azure_openai_configured() is False
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    assert azure_openai_configured() is True
    assert vlm_refine_available() is True
