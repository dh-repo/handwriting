from pipeline.training.english_beam import (
    choose_english_beam,
    choose_english_beam_or_first,
    collapse_decoder_loop,
    is_english_word,
    pick_crop_hypothesis,
    repair_near_miss_tokens,
    score_english_hypothesis,
)


def test_prefers_enjoy_trying_over_employ_trying():
    picked = choose_english_beam(
        ["employ tying . Very slowly !", "enjoy trying . Very slowly !"]
    )
    assert picked.startswith("enjoy")


def test_does_not_prefer_ellipsis_padding():
    picked = choose_english_beam(
        ["the worse it becomes .", "the worse it becomes ..."]
    )
    assert picked == "the worse it becomes ."


def test_prefers_short_name_over_initials():
    picked = choose_english_beam(["Dick N.", "W. D.", "Dick K."])
    assert picked.startswith("Dick")


def test_prefers_real_english_over_oov():
    picked = choose_english_beam(
        [
            "much gricker , but lacks any line .",
            "much quicker , but lacks any line .",
        ]
    )
    assert picked == "much quicker , but lacks any line ."


def test_prefers_it_becomes_over_is_becomes():
    picked = choose_english_beam(
        ["the worse is becomes .", "the worse it becomes ."]
    )
    assert picked == "the worse it becomes ."


def test_prefers_italic_in_handwriting_context():
    picked = choose_english_beam(
        [
            "Italian is still my favourite to",
            "Italic is still my favourite to",
        ],
        previous_text="My everyday hand is similar and much quicker",
    )
    assert picked.startswith("Italic")
    italia = choose_english_beam(
        [
            "Italian is still my favourite to",
            "Italia is still my favourite to",
        ],
        previous_text="My everyday hand is similar and much quicker",
    )
    assert italia.startswith("Italic")


def test_keeps_first_over_case_only_and_oov_swap():
    assert choose_english_beam(
        [
            "the dishes of the country , and , instead",
            "the dishes of the country , and , Instead",
        ]
    ).endswith("instead")
    assert choose_english_beam(
        [
            "The large attendance and atmosphere of",
            "Sle large attendance and atmosphere of",
        ]
    ).startswith("The")
    assert choose_english_beam(
        [
            "simplicity , though nowadays their",
            "simplicity , though Dowadays their",
        ]
    ).lower().find("nowadays") >= 0


def test_prefers_food_context_beams():
    dishes = choose_english_beam(
        [
            "unending succession of dishes from those",
            "unending procession of dishes from those",
        ]
    )
    assert "procession" in dishes
    gorging = choose_english_beam(
        [
            "of carrying the eye with magnitude ,",
            "of gorging the eye with magnitude ,",
        ],
        previous_text="the dishes of the country",
    )
    assert "gorging" in gorging
    gourmands = choose_english_beam(
        [
            "Epicures and governments , stated by the",
            "Epicures and gourmands , stated by the",
            "Epicures and governments , sated by the",
        ]
    )
    assert "gourmand" in gourmands.lower()
    assert "sated" in gourmands.lower()


def test_accra_hyphen_and_place_overlays():
    assert choose_english_beam(
        ["Only a few hours of-", "Only a few hours af-"]
    ).endswith("af-")
    remainder = choose_english_beam(
        [
            "Ever Mr. Lloyd and his 24-strong delegation handed",
            "ter Mr. Lloyd and his 24-strong delegation handed",
            "Ever Mr. Lloyd and his 24-strong delegation landed",
            "ter Mr. Lloyd and his 24-strong delegation landed",
        ],
        previous_text="Only a few hours af-",
    )
    assert remainder.startswith("ter")
    assert "landed" in remainder
    assert "Accra" in choose_english_beam(
        [
            "at Accern this morning, hundreds of shop assistants",
            "at Accra this morning, hundreds of shop assistants",
        ]
    )
    assert choose_english_beam(
        [
            "Stores, the largest in turn.",
            "Stores, the largest in town.",
        ]
    ).endswith("town.")
    assert choose_english_beam(
        [
            "I've General Council of the Trades Union",
            "The General Council of the Trades Union",
        ]
    ).startswith("The")
    assert "Mrs" in choose_english_beam(
        [
            "given by Lady Dorothy Macmillan with Mrs. Kennedy and other",
            "given by Lady Dorothy Macmillan with Miss Kennedy and other",
        ]
    )
    assert "and Mr" in choose_english_beam(
        [
            "guests present, Mr. Kennedy could Mr. Macmillan met three more",
            "guests present, Mr. Kennedy and Mr. Macmillan met three more",
        ]
    )
    assert choose_english_beam(
        ["No details of", "Ns details of", "Ms details of"]
    ).startswith("No")
    assert choose_english_beam(
        [
            "Prince Souvanna Phouma to join this",
            "Prince Souvanna Phouma to join his",
            "Prince Souvanna Phouma to join this time",
        ]
    ).endswith("his")


def test_picks_iam_sentence_page_37_48_leftovers():
    assert "wailings" in choose_english_beam(
        [
            "We started with the plaintive wailings",
            "We started with the plaintive waiting",
        ]
    )
    assert "I agree" in choose_english_beam(
        [
            '" I agree with the Prime',
            '" Degree with the Prime',
        ]
    )
    assert "8.4" in choose_english_beam(
        [
            "There was a computed 8.4 per cent. swing",
            "There was a computed &4 per cent. swing",
        ]
    )
    assert "8 p.m." in choose_english_beam(
        [
            "Long before polling closed at 8 p.m. it was",
            "Long before polling closed at B p.m. it was",
        ]
    )
    assert choose_english_beam(
        [
            "The trend may to rule out such a tragic",
            "The best way to rule out such a tragic",
            "The heart way to rule out such a tragic",
        ]
    ).startswith("The best way")
    assert "to-nights" in choose_english_beam(
        [
            "early stages of to rights debate.",
            "early stages of to-nights debate.",
        ]
    )
    assert choose_english_beam(
        [
            "They nodded at each other and Sir Edward",
            "There nodded at each other and Sir Edward",
        ]
    ).startswith("They")


def test_keeps_grechko_might_godber_and_commentator():
    assert "Grechko" in choose_english_beam(
        [
            "Marshal Andrei Grechko, commanding the",
            "Marshal Andrei Are oh to, commanding the",
        ]
    )
    assert "might" in choose_english_beam(
        [
            "The whole naval might of the Soviet",
            "The whole naval Wright of the Soviet",
        ]
    )
    assert "Godber" in choose_english_beam(
        [
            "Godever's performance merited all the mild",
            "Godber's performance merited all the mild",
        ]
    )
    naval = choose_english_beam(
        [
            'and proved sight, " said Moscow radio\'s',
            'and proud sight, " said Moscow radio\'s',
        ]
    )
    assert "proud" in naval
    assert choose_english_beam(
        ["communator.", "commentator."]
    ).startswith("commentator")
    assert pick_crop_hypothesis(["pur .", "Mr."], 80, 70).lower().startswith("mr")


def test_keeps_no_over_ns_and_his_over_this_time():
    assert choose_english_beam(
        ["No details of", "Ns details of", "N's details of"]
    ).startswith("No")
    his = choose_english_beam(
        [
            "Prince Souvanna Phouma to join this",
            "Prince Souvanna Phouma to join his",
            "Prince Souvanna Phouma to join this time",
        ]
    )
    assert "his" in his
    assert "time" not in his


def test_keeps_that_over_most_and_bitterest_over_different():
    that = choose_english_beam(
        [
            '" That cannot continue without either',
            '" Most cannot continue without either',
        ]
    )
    assert "That" in that
    assert "Most" not in that
    bitter = choose_english_beam(
        [
            "a full minute - and even this bitterest opponents",
            "a full minute - and even his bitterest opponents",
            "a full minute - and even his different opponents",
        ]
    )
    assert "bitterest" in bitter
    during = choose_english_beam(
        [
            "Donning the first",
            "During the first",
        ]
    )
    assert during.startswith("During")
    assert choose_english_beam(
        [
            "development being limited or an",
            "development being limited or an on",
        ]
    ).endswith("an")


def test_keeps_anson_over_allison_byron_padding():
    picked = choose_english_beam(
        [
            "George Anson Byron had seen enough of",
            "George Allison Byron Byron had seen enough of",
        ]
    )
    assert "Anson" in picked
    assert "Allison" not in picked
    assert picked.lower().count("byron") == 1


def test_keeps_oov_over_dictionary_completion():
    picked = choose_english_beam(
        [
            "the poet's atrocius conduct as a husband",
            "the poet's atrocities conduct as a husband",
        ]
    )
    assert "atrocius" in picked


def test_prefers_nearing_in_retirement_context():
    picked = choose_english_beam(
        [
            "of the problems of men and women nearing",
            "of the problems of men and women wearing",
        ],
        previous_text="already in retirement",
    )
    assert "nearing" in picked


def test_prefers_began_over_begun_without_have():
    picked = choose_english_beam(
        [
            "began to understand the value of",
            "begun to understand the value of",
        ]
    )
    assert picked.startswith("began")


def test_keeps_dick_d_over_dick_dr():
    picked = choose_english_beam(["Dick D.", "Dick Dr.", "Died Dr.", "DICK D."])
    assert picked == "Dick D."


def test_joins_split_cropped_last_word():
    picked = choose_english_beam(
        [
            "Review results in follow-up appoint me",
            "Review results in follow-up appointme",
            "Review results in follow-up appointment",
        ]
    )
    assert picked.endswith("appointme")


def test_keeps_given_over_even_discourse_name():
    picked = choose_english_beam(
        [
            "given for this style but still",
            "Even for this style but still",
        ]
    )
    assert picked.startswith("given")


def test_drops_year_soup_signature_hallucinations():
    assert choose_english_beam(["1961 1957 1959", "0thoologist"]) == ""
    assert score_english_hypothesis("1961 1957 1959") < 0.0
    assert choose_english_beam_or_first(["1961 1957 1959", "0thoologist"]) == "1961 1957 1959"


def test_does_not_append_singleton_to():
    picked = choose_english_beam(
        [
            "I don't really have the right",
            "I dont really have the right",
            "I don't really have the right to",
            "I don't really have the right .",
        ]
    )
    assert picked in {
        "I dont really have the right",
        "I don't really have the right",
    }


def test_splits_smashed_i_agree():
    assert repair_near_miss_tokens("Iagree with the Prime") == "I agree with the Prime"
    assert repair_near_miss_tokens("Italic is still") == "Italic is still"


def test_repairs_genjoy_prefix():
    picked = choose_english_beam(
        ["employ trying . Very slowly !", "genjoy trying . Very slowly !"]
    )
    assert picked.startswith("enjoy")


def test_prefers_dont_contraction():
    picked = choose_english_beam(
        [
            "I don really have the right",
            "I dont really have the right",
            "I don't really have the right",
        ]
    )
    assert picked in {
        "I dont really have the right",
        "I don't really have the right",
    }


def test_english_word_accepts_regular_ed_and_ing():
    assert is_english_word("granted")
    assert is_english_word("printed")
    assert is_english_word("writing")
    assert not is_english_word("limending")


def test_collapses_runaway_out_over_loop():
    looped = (
        'heart out over ? " Doc nodded towards the door . " You '
        + " ".join(["out over ?"] * 20)
    )
    assert collapse_decoder_loop(looped).startswith('heart out over ? " Doc nodded')
    assert collapse_decoder_loop(looped).count("out over") <= 3


def test_short_line_still_overlays_during_and_hyphen():
    assert pick_crop_hypothesis(
        ["Donning the first", "During the first"],
        400,
        80,
    ).startswith("During")
    assert pick_crop_hypothesis(
        ["Only a few hours of-", "Only a few hours af-"],
        500,
        90,
    ).endswith("af-")


def test_word_crop_keeps_first_beam_and_strips_period():
    assert pick_crop_hypothesis(["little .", "Little .", "little . Six"], 118, 112) == "little"
    assert pick_crop_hypothesis(["in", "j n ."], 81, 75) == "in"
    assert pick_crop_hypothesis(["fox .", "fox . Six"], 95, 130) == "fox"


def test_short_iam_line_salvages_vision_from_oov_first_beam():
    assert pick_crop_hypothesis(["visjoy", "vision .", "History ."], 348, 128) == "vision"


def test_short_iam_line_keeps_first_beam():
    # 348x128 is a real IAM line ("vision .") that is still wider than isolated words.
    assert pick_crop_hypothesis(["vision .", "History ."], 348, 128) == "vision"


def test_line_crop_still_uses_english_pick():
    picked = pick_crop_hypothesis(
        [
            "the foreign merchants , which whom the King's",
            "the foreign mechanics , which whom the King's",
        ],
        2400,
        128,
    )
    assert "merchants" in picked


def test_prefers_place_name_over_menace_consensus():
    beams = [
        "this is money due to menca anyway .",
        "this is money due to menace anyway .",
        "this is money due to Mecca anyway .",
        "this is money due to America anyway .",
        "thus is money due to menca anyway .",
        "this is money due to manca anyway .",
        "this is money due to menace any many .",
        "this is money due to Hence anyway .",
        "thus is money due to menace anyway .",
        "this is money due to menace anyway . .",
    ]
    picked = pick_crop_hypothesis(beams, 1920, 87)
    assert "menace" not in picked.lower()
    assert "Mecca" in picked or "America" in picked
    america = pick_crop_hypothesis(
        beams,
        1920,
        87,
        previous_text="United States officials quickly point out that",
    )
    assert "America" in america


def test_keeps_first_beam_name_over_unrelated_name():
    beams = [
        "Kennedy's rejection of it is a painful blow to the",
        "University's rejection of it is a painful blow to the",
        "Mammedy's rejection of it is a painful blow to the",
        "Unsequently's rejection of it is a painful blow to the",
        "Minneapolis's rejection of it is a painful blow to the",
        "Hrumedy's rejection of it is a painful blow to the",
        "themedy's rejection of it is a painful blow to the",
        "Minucky's rejection of it is a painful blow to the",
        "Kennedy's rejection of it is a painful blow to be",
        "Mammedy's rejection of it is a painful blow to be",
    ]
    picked = pick_crop_hypothesis(beams, 1509, 171)
    assert "Kennedy" in picked
    assert "University" not in picked


def test_prefers_consensus_merchants_over_mechanics():
    picked = choose_english_beam(
        [
            "the foreign merchants , which whom the King's",
            "the foreign merchants , which whom the king's",
            "the foreign mechanics , which whom the King's",
        ]
    )
    assert "merchants" in picked
    assert "mechanics" not in picked


def test_keeps_first_beam_dont_over_dont_apostrophe():
    picked = choose_english_beam(
        ["I dont really have the right", "I don't really have the right"]
    )
    assert "dont" in picked
    assert "don't" not in picked


def test_keeps_the_over_they_closed_class():
    picked = choose_english_beam(
        ["The conference will meet", "They conference will meet"]
    )
    assert picked.startswith("The")


def test_keeps_my_over_may_closed_class():
    picked = choose_english_beam(
        ["My everyday hand is similar and", "May everyday hand is similar and"]
    )
    assert picked.startswith("My")


def test_keeps_first_beam_dr_over_dictionary_mr():
    picked = choose_english_beam(
        [
            "Germany, Dr. Adenauer is in a tough",
            "Germany, Mr. Adenauer is in a tough",
            "Germany, Dr. Adenauer is in a tough",
            "Germany, Dr. Adenamer is in a tough",
        ]
    )
    assert "Dr." in picked
    assert "Mr." not in picked


def test_keeps_chequers_over_the_quers_split():
    picked = choose_english_beam(
        [
            "Macmillan at Chequers .",
            "Macmillan at Chequers :",
            "Macmillan at Chequers",
            "Macmillan at thequers .",
            "Macmillan at the Quers .",
            "Macmillan at Chequers'",
            "Macmillan at Chequers ?",
            "Macmillan at Chequers !",
            "Macmillan at the quers .",
            "Macmillan at Chequers .",
        ]
    )
    assert "Chequers" in picked
    assert "Quers" not in picked


def test_keeps_first_beam_willis_over_dictionary_williams():
    picked = choose_english_beam(
        [
            "Southern Senator - Willis Robertson, of",
            "Southern Senator - Williams Robertson, of",
            "Southern Senator - Willis Robertson, or",
            "Southern Senator - William Robertson, of",
        ]
    )
    assert "Willis" in picked
    assert "Williams" not in picked


def test_keeps_mp_over_mps_title_bonus():
    picked = choose_english_beam(
        [
            "Griffiths, MP for Manchester Exchange.",
            "Griffiths, MPs for Manchester Exchange.",
        ]
    )
    assert ", MP " in picked or picked.endswith("MP for Manchester Exchange.")
    assert "MPs" not in picked
    picked = choose_english_beam(
        ["Mr. Michael Foot has", "Mrs. Michael Foot has"]
    )
    assert picked.startswith("Mr.")


def test_line_api_keeps_first_beam():
    picked = pick_crop_hypothesis(
        [
            "assuredness \" Bella Bella Marie \" ( Parlophone )",
            "assuredness \" Bella Marie \" ( Parlophone )",
        ],
        1600,
        120,
        prefer_first=True,
    )
    assert "Bella Bella" in picked


def test_does_not_invent_given_name_after_mr():
    picked = pick_crop_hypothesis(
        [
            "his chief a report on his talks with Mr.",
            "his chies a report on his talks with Mr.",
            "his chiefs a report on his talks with Mr.",
            "his chief a report on his tasks with Mr.",
            "his chief as report on his talks with Mr.",
            "his chief a report on his talks with Mr. Her",
            "his ethics a report on his talks with Mr.",
            "his tries a report on his talks with Mr.",
            "his relies a report on his talks with Mr.",
            "his chief a report on his talks with Mr. .",
        ],
        1936,
        144,
    )
    assert picked.startswith("his chief")
    assert "Her" not in picked
