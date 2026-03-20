from models.text_encoder import TextEncoder


def test_language_detection_english_not_forced_to_mixed() -> None:
    encoder = TextEncoder()
    features = encoder.infer(
        chunk_text="We are building an AI copilot for retail demand planning.",
        language_hint="en-ta",
        slide_context="problem solution traction",
    )
    assert features["language_detected"] == "en"


def test_language_detection_tamil_script() -> None:
    encoder = TextEncoder()
    features = encoder.infer(
        chunk_text="நாங்கள் சிறு வணிகங்களுக்கு செயற்கை நுண்ணறிவு தீர்வு உருவாக்குகிறோம்",
        language_hint="en-ta",
        slide_context="",
    )
    assert features["language_detected"] in {"ta", "ta-en"}


def test_language_detection_mixed_script() -> None:
    encoder = TextEncoder()
    features = encoder.infer(
        chunk_text="நாங்கள் AI based தீர்வு உருவாக்குகிறோம் for rural sellers",
        language_hint="en-ta",
        slide_context="",
    )
    assert features["language_detected"] == "ta-en"


def test_language_detection_english_with_noise_and_hint() -> None:
    encoder = TextEncoder()
    features = encoder.infer(
        chunk_text="Our GTM plan is simple: pilot in 12 stores, measure revenue lift, then expand.",
        language_hint="en-ta",
        slide_context="",
    )
    assert features["language_detected"] == "en"


def test_language_detection_empty_defaults_to_english() -> None:
    encoder = TextEncoder()
    features = encoder.infer(
        chunk_text="",
        language_hint="en-ta",
        slide_context="",
    )
    assert features["language_detected"] == "en"
