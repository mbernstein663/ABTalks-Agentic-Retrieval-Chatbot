from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

import re


analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


# Custom member ID: M1003
member_id_recognizer = PatternRecognizer(
    supported_entity="MEMBER_ID",
    patterns=[
        Pattern(
            name="member_id",
            regex=r"\bM\d{4}\b",
            score=0.95
        )
    ],
    context=["member", "member id", "member number"]
)


# Custom claim ID: C1003
claim_id_recognizer = PatternRecognizer(
    supported_entity="CLAIM_ID",
    patterns=[
        Pattern(
            name="claim_id",
            regex=r"\bC\d{4}\b",
            score=0.95
        )
    ],
    context=["claim", "claim id", "claim number"]
)


analyzer.registry.add_recognizer(member_id_recognizer)
analyzer.registry.add_recognizer(claim_id_recognizer)


ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_BANK_NUMBER",
    "MEMBER_ID",
    "CLAIM_ID",
]


# Verified in medical_care_FAQ.txt.
PUBLIC_CONTACT_NUMBERS = frozenset({
    "8003182596",
    "8558894325",
})

# Exact synthetic training identifiers verified in benefits.txt.
VERIFIED_PLAN_IDS = frozenset({
    "P101", "P102", "P103",
})

VERIFIED_SYNTHETIC_CLAIM_IDS = frozenset({
    "C1001", "C1002", "C1003", "C1004", "C1005",
})

VERIFIED_PLAN_NAMES = frozenset({
    "Gold PPO", "Silver HMO", "Bronze HMO",
})


def _normalized_phone(value):
    digits = re.sub(r"[^0-9]", "", value)

    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]

    return digits


def _verified_spans(text, values):
    spans = []

    for value in values:
        pattern = (
            r"(?<!\w)"
            + re.escape(value).replace(r"\ ", r"\s+")
            + r"(?!\w)"
        )

        spans.extend(
            (match.start(), match.end())
            for match in re.finditer(
                pattern, text, flags=re.IGNORECASE
            )
        )

    return spans


def redact_pii(
    text: str,
    *,
    allow_public_contacts: bool = False,
) -> str:
    if not text:
        return text

    results = analyzer.analyze(
        text=text,
        language="en",
        entities=ENTITIES,
        score_threshold=0.4,
    )

    id_spans = _verified_spans(
        text,
        VERIFIED_PLAN_IDS | VERIFIED_SYNTHETIC_CLAIM_IDS,
    )
    plan_name_spans = _verified_spans(text, VERIFIED_PLAN_NAMES)

    identifier_entities = {
        "CLAIM_ID",
        "US_DRIVER_LICENSE",
        "US_PASSPORT",
        "US_ITIN",
        "US_BANK_NUMBER",
        "PHONE_NUMBER",
    }

    def exempt(result):
        if result.entity_type in identifier_entities and any(
            start <= result.start and result.end <= end
            for start, end in id_spans
        ):
            return True

        if result.entity_type == "PERSON" and any(
            start <= result.start and result.end <= end
            for start, end in plan_name_spans
        ):
            return True

        return (
            allow_public_contacts
            and result.entity_type == "PHONE_NUMBER"
            and _normalized_phone(
                text[result.start:result.end]
            ) in PUBLIC_CONTACT_NUMBERS
        )

    results = [
        result for result in results
        if not exempt(result)
    ]

    operators = {
        entity: OperatorConfig(
            "replace",
            {"new_value": f"[{entity}]"},
        )
        for entity in ENTITIES
    }

    return anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    ).text

# sample = ["My member ID is M1002", "My phone number is 516-509-8364", "I'm not sure what my plan ID is"]


# for samp in sample:
#     print("\n \n ===== \n" + redact_pii(samp))