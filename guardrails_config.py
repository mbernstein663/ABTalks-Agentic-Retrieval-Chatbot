import re
from typing import Dict

import os

os.environ["GUARDRAILS_RUN_SYNC"] = "true"

from guardrails import Guard
from guardrails.validators import (
    Validator,
    ValidationResult,
    PassResult,
    FailResult,
    register_validator,
)

from redact_pii import redact_pii   # change to your actual file


PROVIDER_DISCLAIMER = (
    "I can provide general health and benefits information, but I can't "
    "diagnose medical conditions or recommend specific treatment. "
    "Please consult a licensed healthcare provider for medical advice."
)


@register_validator(name="input-safety", data_type="string")
class InputSafetyValidator(Validator):

    injection_patterns = [
        r"\bignore .*previous instructions?\b",
        r"\bignore .*prior instructions?\b",
        r"\boverride .*instructions?\b",
        r"\bshow .*system prompt\b",
        r"\breveal .*system prompt\b",
        r"\bbypass .*guardrails?\b",
    ]

    privacy_patterns = [
        r"\banother member'?s? claims?\b",
        r"\bother member'?s? claims?\b",
        r"\bsomeone else'?s? claims?\b",
        r"\blist .*member ids?\b",
        r"\blist .*diagnoses\b",
        r"\beveryone in the database\b",
    ]

    allowed_topics = [
        "insurance",
        "claim",
        "claims",
        "coverage",
        "plan",
        "premium",
        "deductible",
        "copay",
        "coinsurance",
        "benefit",
        "benefits",
        "provider",
        "medical",
        "health",
        "medication",
        "diagnosis",
        "treatment",
        "surgery",
        "x-ray",

        "network",
        "doctor",
        "provider",
        "physician",
        "hospital",
        "specialist",

    ]

    def _validate(
        self,
        value: str,
        metadata: Dict
    ) -> ValidationResult:

        text = value.lower()

        for pattern in self.injection_patterns:
            if re.search(pattern, text):
                return FailResult(
                    error_message="Prompt injection detected."
                )

        for pattern in self.privacy_patterns:
            if re.search(pattern, text):
                return FailResult(
                    error_message="Unauthorized member-data request detected."
                )

        history_followup = re.search(
            r"\b(?:repeat|summarize|summary|recap|rephrase|"
            r"reword|clarify|elaborate)\b"
            r"|\bexplain\s+(?:that|this|it|your answer)\b"
            r"|\bwhat you (?:said|mentioned|told me)\b",
            text,
            re.IGNORECASE,
        )

        # if not history_followup and not any(
        #     topic in text for topic in self.allowed_topics
        # ):
        #     return FailResult(
        #         error_message="Request is outside the chatbot's supported scope. This chatbot is intended to answer queries about our health insurance plans or outstanding claims. Feel free to try again and ask about what plans we offer."
        #     )

        return PassResult()


@register_validator(name="pii-leakage", data_type="string")
class PIILeakageValidator(Validator):

    def _validate(
        self,
        value: str,
        metadata: Dict
    ) -> ValidationResult:

        redacted = redact_pii(value, allow_public_contacts=True)

        if redacted != value:
            return FailResult(
                error_message="PII/PHI detected in model output.",
                fix_value=redacted,
            )

        return PassResult()


@register_validator(name="medical-advice", data_type="string")
class MedicalAdviceValidator(Validator):

    patterns = [
        r"\byou should take\b",
        r"\byou should start taking\b",
        r"\byou should stop taking\b",
        r"\byour condition is\b",
        r"\byou likely have\b",
        r"\byou probably have\b",
        r"\byou appear to have\b",
    ]

    def _validate(
        self,
        value: str,
        metadata: Dict
    ) -> ValidationResult:

        for pattern in self.patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return FailResult(
                    error_message="Possible medical diagnosis or treatment recommendation.",
                    fix_value=PROVIDER_DISCLAIMER,
                )

        return PassResult()


input_guard = (
    Guard()
    .use(
        InputSafetyValidator(
            on_fail="exception"
        )
    )
)

output_guard = (
    Guard()
    .use(PIILeakageValidator(on_fail="fix"))
    .use(MedicalAdviceValidator(on_fail="fix"))
)


def validate_input(text: str):
    try:
        input_guard.validate(text)
        return True, None

    except Exception as e:
        return False, str(e)


def validate_output(text: str) -> str:
    result = output_guard.validate(text)

    return result.validated_output or PROVIDER_DISCLAIMER