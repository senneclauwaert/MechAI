"""Error codes of the assistant module. Every code starts with ASSISTANT_.

ASSISTANT_AI_UNAVAILABLE and ASSISTANT_AI_DISABLED live in `app/llm.py` (it must not import a
module); the service uses those two strings as reason codes.
"""

from app.core.errors import AppError


# Until login (#26) exists, asking is only allowed in the dev environment.
def not_allowed() -> AppError:
    return AppError("ASSISTANT_NOT_ALLOWED", "You are not allowed to ask questions yet.", 403)


# The questions of the whole day reached DAILY_QUESTION_CAP.
def daily_limit() -> AppError:
    return AppError("ASSISTANT_DAILY_LIMIT", "The daily question limit has been reached.", 429)


# Only used as a reason code in the answer: the AI answered, but not in a usable form.
def answer_invalid() -> AppError:
    return AppError("ASSISTANT_ANSWER_INVALID", "The AI answer could not be used.", 502)
