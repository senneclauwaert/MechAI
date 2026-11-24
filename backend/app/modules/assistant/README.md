# assistant module

Owns the repair assistant: it answers a mechanic's question from the manufacturer's handbook, always
with the pages it used, or says honestly that the handbook does not cover it. Every question leaves
a trace (schema `assistant`, table `traces`). The app shows answers backed by the handbook; it never
presents AI text as the handbook itself.

**Dev only until #26** (login and roles): `POST /ask` refuses with `ASSISTANT_NOT_ALLOWED` (403)
when `ENVIRONMENT` is not `dev`.

## Endpoint

`POST /api/v1/assistant/ask`

Request (`AskIn`): `question` (2-500 characters after stripping), `kart_number` (optional, 1-10
characters, letters, digits and `-`), `input` (`typed` or `voice`, default `typed`).

Response (`AskOut`, status 200 for every outcome): `trace_id`, `outcome`, `safety_level`,
`question_language` (`nl`, `fr`, `en` or null), `answer` (steps, parts, warnings, confidence,
missing information; only when `outcome` is `answer`), `sources` (S1..S5 with section number,
heading path, pages, document title and revision, never the section text) and `reason_code`.

Errors use the standard shape (`code`, `message`, `reference`): `ASSISTANT_NOT_ALLOWED` (403),
`ASSISTANT_DAILY_LIMIT` (429), `KNOWLEDGE_NOTHING_EMBEDDED` (409, after the trace is saved),
`CORE_VALIDATION_FAILED` (422).

## Outcomes

| Outcome | Meaning |
| --- | --- |
| `answer` | Cited steps that passed every check |
| `not_in_handbook` | Best similarity below `REFUSAL_THRESHOLD`, or the AI found no steps. The 3 closest sources are shown |
| `retrieval_only` | Pages only: AI off, AI down, answer invalid, wiring question, embedding down or handbook text not allowed out |
| `high_voltage` | High-voltage question (code, AI flag or a failed translation): pages only, no steps |
| `emergency` | Smoke, fire, swollen or hot battery, injury: no AI call, no search, no sources |
| `refused` | Bypass of a protection: no AI call, no search, no sources |

`reason_code` explains a fallback: `ASSISTANT_AI_DISABLED`, `ASSISTANT_AI_UNAVAILABLE`,
`ASSISTANT_ANSWER_INVALID`, `KNOWLEDGE_EXTERNAL_EMBEDDINGS_OFF` or `KNOWLEDGE_NOTHING_EMBEDDED`.

How a question runs: the word lists (`safety/words_v1.toml`) decide before any AI call; translate
(1 call); search through `knowledge.retrieve_sections`; answer (1 call, 1 retry on invalid output);
then code checks the answer (sources exist, every number and part number is in the cited sections,
no high-voltage, emergency, bypass or wiring words). If a check fails, all steps are dropped. The
stricter safety level always wins. The prompts are versioned files in `backend/prompts/`.

## Settings

`REFUSAL_THRESHOLD` (default 0.70, a placeholder until measured), `ANSWER_SECTIONS`,
`ANSWER_SOURCE_TOKENS`, `ANSWER_MAX_TOKENS`, `AI_TIMEOUT_SECONDS`, `DAILY_QUESTION_CAP` (questions
that used the AI since 00:00 UTC; it never blocks emergency or refused), plus `AI_ENABLED` and
`ALLOW_EXTERNAL_EMBEDDINGS` (which also guards handbook text sent to the chat model).

The worst case of one request is about 80 s (translate, embedding, answer and retry, 20 s each), so
`web/nginx.conf` gives `/api/v1/assistant/` a read timeout of 90 s.

## What comes later

- **#34 (safety):** typo-tolerant word matching, the STOP card, Supercube's fixed texts, the logged
  confirmation and the failure-code lookup exception. The current gate is strict and small.
- **#37 (number check):** units and the check per cited page. Today a number must be in one of the
  cited sections.

## Known limits

- Too strict on purpose: a battery word in an axle answer, or a digit that is not in the cited
  section, drops all steps. The trace flags show how often.
- A long section is stored in parts; search may find only one, so an answer can miss steps.
- The threshold 0.70 is a guess until it is measured (#29).
- Several current handbooks are all searched; keep one in dev.
- Word lists have no typo tolerance yet.
- Slow answers: up to three AI calls plus the embedding inside one request.

## Error and reason texts

`ASSISTANT_AI_UNAVAILABLE` and `ASSISTANT_AI_DISABLED` live in `app/llm.py`; the service uses them as
reason codes. The texts are for the frontend (#24 copies them).

| Code | nl | fr | en |
| --- | --- | --- | --- |
| `ASSISTANT_NOT_ALLOWED` | Je mag nog geen vragen stellen. | Vous ne pouvez pas encore poser de questions. | You are not allowed to ask questions yet. |
| `ASSISTANT_DAILY_LIMIT` | De limiet van vragen voor vandaag is bereikt. | La limite de questions pour aujourd'hui est atteinte. | The daily question limit has been reached. |
| `ASSISTANT_ANSWER_INVALID` | Het AI-antwoord kon niet gebruikt worden. De pagina's worden getoond. | La reponse de l'IA n'a pas pu etre utilisee. Les pages sont affichees. | The AI answer could not be used. The pages are shown. |
| `ASSISTANT_AI_UNAVAILABLE` | De AI is nu niet bereikbaar. De pagina's worden getoond. | L'IA n'est pas disponible pour le moment. Les pages sont affichees. | The AI is not available right now. The pages are shown. |
| `ASSISTANT_AI_DISABLED` | De AI staat uit. Alleen zoeken in het handboek is mogelijk. | L'IA est desactivee. Seule la recherche dans le manuel est possible. | The AI is switched off. Only searching the handbook is possible. |

## Files

| File | What goes in it |
| --- | --- |
| `models.py` | The `Trace` table (schema `assistant`) |
| `schemas.py` | Request, model output and response shapes (Pydantic) |
| `service.py` | The pipeline `ask()`. The only file other modules may call |
| `router.py` | The endpoint. Thin, only calls `service.py` |
| `errors.py` | Error codes, all starting with `ASSISTANT_` |
| `checks.py` | Private: pure safety and answer checks, imported only by `service.py` |
