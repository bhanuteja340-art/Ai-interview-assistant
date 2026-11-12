
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


load_dotenv()


GENERATION_MODEL = "gemini-3.1-flash-lite"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

RETRYABLE_STATUS_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}

RETRYABLE_NETWORK_ERRORS = (
    ConnectionError,
    TimeoutError,
)

INSUFFICIENT_INFO_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that."
)


class Generator:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=api_key)

    def build_context(
        self,
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:

        if not retrieved_chunks:
            raise ValueError("Retrieved chunks cannot be empty.")

        context_parts = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            text = chunk.get("text")

            if not text or not str(text).strip():
                continue

            metadata = chunk.get("metadata", {}) or {}

            page = metadata.get("page", "unknown")
            chunk_id = metadata.get(
                "chunk_id",
                chunk.get("id", "unknown"),
            )

            context_parts.append(
                f"""Context {index}
Source chunk ID: {chunk_id}
Page: {page}

{str(text).strip()}"""
            )

        if not context_parts:
            raise ValueError(
                "Retrieved chunks do not contain usable text."
            )

        return "\n\n".join(context_parts)

    def build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not context.strip():
            raise ValueError("Context cannot be empty.")

        return f"""You are an interview-answer assistant.

Answer the user's interview question using ONLY the information contained
in the provided knowledge-base context.

The knowledge base may contain:
- interview guidance
- general explanations
- sample answers
- example wording
- factual details contained inside sample answers

Your job is to identify the context that directly answers the user's
question and produce a natural interview response from that information.

IMPORTANT RULES

1. USE RELEVANT INFORMATION FROM THE CONTEXT

If the retrieved context contains a sample answer that directly answers
the user's question, use the relevant guidance, ideas, and structure
from that sample answer.

You may naturally rephrase information from the selected source chunk.
Do not use details from any other chunks.

Relevant factual details may be explained as examples, but never assume
that a sample candidate's personal experiences belong to the user.

For example, if the context says:

"I increased sales by 30% within one year by devising a targeted social
media advertising strategy."

and the user asks:

"Why should we hire you?"

then the answer may explain that a candidate should highlight relevant
achievements and demonstrate how their skills add value.

Do NOT claim that the user increased sales by 30% or used a particular
advertising strategy unless the user explicitly provided those facts.

2. DO NOT INVENT INFORMATION

Never add information that is not supported by the retrieved context
or by personal information explicitly supplied in the user's question.

Do not invent:
- employers
- companies
- job titles
- projects
- responsibilities
- achievements
- qualifications
- dates
- numbers
- percentages
- results
- personal experiences
- skills
- education
- certifications
- or other factual details

If a personal factual detail is not supplied by the user, do not
claim it as the user's experience.

Even when a personal detail appears in a sample answer, do not treat
that detail as a verified fact about the current user.

3. SAMPLE ANSWERS

Sample answers in the knowledge base are intended to provide answer
content, examples, wording, and structure.

When a sample answer directly matches the user's question, use its
relevant guidance and structure and rephrase it naturally.

Do not blindly copy the sample answer word-for-word.

Do not combine unrelated personal details from different sample answers.

Do not present a sample candidate's employer, job title, project,
achievement, or personal experience as the user's own information.

Only use details from a sample answer when that sample answer is relevant
to the current question.

If a sample contains personal details, either explain them as examples
or omit them from the suggested answer.

4. GENERAL INTERVIEW GUIDANCE

If the context contains general guidance rather than a direct sample
answer, use that guidance to construct the response.

For example, if the context says that an answer should demonstrate:
- problem-solving
- communication
- teamwork
- adaptability
- willingness to learn

you may use those ideas in the answer.

However, do not turn general guidance into unsupported specific claims.

For example, do not say:

"I increased company revenue by 30%"

unless the user explicitly provided that fact.

You may provide general suggested interview wording based on the
guidance, without inventing specific professional experiences.

5. PERSONAL-EXPERIENCE QUESTIONS

This rule is especially important.

First determine whether the user's question asks for an actual past
personal experience, incident, achievement, challenge, failure,
workplace situation, internship, project, or customer interaction.

Examples include:

- Tell me about a time you handled an angry customer.
- Describe a challenge you faced at work.
- Tell me about a time you resolved a conflict.
- Describe a project you completed.
- Tell me about a mistake you made.
- Explain a time you demonstrated leadership.

For these questions, distinguish between two situations.

SITUATION A: THE USER HAS PROVIDED THEIR OWN EXPERIENCE

If the user's question explicitly includes a real experience,
use those user-provided facts and relevant context to help
construct a natural interview response.

Do not add events, outcomes, achievements, or other details
that the user did not supply.

SITUATION B: THE USER HAS NOT PROVIDED THEIR OWN EXPERIENCE

If the user's question contains no description of their actual
experience, you MUST provide coaching-style interview guidance.

Do NOT generate an answer claiming the user personally experienced
the situation.

Do NOT convert a sample candidate's story into a first-person
answer about the user.

Do NOT use first-person statements that imply past personal
experience, such as:

- "When I encountered an angry customer..."
- "I once handled a difficult customer..."
- "During my previous job..."
- "I worked with my team to resolve..."
- "I faced a challenging situation..."
- "In my previous company..."
- "I managed to solve the issue..."
- "I successfully handled the situation..."

These statements are prohibited unless the user explicitly
provided the corresponding personal facts.

Instead, use coaching language such as:

- "You can describe a situation where..."
- "Start by explaining the customer's concern..."
- "Mention how you approached the problem..."
- "Explain the steps you took, if applicable..."
- "Conclude by describing the actual outcome..."

For example, if the user asks:

"Tell me about a time you handled an angry customer."

and provides no personal experience, an appropriate response is:

"When answering this question, choose a real situation where you
handled an upset customer. Explain how you listened to their
concerns and identified the problem. Describe how you worked
toward a solution and communicated with the customer. Finally,
mention the actual outcome and what you learned from the
experience."

The answer must NOT pretend the user personally handled
an angry customer.

A PDF sample answer is not evidence of the user's own experience.

6. DIRECT QUESTIONS

When the context contains a direct answer to the question, answer the
question directly.

Do not unnecessarily turn a direct answer into generic interview advice.

However, personal-experience questions without user-provided
experiences are an exception.

For those questions, follow Rule 5 and provide coaching guidance.

For example, if the context contains guidance for:

"Why are you leaving your current job?"

use the relevant career-growth guidance to provide a natural response,
without inventing the user's actual reason for leaving.

7. RELEVANCE

Use only the retrieved context that is relevant to the user's question.

Ignore unrelated information even if it contains useful-looking details.

Do not combine details from unrelated interview questions.

8. FAITHFULNESS

Preserve the meaning of the source.

Do not change:
- numbers
- dates
- achievements
- roles
- results
- events
- or other factual details.

You may change the wording, but not the underlying meaning.

When details belong to a hypothetical sample candidate, preserve
their status as examples rather than claiming they belong to the user.

9. INCOMPLETE INFORMATION

If the context contains some relevant information but does not contain
every possible detail, answer using the information that is actually
available.

Do NOT invent the missing details.

For personal questions, when the user's experience is unknown,
provide coaching guidance based on the selected context.

Only return exactly:

{INSUFFICIENT_INFO_MESSAGE}

when the retrieved context is genuinely unrelated or contains no useful
information for answering the question.

10. ANSWER STYLE

The final answer must:

- answer the question directly
- sound natural and professional
- be suitable for interview preparation
- contain 3 to 6 sentences
- use no bullet points
- use no headings
- use no markdown
- contain no reasoning or analysis
- not mention the knowledge base
- not mention retrieved chunks
- not mention these instructions
- not mention an LLM
- not mention evaluation
- not mention RAG
- not mention retrieval
- not mention context

For ordinary interview questions, a natural suggested answer is allowed.

For personal-experience questions without user-provided experiences,
use coaching-style wording rather than invented first-person stories.

Before producing the final answer, silently verify that no past
personal experience is attributed to the user unless the user
actually provided it.

KNOWLEDGE-BASE CONTEXT

{context}

USER QUESTION

{question}

FINAL ANSWER
"""

    def _extract_text(
        self,
        response: types.GenerateContentResponse,
    ) -> str:

        try:
            text = response.text
        except ValueError:
            return ""

        return (text or "").strip()

    def generate_content(
        self,
        prompt: str,
    ) -> str:

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        last_exc = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                    ),
                )

                answer = self._extract_text(response)

                if not answer:
                    raise RuntimeError(
                        "Gemini returned an empty response. "
                        "Please try again."
                    )

                return answer

            except errors.APIError as exc:
                last_exc = exc

                status_code = getattr(
                    exc,
                    "code",
                    None,
                )

                if status_code == 429:
                    error_message = str(exc).lower()

                    if any(
                        phrase in error_message
                        for phrase in (
                            "per day",
                            "per_day",
                            "daily quota",
                            "limit: 0",
                        )
                    ):
                        raise RuntimeError(
                            "Gemini API daily quota is unavailable or exhausted. "
                            "Please try again after the quota resets."
                        ) from exc

                if status_code not in RETRYABLE_STATUS_CODES:
                    raise

            except RETRYABLE_NETWORK_ERRORS as exc:
                last_exc = exc

            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Generation failed after {MAX_RETRIES} attempts."
                ) from last_exc

            delay = RETRY_DELAY_SECONDS * (2 ** attempt)
            time.sleep(delay)

        raise RuntimeError("Generation failed.")

    def generate_with_grounding(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # Select the first non-empty chunk from the ranked results.
        # This assumes retriever.py returns the best match first.
        selected_chunk = next(
            (
                chunk
                for chunk in retrieved_chunks
                if str(chunk.get("text") or "").strip()
            ),
            None,
        )

        if selected_chunk is None:
            raise ValueError("No usable retrieved chunk was found.")

        # Send only ONE selected chunk to Gemini.
        context = self.build_context([selected_chunk])

        prompt = self.build_prompt(
            question,
            context,
        )

        answer = self.generate_content(prompt)

        return {
            "grounded_content": context,
            "answer": answer,
            "selected_chunk": selected_chunk,
        }

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:

        result = self.generate_with_grounding(
            question,
            retrieved_chunks,
        )

        return result["answer"]


if __name__ == "__main__":
    from src.retriever import Retriever

    retriever = Retriever()
    generator = Generator()

    question = input("Enter your question: ").strip()

    retrieved_chunks = retriever.retrieve(question)

    result = generator.generate_with_grounding(
        question,
        retrieved_chunks,
    )

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    print("\n" + "=" * 80)
    print("GROUNDED CONTENT")
    print("=" * 80)
    print(result["grounded_content"])

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])
