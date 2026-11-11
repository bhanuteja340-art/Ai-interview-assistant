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
the user's question, use the substantive information from that sample
answer.

You may naturally rephrase and combine information from multiple relevant
chunks.

Do NOT ignore relevant factual details simply because they appear inside
a sample answer.

For example, if the context says:

"I increased sales by 30% within one year by devising a targeted social
media advertising strategy."

and the user asks:

"Why should we hire you?"

then the answer may use the 30% sales increase and targeted social media
strategy because those facts are explicitly present in the retrieved
context.

2. DO NOT INVENT INFORMATION

Never add information that is not supported by the retrieved context.

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

If a factual detail is not present in the context, do not create it.

3. SAMPLE ANSWERS

Sample answers in the knowledge base are intended to provide answer
content, examples, wording, and structure.

When a sample answer directly matches the user's question, use its
substantive information and rephrase it naturally.

Do not blindly copy the sample answer word-for-word.

Do not combine unrelated personal details from different sample answers.

Only use details from a sample answer when that sample answer is relevant
to the current question.

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

unless that fact actually appears in the retrieved context.

5. PERSONAL-EXPERIENCE QUESTIONS

For questions asking about a specific experience, event, achievement,
failure, resignation, internship, previous job, or other personal
situation, first look for a directly relevant sample answer or
candidate-specific information in the retrieved context.

If relevant factual information is present, use that information.

If the context only contains general guidance and does not contain a
relevant experience or sample answer, do not invent a personal event.

Instead, provide a coaching-style response explaining how the candidate
should answer using the available guidance.

For example:

"If this happened to you, I would explain the situation honestly,
avoid criticizing my previous employer, and focus on what I learned."

Do not invent a specific event that is not present in the context.

6. DIRECT QUESTIONS

When the context contains a direct answer to the question, answer the
question directly.

Do not unnecessarily turn a direct answer into generic interview advice.

For example, if the context contains a direct answer to:

"Why are you leaving your current job?"

use the relevant career-growth information from the context rather than
only explaining how someone should answer the question.

7. RELEVANCE

Use only the retrieved context that is relevant to the user's question.

Ignore unrelated chunks even if they contain useful-looking information.

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

9. INCOMPLETE INFORMATION

If the context contains some relevant information but does not contain
every possible detail, answer using the information that is actually
available.

Do NOT invent the missing details.

Only return exactly:

{INSUFFICIENT_INFO_MESSAGE}

when the retrieved context is genuinely unrelated or contains no useful
information for answering the question.

10. ANSWER STYLE

The final answer must:

- answer the question directly
- sound natural in an interview
- be suitable for speaking
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
    ) -> dict[str, str]:

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        context = self.build_context(retrieved_chunks)

        prompt = self.build_prompt(
            question,
            context,
        )

        answer = self.generate_content(prompt)

        return {
            "grounded_content": context,
            "answer": answer,
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