from collections.abc import Iterator

from openai import OpenAI

from config import settings
from prompts import (
    MAX_HISTORY_TURNS,
    SYSTEM_PROMPT_GENERAL,
    SYSTEM_PROMPT_LEGAL,
    SYSTEM_PROMPT_NO_RAG,
    USER_PROMPT_GENERAL,
    USER_PROMPT_LEGAL,
    USER_PROMPT_NO_RAG,
    USER_PROMPT_STATUTE_LOOKUP,
    USER_PROMPT_CASE_CONSULT,
    format_articles,
)

HistoryItem = dict[str, str]


def get_client() -> OpenAI:
    return OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )


def _trim_history(history: list[HistoryItem] | None) -> list[HistoryItem]:
    if not history:
        return []
    valid = [h for h in history if h.get("role") in ("user", "assistant") and h.get("content")]
    return valid[-MAX_HISTORY_TURNS:]


def build_no_rag_messages(
    question: str,
    history: list[HistoryItem] | None = None,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT_NO_RAG}]
    messages.extend(_trim_history(history))
    messages.append(
        {
            "role": "user",
            "content": USER_PROMPT_NO_RAG.format(question=question),
        }
    )
    return messages


def ask_llm_no_rag(question: str, history: list[HistoryItem] | None = None) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=build_no_rag_messages(question, history),
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def build_legal_messages(
    question: str,
    chunks: list[dict],
    history: list[HistoryItem] | None = None,
    *,
    statute_lookup: bool = False,
    case_consult: bool = False,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT_LEGAL}]
    messages.extend(_trim_history(history))
    if statute_lookup:
        user_template = USER_PROMPT_STATUTE_LOOKUP
    elif case_consult:
        user_template = USER_PROMPT_CASE_CONSULT
    else:
        user_template = USER_PROMPT_LEGAL
    messages.append(
        {
            "role": "user",
            "content": user_template.format(
                articles=format_articles(chunks),
                question=question,
            ),
        }
    )
    return messages


def build_general_messages(
    question: str,
    history: list[HistoryItem] | None = None,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT_GENERAL}]
    messages.extend(_trim_history(history))
    messages.append(
        {
            "role": "user",
            "content": USER_PROMPT_GENERAL.format(question=question),
        }
    )
    return messages


def ask_llm(question: str, chunks: list[dict], history: list[HistoryItem] | None = None) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=build_legal_messages(question, chunks, history),
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def ask_llm_general(question: str, history: list[HistoryItem] | None = None) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=build_general_messages(question, history),
        temperature=0.5,
    )
    return response.choices[0].message.content or ""


def stream_llm(
    question: str,
    chunks: list[dict],
    history: list[HistoryItem] | None = None,
    *,
    statute_lookup: bool = False,
    case_consult: bool = False,
) -> Iterator[str]:
    client = get_client()
    stream = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=build_legal_messages(
            question,
            chunks,
            history,
            statute_lookup=statute_lookup,
            case_consult=case_consult,
        ),
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def stream_llm_general(
    question: str,
    history: list[HistoryItem] | None = None,
) -> Iterator[str]:
    client = get_client()
    stream = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=build_general_messages(question, history),
        temperature=0.5,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


REPAIR_CITATION_SYSTEM = """你是法律回答校对助手。用户回答的【结论】【解读】已写好，但【法律依据】中存在错误或无法核实的法条引用。
请仅重写【法律依据】部分：只能引用「参考法条」列表中真实存在的法律名称与条号，不得编造。
保留【结论】【解读】原文不变（可原样复制）。输出必须包含完整的【结论】【解读】【法律依据】三段。"""


def repair_legal_citations(
    question: str,
    answer: str,
    chunks: list[dict],
    invalid_desc: str,
    history: list[HistoryItem] | None = None,
) -> str:
    """修正回答中错误的【法律依据】引用。"""
    client = get_client()
    user_content = f"""【当前问题】
{question}

【待修正回答】
{answer}

【无法核实的引用】
{invalid_desc}

【参考法条（仅能引用以下条目）】
{format_articles(chunks)}

请输出修正后的完整回答（含【结论】【解读】【法律依据】）。"""
    messages: list[dict] = [{"role": "system", "content": REPAIR_CITATION_SYSTEM}]
    messages.extend(_trim_history(history))
    messages.append({"role": "user", "content": user_content})
    response = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message.content or answer
