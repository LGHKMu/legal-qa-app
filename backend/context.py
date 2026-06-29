import json
import re

from llm import get_client
from config import settings
from prompts import MAX_HISTORY_TURNS

RELATED_SYSTEM = """你是对话上下文分析器。判断【当前问题】与【对话历史】是否属于同一话题的连续交流。

应关联（related=true）的情况：
- 对上文追问、补充、细化（如"那法定年龄呢""刚才说的第二种""详细解释一下"）
- 指代上文内容（如"这个""那种情况""还有呢"）
- 同一法律问题的不同侧面

不应关联（related=false，turns=0）的情况：
- 开启全新、与上文无关的话题（如上文问结婚，当前问合同违约、宪法权利、天气等）
- 两个独立的法律问题之间没有指代或延续关系

输出 JSON，格式：{"related": true/false, "turns": 数字}
- turns 表示从最近一条历史往前纳入的消息条数（user/assistant 交替计数），无关时 turns=0
- 一般追问 turns=2（上一轮问答），同一话题多轮深挖可 turns=4 或 6，最多 10"""

FOLLOWUP_PATTERNS = (
    r"^(那|这|它|其|还|再|另外|继续|详细|具体|进一步)",
    r"(呢|吗)[?？]?$",
    r"刚才|上面|之前|前面|第二种|第一种|第三种",
    r"什么意思|为什么|怎么理解|展开",
)


def filter_relevant_history(question: str, history: list[dict] | None) -> list[dict]:
    """仅保留与当前问题相关的历史；无关问题之间不串联。"""
    if not history:
        return []

    trimmed = [
        h for h in history
        if h.get("role") in ("user", "assistant") and h.get("content", "").strip()
    ]
    if not trimmed:
        return []

    if _is_obvious_followup(question):
        return trimmed[-MAX_HISTORY_TURNS:]

    turns = _detect_related_turns(question, trimmed)
    if turns <= 0:
        return []
    return trimmed[-turns:]


def _is_obvious_followup(question: str) -> bool:
    q = question.strip()
    return any(re.search(p, q) for p in FOLLOWUP_PATTERNS)


def _detect_related_turns(question: str, history: list[dict]) -> int:
    recent = history[-6:]
    lines = [f"{h['role']}: {h['content'][:200]}" for h in recent]
    user_content = (
        "【对话历史（由远及近）】\n"
        + "\n".join(lines)
        + f"\n\n【当前问题】\n{question}"
    )

    client = get_client()
    try:
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": RELATED_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=48,
        )
        text = (response.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if not data.get("related", False):
                return 0
            turns = int(data.get("turns", 0))
            return min(max(turns, 0), MAX_HISTORY_TURNS)
    except Exception:
        pass

    return _fallback_related_turns(question, history)


def _fallback_related_turns(question: str, history: list[dict]) -> int:
    """LLM 不可用时的保守回退：仅当与上一轮用户问题有词重叠时关联。"""
    last_user = next(
        (h["content"] for h in reversed(history) if h.get("role") == "user"),
        "",
    )
    if not last_user:
        return 0
    q_chars = set(re.findall(r"[\u4e00-\u9fff]{2,}", question))
    u_chars = set(re.findall(r"[\u4e00-\u9fff]{2,}", last_user))
    overlap = q_chars & u_chars
    if len(overlap) >= 2 or (len(overlap) >= 1 and len(question) < 15):
        return min(2, len(history))
    return 0
