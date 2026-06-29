import json
import re

from llm import get_client
from config import settings

CLASSIFY_SYSTEM = """你是问题分类器。判断用户输入是否属于「中国法律相关咨询」。
法律相关包括：权利义务、合同纠纷、婚姻家庭、继承、侵权、宪法/民法典知识、法律程序等。
非法律相关包括：天气、编程、娱乐闲聊、数学题、生活常识等与法律无关的内容。
若当前问题是追问且对话历史涉及法律话题，应判定为法律相关。
只输出 JSON，格式：{"is_legal": true} 或 {"is_legal": false}"""


def is_legal_question(question: str, history: list[dict] | None = None) -> bool:
    if _heuristic_legal(question):
        return True
    if _heuristic_non_legal(question) and not history:
        return False

    client = get_client()
    try:
        if history:
            lines = [f"{h['role']}: {h['content']}" for h in history[-6:]]
            user_content = (
                "【相关对话历史】\n"
                + "\n".join(lines)
                + f"\n\n【当前问题】\n{question}"
            )
        else:
            user_content = question
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=32,
        )
        text = (response.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return bool(json.loads(match.group())["is_legal"])
    except Exception:
        pass
    return True


def _heuristic_legal(question: str) -> bool:
    keywords = (
        "法", "条例", "合同", "侵权", "继承", "婚姻", "离婚", "宪法", "民法典",
        "刑法", "犯罪", "刑罚", "劳动", "工伤", "加班", "工资", "劳动合同",
        "监护", "收养", "租赁", "抵押", "担保", "人格权", "隐私",
    )
    return any(k in question for k in keywords)


def _heuristic_non_legal(question: str) -> bool:
    patterns = (
        r"^(你好|您好|hi|hello)[!！]?$",
        r"天气",
        r"代码|编程|python|java",
        r"笑话|故事",
        r"翻译",
    )
    q = question.strip().lower()
    return any(re.search(p, q, re.I) for p in patterns)
