from __future__ import annotations


def build_qa_prompt(*, context: str, query: str) -> str:
    return (
        "你是一个中医临床诊疗问答助手。\n"
        "你的任务是根据下述给定的已知信息回答用户问题。\n\n"
        "已知信息:\n"
        f"{context}\n\n"
        "用户问：\n"
        f"{query}\n\n"
        "如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请直接回复“我无法回答您的问题”。\n"
        "请不要输出已知信息中不包含的信息或答案。\n"
        "请用中文回答用户问题。"
    )

