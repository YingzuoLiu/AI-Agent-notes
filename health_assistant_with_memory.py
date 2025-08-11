
# 简单版：对话式长期记忆健康助手（ConversationChain + BufferMemory + 向量库）
# 依赖：pip install langchain langchain-openai langchain-community chromadb

from __future__ import annotations
import os, json, datetime
from typing import List, Dict, Any

# ========== 0) LangChain 基础组件 ==========
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# 向量库（长期记忆）
from langchain_community.vectorstores import Chroma
try:
    from langchain_core.documents import Document
except Exception:
    from langchain.docstore.document import Document  # type: ignore

# ====== 1) LLM 与短期记忆（最近对话历史）======
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
session_id = "user-001"

memory = ConversationBufferMemory(
    memory_key="history",     # prompt 里用 {history}
    return_messages=True      # 以消息形式注入
)

# ====== 2) 长期记忆：向量库（Chroma 示例）======
emb = OpenAIEmbeddings()
vstore = Chroma(
    collection_name="health_memory",
    embedding_function=emb,
    persist_directory="./chroma_health"   # 落地到本地目录，持久化长期记忆
)

def lt_save(user_id: str, text: str, tags: List[str] = None, meta_extra: Dict[str, Any] = None):
    """把本轮“健康要点/摘要”存入向量库"""
    meta = {
        "user_id": user_id,
        "date": datetime.date.today().isoformat(),
        "tags": tags or []
    }
    if meta_extra:
        meta.update(meta_extra)
    vstore.add_documents([Document(page_content=text, metadata=meta)])

def lt_search(user_id: str, query: str, k: int = 4) -> List[Document]:
    """按 user_id 过滤 + 语义检索"""
    retriever = vstore.as_retriever(search_kwargs={"k": k, "filter": {"user_id": user_id}})
    return retriever.get_relevant_documents(query)

# ====== 3) 对话链（把“长期记忆摘要”塞进 {context}）======
prompt = PromptTemplate.from_template(
    "你是健康助手，只提供一般健康信息，不做诊断；遇到胸痛/呼吸困难/意识改变等红旗症状须建议尽快就医。\n"
    "【长期记忆摘要】\n{context}\n"
    "【对话历史】\n{history}\n"
    "【用户提问】{input}\n"
    "请结合长期记忆与对话历史，给出个性化、可执行的建议与提醒（用项目符号，尽量具体）。"
)
chat = ConversationChain(llm=llm, prompt=prompt, memory=memory, verbose=False)

# ====== 4) 一轮对话：检索→回答→回写长期记忆 ======
def handle_turn(user_id: str, user_input: str) -> str:
    docs = lt_search(user_id, user_input, k=4)
    context = "\n".join([f"- {d.metadata.get('date')}: {d.page_content}" for d in docs]) or "（暂无长期记忆命中）"
    answer = chat.run({"input": user_input, "context": context})
    summary = f"用户问：{user_input}；助手答要点：{answer[:200]}..."
    lt_save(user_id, summary, tags=["session_summary"])
    return answer

# ====== 5) 示例：运行 ======
if __name__ == "__main__":
    lt_save(session_id, "一周内晚饭后散步30分钟；血压多在 130/85；按时服用降压药。目标：收缩压 <125。", tags=["lifestyle","bp"])
    lt_save(session_id, "近两月体重从76kg降到74kg，早餐燕麦、午餐少盐，周末爬楼20层。", tags=["weight","diet"])

    print("\n--- 回合1 ---")
    print(handle_turn(session_id, "我最近偶尔头痛，血压偶尔到 138/88，怎么调整比较好？"))

    print("\n--- 回合2 ---")
    print(handle_turn(session_id, "下周要出差三天，如何维持血压稳定？帮我列个每日提醒清单。"))
