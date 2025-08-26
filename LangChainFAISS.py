"""
LangChain + FAISS 对话式推荐演示

功能：
- 用模拟商品生成 Document 数据
- 使用 HuggingFace sentence-transformers 生成 embeddings（或可用 OpenAI Embeddings）
- 用 FAISS 构建向量检索库
- 使用 ConversationalRetrievalChain + ConversationBufferMemory 管理多轮对话
- 提供 FastAPI HTTP 接口：/recommend (POST) 接受 {user_id, utterance}
- 包含本地多轮对话模拟函数 `simulate_conversation`
"""

import os
import typing as t
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn

# LangChain imports
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # fallback local embeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# LLM choices
from langchain.llms import OpenAI  # optional
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# ---------- 1) 数据准备：模拟商品并转为 Document ----------
def generate_products(n=200):
    """生成一组模拟商品（简单示例）"""
    import random
    categories = ["手机", "笔记本", "耳机", "家居", "图书", "运动", "美容"]
    brands = ["A牌", "B牌", "C牌", "D牌"]
    products = []
    for i in range(1, n+1):
        cat = random.choice(categories)
        brand = random.choice(brands)
        title = f"{brand} {cat} 型号{i}"
        desc = (
            f"{title}：高性价比，适合追求性价比的用户。"
            f" 类别：{cat}。品牌：{brand}。主要卖点：快速、稳定、优质体验。"
            f" 适合人群：喜欢{cat}的用户。"
        )
        price = round(random.uniform(10, 2000), 2)
        prod = {
            "id": f"P{i:04d}",
            "title": title,
            "description": desc,
            "category": cat,
            "brand": brand,
            "price": price
        }
        products.append(prod)
    return products

def products_to_documents(products):
    docs = []
    for p in products:
        # 在 content 中合并字段，方便检索
        content = f"{p['title']}\n描述：{p['description']}\n品类：{p['category']}\n品牌：{p['brand']}\n价格：{p['price']}"
        metadata = {"product_id": p["id"], "title": p["title"], "price": p["price"], "category": p["category"], "brand": p["brand"]}
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

# ---------- 2) Embeddings + FAISS 构建 ----------
def build_faiss_index(documents, embedding_model=None, index_path="faiss_index"):
    """
    documents: list of langchain.Document
    embedding_model: a langchain embeddings object (if None, use HuggingFaceEmbeddings)
    """
    if embedding_model is None:
        # 使用 sentence-transformers 的多语言/中文模型（如果机器能下载）
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 构建 FAISS 索引（内存版）；如需持久化，可保存到磁盘
    vectorstore = FAISS.from_documents(documents, embedding_model)
    # 保存索引到本地目录
    vectorstore.save_local(index_path)
    return vectorstore

def load_faiss_index(index_path="faiss_index", embedding_model=None):
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(index_path):
        raise FileNotFoundError("Index path not found. Run build first.")
    vs = FAISS.load_local(index_path, embedding_model)
    return vs

# ---------- 3) LLM backend factory ----------
def get_llm():
    """
    优先使用 OpenAI（如果有 API KEY），否则尝试用本地 transformers pipeline。
    调整这里可切换到你偏好的 LLM（如 Llama 绑定 / GPT4All / etc.）
    """
    if os.getenv("OPENAI_API_KEY"):
        # 可通过 env var 提供 OPENAI_API_KEY
        llm = OpenAI(temperature=0.2, max_tokens=512)
        return llm
    else:
        # HuggingFace pipeline: 这里示例使用 text-generation pipeline 和小型模型
        # 你需要本地能下载该模型，或换成你本地已有模型名
        model_name = os.getenv("HF_LOCAL_MODEL", "bigscience/bloom-3b")  # 修改为可用模型
        pipe = pipeline(
            "text-generation",
            model=model_name,
            device=0 if (os.getenv("USE_CUDA") == "1") else -1,
            max_length=512,
            do_sample=False,
        )
        hf_llm = HuggingFacePipeline(pipeline=pipe)
        return hf_llm

# ---------- 4) Prompt 模板（用于文档组合 / 推荐生成） ----------
PROMPT_TEMPLATE = """
你是一个商品推荐助手。基于用户的当前查询以及对话历史，挑选最相关的商品并给出理由。
检索到的商品信息如下：
{context}

用户当前输入: {question}

请：
1) 简短总结用户需求（1-2句）
2) 基于检索到的商品，推荐 3 个最合适的商品，每个给出 1-2 行推荐理由（要结合用户需求和商品的特点）
3) 如果需要澄清问题，给出一条澄清问题建议（可选）

严格输出 JSON 格式：
{{
  "summary": "...",
  "recommendations": [
     {{"product_id":"...", "title":"...", "price":..., "reason":"..."}}
  ],
  "clarify": "..." 或 null
}}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE
)

# ---------- 5) ConversationalRetrievalChain 构建 ----------
def build_conversational_chain(vectorstore, llm=None):
    if llm is None:
        llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        # combine_docs_chain_kwargs 可以接 prompt 等
        combine_docs_chain_kwargs={"prompt": prompt, "verbose": False},
    )
    return chain, memory

# ---------- 6) FastAPI 服务 ----------
app = FastAPI()

class ReqSchema(BaseModel):
    user_id: str
    utterance: str

# global placeholders (in-memory for demo)
GLOBAL = {
    "vectorstore": None,
    "chain": None,
    "memory": None,
    "products": None
}

@app.on_event("startup")
def startup_event():
    # 1) 生成/加载数据与索引（若已存在则加载）
    products = generate_products(300)
    docs = products_to_documents(products)
    GLOBAL["products"] = {p["id"]: p for p in products}

    index_path = "faiss_index"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(index_path):
        print("加载已存在 FAISS 索引...")
        vs = load_faiss_index(index_path, embedding_model)
    else:
        print("构建 FAISS 索引（可能需要下载 embeddings 模型）...")
        vs = build_faiss_index(docs, embedding_model, index_path=index_path)

    chain, memory = build_conversational_chain(vs, llm=get_llm())
    GLOBAL["vectorstore"] = vs
    GLOBAL["chain"] = chain
    GLOBAL["memory"] = memory
    print("服务启动完成。")

@app.post("/recommend")
def recommend(req: ReqSchema):
    chain = GLOBAL.get("chain")
    if chain is None:
        return {"error": "系统未就绪，请稍后重试。"}
    # LangChain chain 期望传入参数名通常为 "question"（不同版本略有差异）
    result = chain({"question": req.utterance})
    # chain 输出结构通常包含 "answer"（模型文本响应）和 memory 内部会更新
    return {"llm_response": result.get("answer"), "chat_history": GLOBAL["memory"].load_memory_messages()}

# ---------- 7) 简单的多轮对话模拟（可在本地脚本运行） ----------
def simulate_conversation(turns: t.List[str]):
    """
    用于在启动后模拟多轮对话：
    - 先确保 app 启动、GLOBAL['chain'] 已建立
    """
    chain = GLOBAL.get("chain")
    if chain is None:
        raise RuntimeError("Chain 尚未初始化。")
    print("开始模拟对话：")
    for i, u in enumerate(turns, 1):
        print(f"\n用户第{i}轮: {u}")
        res = chain({"question": u})
        print("系统：", res.get("answer"))

# ---------- 8) 本地运行 ----------
if __name__ == "__main__":
    # 启动 uvicorn 服务（调试用）
    # 在真实部署时，可使用： uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
