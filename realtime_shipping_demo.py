"""
功能一览：
- ✅ LlamaIndex 倒排索引 + 向量索引（两种检索）
- ✅ LangChain 做一个回答模板（把检索结果组织成人话）
- ✅ 三层缓存：嵌入 / 检索候选 / 最终答案
- ✅ 异步增量更新（后台微批，查询不被阻塞）
- ✅ 简单用户验证（X-API-Key -> customer_id）
- ✅ 基于 customer_id 的结果过滤（只看自己货）

运行：
    uvicorn realtime_shipping_demo:app --reload
试试看：
    curl -H "X-API-Key: key_alice" "http://127.0.0.1:8000/query?q=SF1234567890"
    curl -H "X-API-Key: key_alice" "http://127.0.0.1:8000/query?q=我那双球鞋现在到了哪"
    # 模拟新事件（异步入库）
    curl -X POST -H "Content-Type: application/json" -H "X-API-Key: key_alice" \
      -d '{"tracking_no":"SF1234567890","status":"Out for delivery","location":"上海虹口","ts":1723559999}' \
      http://127.0.0.1:8000/event
"""

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Body, Query
from pydantic import BaseModel, Field

# —— LlamaIndex：索引与检索 —— #
from llama_index.core import (
    VectorStoreIndex,
    KeywordTableIndex,
    StorageContext,
    Settings,
)
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# —— LangChain：回答模板（轻编排）—— #
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


# ========== 0) 极简“数据库”：初始运单数据（内存） ==========
@dataclass
class Shipment:
    tracking_no: str
    customer_id: str
    title: str            # 简短描述（展示用）
    status: str           # 当前状态
    location: str         # 当前位置
    history: List[str]    # 事件轨迹（简单字符串）

# 两个客户（Alice/Ben）各有一票货
DB: Dict[str, Shipment] = {
    "SF1234567890": Shipment(
        tracking_no="SF1234567890",
        customer_id="alice",
        title="球鞋（红色 42 码）",
        status="In transit",
        location="杭州中转场",
        history=["打包完成", "已揽收 杭州", "中转至 杭州中转场"],
    ),
    "YT9990008887": Shipment(
        tracking_no="YT9990008887",
        customer_id="ben",
        title="咖啡机",
        status="Out for delivery",
        location="上海浦东",
        history=["打包完成", "已揽收 上海", "派送中 上海浦东"],
    ),
}

# ========== 1) 用户身份验证（超简版） ==========
USERS = {
    "key_alice": {"customer_id": "alice"},
    "key_ben": {"customer_id": "ben"},
}
def auth(api_key: Optional[str]) -> str:
    u = USERS.get(api_key or "")
    if not u:
        raise HTTPException(401, "Invalid or missing X-API-Key")
    return u["customer_id"]

# ========== 2) 嵌入与索引（LlamaIndex） ==========
# 嵌入模型（并带一个极简“内存缓存”避免重复计算）
class CachedHF(HuggingFaceEmbedding):
    _memo: Dict[str, List[float]] = {}
    def get_text_embedding(self, text: str) -> List[float]:
        key = hashlib.sha256(("all-minilm-l6-v2::" + text.lower().strip()).encode()).hexdigest()
        if key in self._memo:
            return self._memo[key]
        vec = super().get_text_embedding(text)
        self._memo[key] = vec
        return vec

Settings.embed_model = CachedHF(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 把 Shipment 变成文本节点（带元数据）
def shipment_to_node(s: Shipment) -> TextNode:
    text = (
        f"运单号 {s.tracking_no}\n"
        f"客户 {s.customer_id}\n"
        f"标题 {s.title}\n"
        f"当前状态 {s.status} @ {s.location}\n"
        f"轨迹：{'；'.join(s.history)}"
    )
    return TextNode(
        id_=s.tracking_no, text=text,
        metadata={"tracking_no": s.tracking_no, "customer_id": s.customer_id}
    )

# 用倒排 + 向量两套索引
def build_indexes(nodes: List[TextNode]) -> Tuple[KeywordTableIndex, VectorStoreIndex]:
    storage = StorageContext.from_defaults()
    kw = KeywordTableIndex(nodes, storage_context=storage)     # 倒排索引：关键词/单号精确匹配
    vec = VectorStoreIndex(nodes, storage_context=storage)     # 向量索引：语义近邻
    return kw, vec

# 初始化索引
_nodes = [shipment_to_node(s) for s in DB.values()]
KW_INDEX, VEC_INDEX = build_indexes(_nodes)

# ========== 3) 缓存层 ==========
# （a）检索候选缓存： (customer_id, normalized_query) -> [tracking_no...]
RETR_CACHE: Dict[str, Tuple[float, List[str]]] = {}  # val=(expire_ts, ids)
# （b）答案缓存： (customer_id, query, ids_digest) -> answer_text
ANS_CACHE: Dict[str, Tuple[float, str]] = {}

def _now(): return time.time()
def _norm(q: str) -> str: return " ".join(q.lower().split())

def get_retr_cache_key(customer_id: str, q: str) -> str:
    return f"ret:{customer_id}:{hashlib.sha256(_norm(q).encode()).hexdigest()}"

def get_ans_cache_key(customer_id: str, q: str, ids: List[str]) -> str:
    digest = hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
    return f"ans:{customer_id}:{hashlib.sha256((_norm(q)+'::'+digest).encode()).hexdigest()}"

# ========== 4) 简单路由：像单号就走倒排；否则向量 ==========
TRACK_RE = re.compile(r"[A-Z]{2}\d{8,}")

def route_and_retrieve(customer_id: str, q: str, k: int = 5) -> List[NodeWithScore]:
    # 先看检索缓存
    ck = get_retr_cache_key(customer_id, q)
    exp, ids = RETR_CACHE.get(ck, (0, []))
    if exp > _now():
        # 命中缓存：直接把 ids 变回 nodes
        nodes = []
        for tid in ids:
            n = next((n for n in _nodes if n.metadata["tracking_no"] == tid), None)
            if n and n.metadata["customer_id"] == customer_id:
                nodes.append(NodeWithScore(node=n, score=1.0))
        return nodes[:k]

    # 没命中缓存 → 真检索
    results: List[NodeWithScore] = []
    if TRACK_RE.fullmatch(q.strip().upper()):
        # 倒排：关键词/单号
        qe = KW_INDEX.as_query_engine(similarity_top_k=k)
        resp = qe.query(q)
        results = list(resp.source_nodes or [])
    else:
        # 向量：自然语言语义检索
        qe = VEC_INDEX.as_query_engine(similarity_top_k=k)
        resp = qe.query(q)
        results = list(resp.source_nodes or [])

    # 结果按 customer_id 过滤
    filtered = [r for r in results if r.node.metadata.get("customer_id") == customer_id]
    # 写入检索缓存（短 TTL：60s）
    RETR_CACHE[ck] = (_now() + 60, [r.node.metadata["tracking_no"] for r in filtered])
    return filtered[:k]

# ========== 5) LangChain：把命中结果组织成“人话答案” ==========
PROMPT = ChatPromptTemplate.from_template(
    "你是物流助手。请基于检索到的运单信息（可能是多票）用简洁中文回答：\n"
    "问题：{question}\n"
    "命中列表：\n{hits}\n"
    "要求：给出每个运单的当前状态与位置，列出最近一条轨迹；如果没有命中，礼貌说明没找到。"
)

def format_hits(nodes: List[NodeWithScore]) -> str:
    if not nodes: return "(无命中)"
    lines = []
    for r in nodes:
        # 从 node 文本里抓关键行（仅示意）
        lines.append(r.node.text.splitlines()[0] + "｜" + r.node.text.splitlines()[3])
    return "\n".join(lines)

# 这里不连在线 LLM，为了可跑，用一个“伪 LLM”（把模板格式化输出）
def fake_llm_answer(inputs: dict) -> dict:
    if "(无命中)" in inputs["hits"]:
        txt = f"没有找到与“{inputs['question']}”相关的运单。请核对运单号或描述。"
    else:
        txt = f"关于“{inputs['question']}”，为你找到这些运单：\n{inputs['hits']}"
    return {"answer": txt}

CHAIN = PROMPT | RunnableLambda(fake_llm_answer)  # LangChain 的“可运行链”

def answer_with_cache(customer_id: str, q: str, nodes: List[NodeWithScore]) -> str:
    ids = [n.node.metadata["tracking_no"] for n in nodes]
    ak = get_ans_cache_key(customer_id, q, ids)
    exp, ans = ANS_CACHE.get(ak, (0, ""))
    if exp > _now():
        return ans
    # 生成答案（这里用 fake_llm；接入真 LLM 时只要换 Runnable 即可）
    out = CHAIN.invoke({"question": q, "hits": format_hits(nodes)})
    ans = out["answer"]
    ANS_CACHE[ak] = (_now() + 10 * 60, ans)  # 10 分钟答案缓存
    return ans

# ========== 6) 异步更新：事件入队，后台微批更新索引 ==========
EVENT_Q: asyncio.Queue = asyncio.Queue()
BATCH, WAIT = 64, 2.0  # 微批：最多 64 条或 2 秒

class ShipEvent(BaseModel):
    tracking_no: str = Field(..., description="运单号")
    status: str = Field(..., description="新状态")
    location: str = Field(..., description="位置")
    ts: int = Field(..., description="事件时间戳")

async def index_updater():
    """后台任务：攒一小批事件，统一更新 DB -> 重新构建索引（小规模用重建最省心）"""
    global _nodes, KW_INDEX, VEC_INDEX
    buf: List[ShipEvent] = []
    last = time.time()
    while True:
        try:
            evt: ShipEvent = await asyncio.wait_for(EVENT_Q.get(), timeout=0.5)
            buf.append(evt)
        except asyncio.TimeoutError:
            pass
        # 触发微批
        if buf and (len(buf) >= BATCH or time.time() - last >= WAIT):
            for e in buf:
                s = DB.get(e.tracking_no)
                if not s:
                    # 新票创建：挂给 alice（演示），真实场景按授权来
                    s = Shipment(e.tracking_no, "alice", "未知商品", e.status, e.location, [])
                    DB[e.tracking_no] = s
                # 更新对象
                s.status = e.status
                s.location = e.location
                s.history.append(f"{e.status} @ {e.location} ({e.ts})")
            # 重建节点与索引（数据量小这样最简单；大规模可做增量 insert_nodes）
            _nodes = [shipment_to_node(s) for s in DB.values()]
            KW_INDEX, VEC_INDEX = build_indexes(_nodes)
            buf.clear()
            last = time.time()
            # 简单的缓存失效策略：清检索缓存，让新状态更快被看到
            RETR_CACHE.clear()

# ========== 7) FastAPI 接口 ==========
app = FastAPI(title="Realtime Shipping Query (LlamaIndex + LangChain)")

@app.on_event("startup")
async def _startup():
    asyncio.create_task(index_updater())

@app.get("/query")
async def query_endpoint(
    q: str = Query(..., description="问题：可填运单号或自然语言"),
    k: int = Query(5, ge=1, le=10),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    customer_id = auth(x_api_key)
    nodes = route_and_retrieve(customer_id, q, k)
    answer = answer_with_cache(customer_id, q, nodes)
    return {
        "customer_id": customer_id,
        "question": q,
        "hits": [n.node.metadata for n in nodes],
        "answer": answer,
    }

@app.post("/event")
async def event_endpoint(
    e: ShipEvent = Body(...),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    # 鉴权：只有属于这家客户的单，或你在真实系统中做服务端鉴权
    _ = auth(x_api_key)
    await EVENT_Q.put(e)
    return {"queued": True, "tracking_no": e.tracking_no}
