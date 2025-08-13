"""
========================================================
面向 RAG 检索链路的四层缓存示例（Redis）+ 局部失效：
1) 嵌入缓存 (Embedding Cache)
2) 检索候选缓存 (Retrieval Candidate Cache)
3) Rerank 缓存 (Cross-Encoder Re-ranking Cache)
4) LLM 输出缓存 (LLM Output Cache)

设计目标（从下到上，越早命中越省时省钱）：
- 嵌入缓存：避免重复向量化（最耗算力）
- 检索候选缓存：避免重复向量库 TopK 搜索
- Rerank 缓存：避免重复 Cross-Encoder 精排（最贵的一跳）
- LLM 输出缓存：热门/FAQ 直接复用最终答案

注意：
- 本文件包含可运行骨架；rerank_fn / llm_fn 用占位实现以便快速接入。
- 生产中请替换为你自己的向量库检索、Cross-Encoder 精排与 LLM 调用。
========================================================
"""

import hashlib
import json
import time
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from redis import Redis
from sentence_transformers import SentenceTransformer

# =========================
# Redis 连接（统一缓存层）
# =========================
# 选择 Redis 的原因：
# - 内存级读写，延迟低
# - 支持 TTL、集合/哈希/有序集合，便于维护反查表做“局部失效”
# - 进程/服务间共享方便
r = Redis(host="localhost", port=6379, decode_responses=False)

# =========================
# 全局模型 / 版本标识
# =========================
EMB_MODEL_ID = "all-MiniLM-L6-v2"
EMB_VERSION = "v1"         # 嵌入切块/预处理/模型升级时 bump
RETR_VERSION = "v1"        # 检索策略变更（过滤/TopK）时 bump
RR_MODEL_ID = "ce-msmarco-MiniLM-L-6-v2"
RR_VERSION = "v2"          # 精排模板/截断/归一化策略升级时 bump
LLM_ID = "llm-mini-instruct"
LLM_VERSION = "v1"         # Prompt 模板/系统指令/温度等变更时 bump

# ================
# 小工具函数
# ================
def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))

def norm_text(text: str) -> str:
    """统一化文本，避免“看起来一样其实不同”的字符串导致缓存未命中"""
    return " ".join(text.lower().split())

def dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

def loads(b: bytes) -> Any:
    return json.loads(b.decode("utf-8"))

def stable_doc_list(doc_ids: Sequence[str]) -> List[str]:
    """为 Rerank 缓存稳定候选集合：排序后得到确定性的列表"""
    return sorted(doc_ids)

# =========================
# 1) 嵌入缓存（Embedding Cache）
#   - 放：向量（可选元信息）
#   - 为什么：向量化最耗时/耗算力；同文本+同配置应完全复用
#   - TTL：长（天~月）；升级模型/切块时 bump EMB_VERSION 自然失效
# =========================
emb_model = SentenceTransformer(EMB_MODEL_ID)

def emb_cache_key(text: str) -> str:
    sig = f"{EMB_MODEL_ID}::{norm_text(text)}::{EMB_VERSION}"
    return f"emb:{sha256_text(sig)}"

def get_embedding(text: str) -> List[float]:
    k = emb_cache_key(text)
    v = r.get(k)
    if v:
        return loads(v)
    vec = emb_model.encode(text, normalize_embeddings=True).tolist()
    r.set(k, dumps(vec))  # 可加 ex=30*24*3600 做长 TTL
    return vec

# =========================
# 2) 检索候选缓存（Retrieval Candidate Cache）
#   - 放：TopK 的 doc_id 列表（也可带相似度）
#   - 为什么：热门/重复查询无需每次都命中向量库
#   - TTL：短~中（30s~10min）；实时数据多时更短
# =========================
def retr_cache_key(q_vec: List[float], filt: Dict[str, Any], k: int) -> str:
    # 生产中建议用“query 原文”的 embedding key 作为签名的一部分
    payload = dumps({"v": RETR_VERSION, "k": k, "filt": filt, "vec": q_vec})
    return f"ret:{sha256_bytes(payload)}"

def cached_retrieve(
    query_text: str,
    k: int,
    filt: Dict[str, Any],
    search_fn: Callable[[List[float], Dict[str, Any], int], List[str]],
    ttl_seconds: int = 300,
) -> List[str]:
    q_vec = get_embedding(query_text)
    ck = retr_cache_key(q_vec, filt, k)
    hit = r.get(ck)
    if hit:
        return loads(hit)  # List[str] doc_ids

    doc_ids = search_fn(q_vec, filt, k)
    r.setex(ck, ttl_seconds, dumps(doc_ids))
    return doc_ids

# =========================
# 3) Rerank 缓存（Cross-Encoder）
#   - 放：[(doc_id, score), ...]（已按分数排序）
#   - 为什么：Cross-Encoder 精排最贵；重复的“query+候选集”应直接复用
#   - 关键：候选集合的“顺序/内容”必须稳定 → 对 doc_ids 做排序，或按 (score, id) 归一
#   - TTL：短~中（分钟~小时）；文档更新时需“局部失效”
#   - 局部失效：为每个 doc_id 维护反查集合 inv:rr:doc:{doc_id} → {rr_keys...}
# =========================
def rr_cache_key(query_text: str, doc_ids: Sequence[str]) -> str:
    stable_ids = stable_doc_list(doc_ids)
    payload = dumps(
        {"rr_model": RR_MODEL_ID, "rr_ver": RR_VERSION, "q": norm_text(query_text), "docs": stable_ids}
    )
    return f"rr:{sha256_bytes(payload)}"

def _link_rr_invalidation(rr_key: str, doc_ids: Iterable[str]) -> None:
    """把 rr_key 记录到每个 doc 的反查集合，便于后续局部失效"""
    pipe = r.pipeline(transaction=False)
    for did in doc_ids:
        pipe.sadd(f"inv:rr:doc:{did}", rr_key)
    pipe.execute()

def invalidate_rerank_by_docs(doc_ids: Iterable[str]) -> int:
    """根据 doc_id 集合删除相关的 Rerank 缓存，返回删除条数"""
    to_delete = set()
    # 收集所有需要删的 rr keys
    for did in doc_ids:
        keyset = r.smembers(f"inv:rr:doc:{did}")
        if keyset:
            to_delete.update(keyset)
    # 批量删除 rr keys + 清理反查集合
    deleted = 0
    if to_delete:
        pipe = r.pipeline(transaction=False)
        for k in to_delete:
            pipe.delete(k)
            deleted += 1
        # 清理对应的 inv sets
        for did in doc_ids:
            pipe.delete(f"inv:rr:doc:{did}")
        pipe.execute()
    return deleted

def cached_rerank(
    query_text: str,
    doc_ids: Sequence[str],
    rerank_fn: Callable[[str, Sequence[str]], List[Tuple[str, float]]],
    ttl_seconds: int = 900,
) -> List[Tuple[str, float]]:
    rrk = rr_cache_key(query_text, doc_ids)
    hit = r.get(rrk)
    if hit:
        return loads(hit)  # [(doc_id, score), ...]

    # 未命中 → 跑精排
    scored = rerank_fn(query_text, doc_ids)  # 需返回按分数降序的列表
    r.setex(rrk, ttl_seconds, dumps(scored))
    _link_rr_invalidation(rrk, doc_ids)  # 建立局部失效反查关系
    return scored

# =========================
# 4) LLM 输出缓存（最终答案）
#   - 放：最终文本答案（可包含结构化字段：answer、citations、confidence 等）
#   - 为什么：热门问题/FAQ 直接秒回；节省 LLM 成本与时延
#   - Key：必须绑定 prompt template + query + 引用摘要 + llm_id/version
#   - TTL：中（小时~天）；含实时内容时适当缩短
#   - 局部失效：若答案包含引用 doc_ids，同样维护反查集合 inv:llm:doc:{doc_id}
# =========================
def llm_cache_key(
    prompt_tpl_id: str,
    query_text: str,
    citations_digest: str,
) -> str:
    payload = dumps(
        {
            "llm": LLM_ID,
            "llm_ver": LLM_VERSION,
            "tpl": prompt_tpl_id,
            "q": norm_text(query_text),
            "cits": citations_digest,
        }
    )
    return f"llm:{sha256_bytes(payload)}"

def _link_llm_invalidation(llm_key: str, cited_doc_ids: Iterable[str]) -> None:
    pipe = r.pipeline(transaction=False)
    for did in cited_doc_ids:
        pipe.sadd(f"inv:llm:doc:{did}", llm_key)
    pipe.execute()

def invalidate_llm_by_docs(doc_ids: Iterable[str]) -> int:
    to_delete = set()
    for did in doc_ids:
        keyset = r.smembers(f"inv:llm:doc:{did}")
        if keyset:
            to_delete.update(keyset)
    deleted = 0
    if to_delete:
        pipe = r.pipeline(transaction=False)
        for k in to_delete:
            pipe.delete(k)
            deleted += 1
        for did in doc_ids:
            pipe.delete(f"inv:llm:doc:{did}")
        pipe.execute()
    return deleted

def cached_llm_answer(
    prompt_tpl_id: str,
    query_text: str,
    citations: Sequence[str],  # doc_id 列表（生成答案必须基于这些引用）
    llm_fn: Callable[[str, str, Sequence[str]], Dict[str, Any]],
    ttl_seconds: int = 6 * 3600,
) -> Dict[str, Any]:
    # 引用摘要必须稳定（排序后做摘要）
    stable_cits = stable_doc_list(citations)
    cit_digest = sha256_text(json.dumps(stable_cits, ensure_ascii=False))
    llmk = llm_cache_key(prompt_tpl_id, query_text, cit_digest)

    hit = r.get(llmk)
    if hit:
        return loads(hit)  # {"answer": "...", "citations": [...], ...}

    result = llm_fn(prompt_tpl_id, query_text, stable_cits)
    # 建议在 result 中包含 citations，便于审计与失效
    r.setex(llmk, ttl_seconds, dumps(result))
    _link_llm_invalidation(llmk, stable_cits)
    return result

# =======================================================
# 占位实现（替换成你的向量检索 / rerank / LLM 调用）
# =======================================================
def dummy_vector_search(query_vec: List[float], filt: Dict[str, Any], k: int) -> List[str]:
    # TODO: 替换为 Qdrant/Weaviate/Milvus/FAISS 的真实检索
    # 这里只返回固定示例
    return ["doc_42", "doc_7", "doc_3"][:k]

def dummy_rerank(query_text: str, doc_ids: Sequence[str]) -> List[Tuple[str, float]]:
    # TODO: 替换为 Cross-Encoder 推理（如 transformers 的 CE 模型）
    # 这里按 doc_id 做个伪分数并排序，示意结构
    scored = [(d, 1.0 - (hash(d + query_text) % 100) / 1000.0) for d in doc_ids]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def dummy_llm(prompt_tpl_id: str, query_text: str, citations: Sequence[str]) -> Dict[str, Any]:
    # TODO: 替换为真实 LLM 调用（OpenAI/本地模型），带上模板与引用
    answer = f"[{prompt_tpl_id}] A concise answer for '{query_text}'. Based on {list(citations)}."
    return {"answer": answer, "citations": list(citations), "model": LLM_ID, "ts": int(time.time())}

# =======================================================
# 示例链路（组合调用）
# =======================================================
def handle_query(query_text: str) -> Dict[str, Any]:
    # 1) 检索（命中“嵌入缓存/检索缓存”）
    doc_ids = cached_retrieve(query_text, k=5, filt={"lang": "zh"}, search_fn=dummy_vector_search)

    # 2) 精排（命中“Rerank 缓存”）
    reranked = cached_rerank(query_text, doc_ids, rerank_fn=dummy_rerank)
    top_docs = [d for d, _ in reranked[:3]]  # 例如只取 Top3 作为引用

    # 3) 生成（命中“LLM 输出缓存”）
    result = cached_llm_answer(
        prompt_tpl_id="ans_v1_strict_cited",
        query_text=query_text,
        citations=top_docs,
        llm_fn=dummy_llm,
    )
    return result

# =======================================================
# 局部失效示例
# =======================================================
def invalidate_on_docs_changed(changed_doc_ids: Iterable[str]) -> Dict[str, int]:
    """
    当文档被更新/下线时调用：
    - 删除相关的 Rerank 缓存
    - 删除相关的 LLM 输出缓存
    检索候选缓存（ret:*）可用短 TTL，自然失效；如需更快，可扩展反查表策略。
    """
    rr_deleted = invalidate_rerank_by_docs(changed_doc_ids)
    llm_deleted = invalidate_llm_by_docs(changed_doc_ids)
    return {"rerank_deleted": rr_deleted, "llm_deleted": llm_deleted}

# =========================
# 快速自测
# =========================
if __name__ == "__main__":
    q = "如何降低 RAG 的延迟与成本？"
    print("第一次：完整链路（多为未命中）")
    print(handle_query(q))

    print("\n第二次：应大量命中缓存（更快）")
    print(handle_query(q))

    print("\n文档 doc_7 发生更新 → 触发局部失效")
    print(invalidate_on_docs_changed(["doc_7"]))

    print("\n再次查询：涉及 doc_7 的 Rerank/LLM 缓存会重算，其它仍命中")
    print(handle_query(q))
