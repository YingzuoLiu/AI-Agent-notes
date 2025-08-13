"""
多模态（文本 + 图片）检索示例
使用 CLIP 统一编码文本与图片，将 embedding 存入 FAISS，
并通过 LlamaIndex 管理节点、存储和检索。
"""

from pathlib import Path
from typing import List, Union
import faiss
from PIL import Image

from sentence_transformers import SentenceTransformer

# ===== LlamaIndex 相关模块 =====
from llama_index.core import VectorStoreIndex, StorageContext, Settings  # 管理索引、存储上下文
from llama_index.core.schema import TextNode, NodeWithScore               # 节点结构与检索结果结构
from llama_index.vector_stores.faiss import FaissVectorStore              # 向量存储适配器（FAISS 后端）

# ========= 1) 统一的 CLIP 编码器（文本 & 图像） =========
class CLIPEncoder:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        # sentence-transformers 会自动加载对应的 CLIP 模型
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_text(self, text: str):
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec

    def encode_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        vec = self.model.encode(img, normalize_embeddings=True)
        return vec

# ========= 2) 构造多模态节点 =========
def build_text_nodes(encoder: CLIPEncoder, data_dir: str = "data") -> List[TextNode]:
    nodes = []
    for p in Path(data_dir).glob("**/*.txt"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        emb = encoder.encode_text(text)
        # LlamaIndex TextNode：存储文本、元数据、embedding
        node = TextNode(
            text=text,
            metadata={"modality": "text", "path": str(p)},
            embedding=emb.tolist(),   # embedding 直接传入节点，避免重复计算
        )
        nodes.append(node)
    return nodes

def build_image_nodes(encoder: CLIPEncoder, img_dir: str = "images") -> List[TextNode]:
    nodes = []
    for p in list(Path(img_dir).glob("**/*.jpg")) + \
              list(Path(img_dir).glob("**/*.png")) + \
              list(Path(img_dir).glob("**/*.webp")):
        emb = encoder.encode_image(str(p))
        caption = p.stem.replace("_", " ")
        # LlamaIndex TextNode 也可以存图片元数据，text 可放图片标题
        node = TextNode(
            text=f"[IMAGE] {caption}",
            metadata={"modality": "image", "path": str(p)},
            embedding=emb.tolist(),
        )
        nodes.append(node)
    return nodes

# ========= 3) 建立 FAISS + LlamaIndex 索引 =========
def build_index(nodes: List[TextNode], dim: int) -> VectorStoreIndex:
    # ① FAISS 原生索引
    faiss_index = faiss.IndexFlatIP(dim)  # 内积（配合归一化≈余弦相似度）
    # ② LlamaIndex 的 FaissVectorStore 适配器：封装 FAISS 作为后端
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    # ③ StorageContext：LlamaIndex 存储上下文，统一管理 vector_store、docstore 等
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # ④ VectorStoreIndex：LlamaIndex 主索引对象，封装检索逻辑
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    return index

# ========= 4) 自定义多模态检索器 =========
class MultiModalRetriever:
    def __init__(self, index: VectorStoreIndex, encoder: CLIPEncoder, top_k: int = 5):
        self.index = index
        self.encoder = encoder
        self.top_k = top_k

    def query(self, query: Union[str, Path]) -> List[NodeWithScore]:
        # 文本 or 图片查询 → 编码成统一向量
        if isinstance(query, (str,)):
            q_vec = self.encoder.encode_text(query).tolist()
        else:
            q_vec = self.encoder.encode_image(str(query)).tolist()

        # 使用 LlamaIndex 的底层 vector_store 做向量相似度检索
        vs = self.index._vector_store  # 注意：这里直接访问底层存储
        results = vs.query(query_embedding=q_vec, similarity_top_k=self.top_k)

        # LlamaIndex 返回的 VectorStoreQueryResult 转换为 NodeWithScore
        nodes = []
        for node, score in zip(results.nodes, results.similarities or []):
            nodes.append(NodeWithScore(node=node, score=float(score)))
        return nodes

# ========= 5) DEMO：text→image/text & image→image/text =========
if __name__ == "__main__":
    encoder = CLIPEncoder("clip-ViT-B-32")
    text_nodes = build_text_nodes(encoder, "data")
    image_nodes = build_image_nodes(encoder, "images")
    all_nodes = text_nodes + image_nodes

    if not all_nodes:
        print("请在 ./data 放一些 .txt，或在 ./images 放一些图片再运行。")
        exit(0)

    # 用 LlamaIndex 建索引
    index = build_index(all_nodes, encoder.dim)

    retriever = MultiModalRetriever(index, encoder, top_k=5)

    # 文本查询
    q_text = "红色跑鞋，轻便，适合夜跑"
    print(f"\n[TEXT QUERY] {q_text}")
    for r in retriever.query(q_text):
        print(f"- score={r.score:.3f}  [{r.node.metadata.get('modality')}]  {r.node.metadata.get('path')}")

    # 图片查询
    sample_img = next((Path("images").glob("**/*.*")), None)
    if sample_img:
        print(f"\n[IMAGE QUERY] {sample_img}")
        for r in retriever.query(sample_img):
            print(f"- score={r.score:.3f}  [{r.node.metadata.get('modality')}]  {r.node.metadata.get('path')}")
