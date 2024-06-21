import fastapi
from sentence_transformers import SentenceTransformer
import pydantic
import logging

logging.basicConfig(level=logging.INFO)
model = SentenceTransformer("all-MiniLM-L6-v2")

app = fastapi.FastAPI(docs_url=None, redoc_url=None)


class SimilarityQuery(pydantic.BaseModel):
    text1: str
    text2: str


@app.post("/calculate_similarity")
def calculate_similarity(data: SimilarityQuery):
    # 将文本编码为向量表示
    logging.info("Calculating similarity:\n\nText 1:\n%s\n\nText 2:\n%s\n\n", data.text1, data.text2)
    embedding1 = model.encode([data.text1])
    embedding2 = model.encode([data.text2])

    # 计算余弦相似度
    similarity = model.similarity(embedding1, embedding2)
    similarity = float(similarity[0][0])
    logging.info("Similarity formatted as float: %.2f", similarity)

    return {"similarity": similarity}
