from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

class Recommendation(BaseModel):
    name: str
    url: str
    remote: str
    adaptive: str
    score: float

def fetch_shl_catalog() -> pd.DataFrame:
    url = "https://www.shl.com/solutions/products/product-catalog/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception:
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")
    products = []

    product_tiles = soup.find_all("tr")
    for tile in product_tiles:
        anchor = tile.find("a", href=True)
        if anchor:
            name = anchor.get_text(strip=True)
            link = anchor["href"]
            if not link.startswith("http"):
                link = "https://www.shl.com" + link
        else:
            name, link = "Unknown", "#"

        text = tile.get_text(" ", strip=True).lower()
        remote = "Yes" if "remote" in text else "No"
        adaptive = "Yes" if ("adaptive" in text or "irt" in text) else "No"

        products.append({
            "Assessment Name": name,
            "URL": link,
            "Remote Testing Support": remote,
            "Adaptive/IRT Support": adaptive
        })

    return pd.DataFrame(products)

def recommend_assessments(query: str, df: pd.DataFrame, top_n: int = 10) -> List[Recommendation]:
    names = df["Assessment Name"].tolist()
    vec = TfidfVectorizer(stop_words="english")
    name_vectors = vec.fit_transform(names)
    query_vector = vec.transform([query])
    sims = cosine_similarity(query_vector, name_vectors).flatten()
    idx = sims.argsort()[::-1][:top_n]
    result = df.iloc[idx].copy()
    result["Score"] = sims[idx]

    recommendations = []
    for _, row in result.iterrows():
        recommendations.append(Recommendation(
            name=row["Assessment Name"],
            url=row["URL"],
            remote=row["Remote Testing Support"],
            adaptive=row["Adaptive/IRT Support"],
            score=round(float(row["Score"]), 4)
        ))
    return recommendations

@app.get("/recommend", response_model=List[Recommendation])
def get_recommendations(query: str = Query(..., description="Job description or search text")):
    df = fetch_shl_catalog()
    if df.empty:
        return []
    return recommend_assessments(query, df)
