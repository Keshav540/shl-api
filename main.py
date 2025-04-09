from fastapi import FastAPI, HTTPException
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# -----------------------------
# Scrape SHL Catalog
# -----------------------------
def fetch_shl_catalog():
    url = "https://www.shl.com/solutions/products/product-catalog/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching SHL catalog: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")
    products = []
    product_tiles = soup.find_all("div", class_="custom__table-responsive")
    if not product_tiles:
        product_tiles = soup.find_all("li", class_="product-item")

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

df_assessments = fetch_shl_catalog()
if df_assessments.empty:
    print("Warning: No data fetched from SHL.")

# -----------------------------
# Recommendation Logic
# -----------------------------
def recommend_assessments(query: str, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    names = df["Assessment Name"].tolist()
    corpus = [query] + names
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    df_copy = df.copy()
    df_copy["Score"] = sims

    # Always return top_n regardless of score
    return df_copy.sort_values(by="Score", ascending=False).head(top_n)



# -----------------------------
# FastAPI Endpoint
# -----------------------------
@app.get("/query")
def query_recommendations(q: str, top_n: int = 10):
    if df_assessments.empty:
        raise HTTPException(status_code=404, detail="No product data available")
    try:
        df_result = recommend_assessments(q, df_assessments, top_n=len(df_assessments))

        # Add scores and sort
        df_result = df_result.sort_values(by="Score", ascending=False)

        # Keep only top_n
        df_result = df_result.head(top_n)

        results = df_result.drop(columns=["Score"]).to_dict(orient="records")
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


