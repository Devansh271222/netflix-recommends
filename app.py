import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="centered"
)

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")

    # Handle missing text values used by the recommender
    for col in ["director", "cast", "country", "listed_in"]:
        df[col] = df[col].fillna("")

    # Weighted feature engineering:
    # Director ×2, Genre ×3, Cast ×1, Country ×1
    df["weighted_tags"] = (
        df["director"] + " " +
        df["director"] + " " +
        df["cast"] + " " +
        df["country"] + " " +
        df["listed_in"] + " " +
        df["listed_in"] + " " +
        df["listed_in"]
    )

    return df


@st.cache_resource
def build_model(texts):
    tfidf = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]+\b"
    )

    matrix = tfidf.fit_transform(texts)

    # More memory-efficient than creating a full dense
    # 8807 × 8807 cosine-similarity matrix.
    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )
    model.fit(matrix)

    return model, matrix


df = load_data()
model, matrix = build_model(df["weighted_tags"])

# Case-insensitive title lookup
title_lookup = pd.Series(
    df.index,
    index=df["title"].str.lower()
).drop_duplicates()


def recommend(title, n=10):
    key = title.lower().strip()

    if key not in title_lookup:
        return []

    idx = title_lookup[key]

    distances, indices_found = model.kneighbors(
        matrix[idx],
        n_neighbors=n + 1
    )

    recommendations = []

    for distance, movie_idx in zip(distances[0], indices_found[0]):
        if movie_idx == idx:
            continue

        similarity = 1 - distance

        recommendations.append({
            "title": df.iloc[movie_idx]["title"],
            "type": df.iloc[movie_idx]["type"],
            "release_year": df.iloc[movie_idx]["release_year"],
            "genres": df.iloc[movie_idx]["listed_in"],
            "similarity": round(float(similarity), 3)
        })

    return recommendations[:n]


# ---------------- UI ----------------

st.title("🎬 Netflix Movie & TV Show Recommender")
st.write(
    "Enter a Netflix title and get recommendations based on "
    "genre, director, cast, and country."
)

selected_title = st.selectbox(
    "Choose a title",
    sorted(df["title"].dropna().unique())
)

if st.button("Recommend"):
    results = recommend(selected_title, 10)

    st.subheader(f"Because you liked: {selected_title}")

    if not results:
        st.error("Movie not found.")
    else:
        for i, item in enumerate(results, start=1):
            st.markdown(
                f"### {i}. {item['title']}"
            )
            st.write(
                f"**Type:** {item['type']}  |  "
                f"**Year:** {item['release_year']}  |  "
                f"**Similarity:** {item['similarity']}"
            )
            st.write(f"**Genres:** {item['genres']}")
            st.divider()

st.caption(
    "ML approach: TF-IDF + cosine-distance nearest neighbors. "
    "Genre is weighted 3× and director 2×."
)
