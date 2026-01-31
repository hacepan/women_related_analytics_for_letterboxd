import io
import os
import json
import asyncio
import pandas as pd
import httpx
from collections import Counter
from fastapi import FastAPI, File, UploadFile
from dotenv import load_dotenv

load_dotenv()
TMDB_TOKEN = os.getenv("TMDB_READ_TOKEN")
CACHE_FILE = "movie_cache.csv"

CACHE_COLS = ["Name", "Year", "URI", "TMDB_ID", "Directors"]

if not os.path.exists(CACHE_FILE):
    pd.DataFrame(columns=CACHE_COLS).to_csv(CACHE_FILE, index=False)

app = FastAPI()
semaphore = asyncio.Semaphore(10)

async def get_movie_data(client: httpx.AsyncClient, name: str, year: int):
    """Fetches TMDB ID and Director info."""
    async with semaphore:
        headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
        try:
            # 1. Get Movie ID
            search_res = await client.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"query": name, "primary_release_year": year},
                headers=headers
            )
            results = search_res.json().get("results")
            if not results: return None, []

            movie_id = results[0]["id"]

            credits_res = await client.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits", headers=headers)
            crew = credits_res.json().get("crew", [])

            directors = [
                {"name": p["name"], "gender": "Female" if p["gender"] == 1 else "Other"}
                for p in crew if p["job"] == "Director"
            ]
            return movie_id, directors
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            return None, []

@app.post("/analyze-letterboxd")
async def analyze_letterboxd(file: UploadFile = File(...)):
    # 1. Load and Clean User Data
    df_user = pd.read_csv(io.BytesIO(await file.read()))
    df_user["Rating"] = pd.to_numeric(df_user["Rating"], errors="coerce")
    df_user = df_user.dropna(subset=["Name", "Rating"])
    df_user["key"] = df_user["Name"].astype(str) + df_user["Year"].astype(str)

    # 2. Handle Cache with Integer Formatting
    if not os.path.exists(CACHE_FILE):
        pd.DataFrame(columns=CACHE_COLS).to_csv(CACHE_FILE, index=False)

    # Force TMDB_ID to be read as a nullable integer
    cache_df = pd.read_csv(CACHE_FILE, dtype={"TMDB_ID": "Int64"})
    cache_df["key"] = cache_df["Name"].astype(str) + cache_df["Year"].astype(str)

    known_mask = df_user["key"].isin(cache_df["key"])
    to_fetch = df_user[~known_mask]

    if not to_fetch.empty:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            tasks = [
                get_movie_data(client, row["Name"], row["Year"])
                for _, row in to_fetch.iterrows()
            ]
            api_results = await asyncio.gather(*tasks)

            new_rows = []
            for i, (_, row) in enumerate(to_fetch.iterrows()):
                m_id, directors = api_results[i]
                new_rows.append(
                    {
                        "Name": row["Name"],
                        "Year": row["Year"],
                        "URI": row["Letterboxd URI"],
                        "TMDB_ID": int(m_id) if m_id else None,  # Cast to int here
                        "Directors": json.dumps(directors),
                    }
                )

            # Combine and save
            combined_cache = pd.concat(
                [pd.read_csv(CACHE_FILE), pd.DataFrame(new_rows)]
            )
            # Ensure the column is Int64 before saving
            combined_cache["TMDB_ID"] = combined_cache["TMDB_ID"].astype("Int64")
            combined_cache.to_csv(CACHE_FILE, index=False)

    # 3. Merge & Process
    full_cache = pd.read_csv(CACHE_FILE, dtype={"TMDB_ID": "Int64"})
    full_cache["Directors"] = full_cache["Directors"].apply(json.loads)
    merged = df_user.merge(
        full_cache[["Name", "Year", "TMDB_ID", "Directors"]], on=["Name", "Year"]
    )

    female_director_list = []
    female_ratings = []
    other_ratings = []

    for _, row in merged.iterrows():
        is_female_movie = False
        for d in row['Directors']:
            if d['gender'] == "Female":
                female_director_list.append(d['name'])
                female_ratings.append(row['Rating'])
                is_female_movie = True
            else:
                other_ratings.append(row['Rating'])

    director_counts = Counter(female_director_list)
    top_director_data = None
    if director_counts:
        top_name, count = director_counts.most_common(1)[0]
        top_director_data = {"name": top_name, "watch_count": count}

    female_directed_df = merged[
        merged["Directors"].apply(
            lambda d_list: any(d["gender"] == "Female" for d in d_list)
        )
    ]

    top_female_picks = []
    if not female_directed_df.empty:
        max_rating = female_directed_df["Rating"].max()
        best_df = female_directed_df[female_directed_df["Rating"] == max_rating].copy()

        best_df["TMDB_ID"] = best_df["TMDB_ID"].fillna(0).astype(int)

        top_female_picks = best_df[
            ["Name", "Year", "TMDB_ID", "Rating", "Letterboxd URI"]
        ].to_dict(orient="records")

    return {
        "stats": {
            "female_avg": sum(female_ratings)/len(female_ratings) if female_ratings else 0,
            "other_avg": sum(other_ratings)/len(other_ratings) if other_ratings else 0,
            "most_watched_female_director": top_director_data
        },
        "best_rated_female_directed": top_female_picks
    }