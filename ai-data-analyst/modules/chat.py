from openai import OpenAI
import streamlit as st
import pandas as pd
import re

client = OpenAI(
    api_key=st.secrets["MY_OPENAI_KEY"].strip()
)


def generate_reasoning(insights):
    prompt = f"""
You are a data analyst.

Explain WHY these insights might be happening.

Rules:
- Keep it short
- No steps
- Simple explanation

Insights:
{insights}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def chat_with_data(df, question):
    question_lower = question.lower().strip()

    numeric_cols = df.select_dtypes(include="number").columns
    all_cols = list(df.columns)

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    stop_words = {
        "what", "is", "the", "of", "a", "an", "in", "on", "for", "to",
        "movie", "show", "series", "film", "release", "date", "year",
        "details", "about", "give", "find", "tell", "me", "which", "who"
    }

    # Detect requested column like duration, release_year, country, director
    requested_col = None
    for col in all_cols:
        readable_col = col.lower().replace("_", " ")
        if readable_col in question_lower:
            requested_col = col
            break

    # Build entity query by removing stop words and requested column words
    words = []
    requested_col_words = set(clean_text(requested_col).split()) if requested_col else set()

    for w in question.split():
        w_clean = clean_text(w)

        if (
            w_clean
            and w_clean not in stop_words
            and w_clean not in requested_col_words
        ):
            words.append(w_clean)

    entity_query = " ".join(words).strip()

    # -------- FULL DATASET ENTITY SEARCH --------
    if entity_query:
        row_text = df.fillna("").astype(str).agg(" ".join, axis=1).apply(clean_text)

        # Search all entity words anywhere in the row
        mask = pd.Series(True, index=df.index)

        for word in entity_query.split():
            mask = mask & row_text.str.contains(word, case=False, na=False, regex=False)

        matched_rows = df[mask]

        if not matched_rows.empty:
            row = matched_rows.iloc[0]

            title_value = row["title"] if "title" in df.columns else entity_query

            # If specific column is requested
            if requested_col:
                return f"The {requested_col} of {title_value} is {row[requested_col]}."

            # Release handling
            if "release" in question_lower:
                if "release_date" in df.columns:
                    return f"The release date of {title_value} is {row['release_date']}."
                if "release_year" in df.columns:
                    return f"The release year of {title_value} is {row['release_year']}."

            details = [f"{col}: {row[col]}" for col in all_cols]
            return "Match found:\n" + "\n".join(details)

        return f"No matching record found for '{entity_query}' in the uploaded dataset."

    # -------- NUMERIC ANALYSIS --------
    target_col = None

    for col in numeric_cols:
        if col.lower() in question_lower:
            target_col = col
            break

    if target_col is None and len(numeric_cols) > 0:
        target_col = numeric_cols[0]

    if target_col is not None:
        label_col = next((col for col in all_cols if col not in numeric_cols), None)

        if "lowest" in question_lower or "minimum" in question_lower:
            row = df.loc[df[target_col].idxmin()]
            return f"The lowest {target_col} is {row[target_col]} on {label_col} {row[label_col]}."

        if "highest" in question_lower or "maximum" in question_lower:
            row = df.loc[df[target_col].idxmax()]
            return f"The highest {target_col} is {row[target_col]} on {label_col} {row[label_col]}."

        if "total" in question_lower or "sum" in question_lower:
            return f"The total {target_col} is {df[target_col].sum()}."

        if "average" in question_lower or "mean" in question_lower:
            return f"The average {target_col} is {round(df[target_col].mean(), 2)}."

    # -------- LLM FALLBACK --------
    prompt = f"""
You are a data analyst.

Dataset sample:
{df.head(20).to_string()}

Question:
{question}

Answer clearly. If the answer is not visible in the sample, say that more data search is needed.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content