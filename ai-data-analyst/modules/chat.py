from openai import OpenAI
import streamlit as st
import pandas as pd

client = OpenAI(
    api_key=st.secrets["MY_OPENAI_KEY"].strip()
)


# 🔹 AI reasoning (WHY explanations)
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


# 🔹 Smart Chat with Data (dynamic + accurate)
def chat_with_data(df, question):
    question_lower = question.lower().strip()

    numeric_cols = df.select_dtypes(include='number').columns
    all_cols = list(df.columns)

    # -------- SMART ENTITY SEARCH --------
    stop_words = {
        "what", "is", "the", "of", "a", "an", "in", "on", "for", "to",
        "movie", "show", "series", "film", "release", "date", "year",
        "details", "about", "give", "find", "tell", "me"
    }

    words = [
        w.strip(" ?.,!").lower()
        for w in question.split()
        if w.strip(" ?.,!").lower() not in stop_words
    ]

    entity_query = " ".join(words).strip()

    if entity_query:
        mask = pd.Series(False, index=df.index)

        for col in all_cols:
            col_values = df[col].fillna("").astype(str).str.lower()
            mask = mask | col_values.str.contains(
                entity_query,
                case=False,
                na=False,
                regex=False
            )

        matched_rows = df[mask]

        if not matched_rows.empty:
            row = matched_rows.iloc[0]

            if "release" in question_lower:
                if "release_date" in df.columns:
                    return f"The release date of {row.get('title', entity_query)} is {row['release_date']}."
                if "release_year" in df.columns:
                    return f"The release year of {row.get('title', entity_query)} is {row['release_year']}."

            for col in all_cols:
                readable_col = col.lower().replace("_", " ")
                if readable_col in question_lower:
                    return f"The {readable_col} of {row.get('title', entity_query)} is {row[col]}."

            details = []
            for col in all_cols:
                details.append(f"{col}: {row[col]}")

            return "Match found:\n" + "\n".join(details)

    # -------- NUMERIC ANALYSIS --------
    target_col = None
    for col in numeric_cols:
        if col.lower() in question_lower:
            target_col = col
            break

    if target_col is None and len(numeric_cols) > 0:
        target_col = numeric_cols[0]

    if target_col is not None:
        label_col = None
        for col in all_cols:
            if col not in numeric_cols:
                label_col = col
                break

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