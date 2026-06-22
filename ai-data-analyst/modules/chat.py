from openai import OpenAI
import streamlit as st

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

    # -------- FULL DATASET SEARCH --------
    cleaned_query = (
        question_lower
        .replace("what is", "")
        .replace("what are", "")
        .replace("show me", "")
        .replace("find", "")
        .replace("give", "")
        .replace("details of", "")
        .replace("details about", "")
        .replace("movie", "")
        .replace("?", "")
        .strip()
    )

    search_words = [w for w in cleaned_query.split() if len(w) > 2]

    import pandas as pd

    if search_words:
      mask = pd.Series(False, index=df.index)

    for col in all_cols:
        col_values = df[col].fillna("").astype(str).str.lower()

        col_mask = col_values.str.contains(
            search_words[0],
            case=False,
            na=False
        )

        for word in search_words[1:]:
            col_mask = col_mask & col_values.str.contains(
                word,
                case=False,
                na=False
            )

        mask = mask | col_mask

        matched_rows = df[mask]

        if not matched_rows.empty:
            row = matched_rows.iloc[0]

            # If user asks for a specific column, return that column
            for col in all_cols:
                if col.lower().replace("_", " ") in question_lower:
                    return f"The {col} is {row[col]}."

            # Otherwise return full matching row details
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
    # =========================
    # 🤖 LLM fallback
    # =========================

    prompt = f"""
You are a data analyst.

Dataset:
{df.head(20).to_string()}

Answer clearly and correctly.

Question: {question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content