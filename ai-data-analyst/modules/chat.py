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
    question_lower = question.lower()

    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns

    # Detect numeric column
    target_col = None
    for col in numeric_cols:
        if col.lower() in question_lower:
            target_col = col
            break

    if target_col is None and len(numeric_cols) > 0:
        target_col = numeric_cols[0]

    # Detect category column
    category_col = None
    for col in categorical_cols:
        if col.lower() in question_lower:
            category_col = col
            break

    # Label column (like date)
    label_col = None
    for col in df.columns:
        if col not in numeric_cols:
            label_col = col
            break

    # =========================
    # OPERATIONS
    # =========================

    if "lowest" in question_lower or "minimum" in question_lower:
        row = df.loc[df[target_col].idxmin()]
        return f"The {label_col} with the lowest {target_col} is {row[label_col]} ({row[target_col]})."

    if "highest" in question_lower or "maximum" in question_lower:
        row = df.loc[df[target_col].idxmax()]
        return f"The {label_col} with the highest {target_col} is {row[label_col]} ({row[target_col]})."

    if "total" in question_lower or "sum" in question_lower:
        total = df[target_col].sum()
        return f"The total {target_col} is {total}."

    if "average" in question_lower or "mean" in question_lower:
        avg = df[target_col].mean()
        return f"The average {target_col} is {round(avg, 2)}."

    if category_col and ("highest" in question_lower or "best" in question_lower):
        grouped = df.groupby(category_col)[target_col].sum()
        top = grouped.idxmax()
        return f"The {category_col} with the highest {target_col} is {top}."

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