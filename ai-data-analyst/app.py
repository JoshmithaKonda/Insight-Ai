import streamlit as st
from modules.data_loader import load_data, analyze_data
from modules.visualizer import generate_charts
from modules.insights import generate_insights
from modules.chat import generate_reasoning, chat_with_data


st.set_page_config(
    page_title="InsightAI",
    page_icon="logo.png",
    layout="wide"
)

st.sidebar.image("logo.png", width=150)

#  Custom UI
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
h1 {
    text-align: center;
    color: white;
}
.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 2rem;
}
.upload-box {
    border: 2px dashed #4f46e5;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    background: rgba(255,255,255,0.03);
}
.section {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

#  Header
# Header with logo
header_col1, header_col2 = st.columns([1, 5])

with header_col1:
    st.image("logo.png", width=90)

with header_col2:
    st.markdown("<h1>Insight-AI</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">AI-powered data analysis & insights platform</p>',
        unsafe_allow_html=True
    )#  Header with logo
header_col1, header_col2 = st.columns([1, 5])

with header_col1:
    st.image("logo.png", width=90)

with header_col2:
    st.markdown("<h1>Insight-AI</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">AI-powered data analysis & insights platform</p>',
        unsafe_allow_html=True
    )
#  Upload
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload one or more datasets",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)
st.markdown('</div>', unsafe_allow_html=True)

#  Main App
if uploaded_files:
    dataframes = {}

    for file in uploaded_files:
        df = load_data(file)
        dataframes[file.name] = df
    st.write(f"📂 {len(dataframes)} dataset(s) uploaded")

    # Dataset selector
    selected_file = st.selectbox("Select Dataset", list(dataframes.keys()))
    df = dataframes[selected_file]

    #  Preview
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())
    st.markdown('</div>', unsafe_allow_html=True)

    #  Summary
    summary = analyze_data(df)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Data Summary")
    st.json(summary)
    st.markdown('</div>', unsafe_allow_html=True)

    #  KPI Cards
    col1, col2, col3 = st.columns(3)
    if "sales" in df.columns:
        col1.metric("Total Sales", int(df["sales"].sum()))
    if "profit" in df.columns:
        col2.metric("Average Profit", int(df["profit"].mean()))
    col3.metric("Total Records", len(df))

    #  Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🧠 Insights",
        "🤖 Explanation",
        "💬 Chat"
    ])

    #  Dashboard Tab
    with tab1:
        st.subheader("Interactive Dashboard")
        charts = generate_charts(df)
        for chart in charts:
            st.plotly_chart(chart, use_container_width=True)

    #  Insights Tab
    with tab2:
        st.subheader("Key Insights")
        insights = generate_insights(df)
        for ins in insights:
            st.write(f"• {ins}")

    #  Explanation Tab
    with tab3:
        st.subheader("AI Explanation")
        insights = generate_insights(df)
        reasoning = generate_reasoning(insights)
        st.write(reasoning)

    #  Chat Tab
    with tab4:
        st.subheader("💬 Chat with Data")
        user_question = st.text_input("Ask something about your data:")

        if user_question:
            answer = chat_with_data(df, user_question)
            st.write(answer)
    
