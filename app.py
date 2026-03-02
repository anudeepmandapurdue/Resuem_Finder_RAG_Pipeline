"""
Resume Finder UI - Simple local interface for job description search.
Run with: streamlit run app.py
"""
import streamlit as st

from resume_search import search

st.set_page_config(page_title="Resume Finder", page_icon="🔍", layout="centered")

st.title("🔍 Resume Finder")
st.caption("Paste a job description to find your best-matching resumes")

job_description = st.text_area(
    "Job description",
    placeholder="Paste the full job posting here (role, requirements, responsibilities...)",
    height=200,
    help="The more detail you include, the better the match.",
)

if st.button("Find best resumes", type="primary"):
    if not job_description or not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        with st.spinner("Searching resumes..."):
            try:
                ranked_results = search(job_description.strip(), fetch_k=20)
            except Exception as e:
                st.error(f"Error: {e}")
                st.info(
                    "Make sure you've run pdf_loader.ipynb at least once to build the vector store."
                )
                st.stop()

        if not ranked_results:
            st.info("No matches found. Have you run the notebook to build the vector store?")
        else:
            st.success("Here are your top matches:")
            st.divider()

            for rank, (filename, stats) in enumerate(ranked_results[:5], 1):
                score = stats["max_score"]
                quality = (
                    "🟢 High"
                    if score > 0.7
                    else "🟡 Moderate"
                    if score > 0.4
                    else "🔴 Low"
                )

                with st.expander(f"**#{rank}** {filename} — {quality} match ({score:.2f})"):
                    st.caption(f"Relevance score: {score:.4f}")
                    st.text(stats["top_text"][:500] + ("..." if len(stats["top_text"]) > 500 else ""))
