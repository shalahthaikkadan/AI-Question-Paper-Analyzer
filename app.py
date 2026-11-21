
from pathlib import Path

import streamlit as st
import pandas as pd
import pymupdf

from modules.question_classifier import extract_questions
from modules.data_preprocessor import prepare_dataset, extract_year
from modules.similarity_engine import load_model, calculate_similarity
from modules.question_cluster import create_question_clusters
from modules.frequency_analyzer import analyze_question_frequency


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
OUTPUT_DIR = BASE_DIR / "output"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="AI Question Paper Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #243b53;
    }

    .sub-title {
        color: #627d98;
        font-size: 15px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e6eaf0;
        padding: 18px;
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 8px;
        min-height: 42px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD AI MODEL
# =====================================================

@st.cache_resource(show_spinner=False)
def get_ai_model():
    return load_model()


# =====================================================
# 4. RUN COMPLETE ANALYSIS
# =====================================================

def run_analysis(uploaded_files):

    all_questions = []
    processing_errors = []

    # ---------------------------------------------
    # Extract questions from uploaded PDFs
    # ---------------------------------------------

    progress = st.progress(0)
    status = st.empty()

    for index, uploaded_file in enumerate(uploaded_files):

        filename = Path(uploaded_file.name).name

        status.write(f"Processing PDF: {filename}")

        try:

            pdf_path = UPLOAD_DIR / filename

            # Save uploaded PDF
            with open(pdf_path, "wb") as file:
                file.write(uploaded_file.getvalue())

            # Read PDF text
            doc = pymupdf.open(str(pdf_path))

            full_text = ""

            for page in doc:
                full_text += page.get_text("text") + "\n"

            doc.close()

            # Extract questions
            questions = extract_questions(full_text)

            if not questions:
                processing_errors.append(
                    f"{filename}: No questions extracted."
                )
                progress.progress((index + 1) / len(uploaded_files))
                continue

            # Detect exam year
            exam_year = extract_year(
                filename,
                full_text
            )

            # Add PDF metadata
            for question in questions:

                question["source_file"] = filename
                question["exam_year"] = exam_year

            all_questions.extend(questions)

        except Exception as error:

            processing_errors.append(
                f"{filename}: {str(error)}"
            )

        progress.progress((index + 1) / len(uploaded_files))

    progress.empty()
    status.empty()

    # ---------------------------------------------
    # Validate extracted questions
    # ---------------------------------------------

    if not all_questions:
        raise ValueError(
            "No questions could be extracted from the uploaded PDFs."
        )

    extracted_df = pd.DataFrame(all_questions)

    # Save extracted questions
    extracted_df.to_csv(
        OUTPUT_DIR / "extracted_questions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ---------------------------------------------
    # Preprocess dataset
    # ---------------------------------------------

    prepared_df = prepare_dataset(extracted_df)

    prepared_df.to_csv(
        OUTPUT_DIR / "prepared_questions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if len(prepared_df) < 2:
        raise ValueError(
            "At least two valid questions are required for AI comparison."
        )

    # ---------------------------------------------
    # AI semantic similarity
    # ---------------------------------------------

    status = st.empty()

    status.write("Loading AI semantic similarity model...")

    model = get_ai_model()

    status.write("Comparing question meanings...")

    similarity_df = calculate_similarity(
        prepared_df,
        model,
        threshold=0.85
    )

    # Ensure expected columns exist, even if no pairs were found
    similarity_columns = [
        "index_1",
        "index_2",
        "similarity_score",
        "question_1",
        "question_2",
        "year_1",
        "year_2",
        "part_1",
        "part_2",
        "module_1",
        "module_2",
        "source_1",
        "source_2"
    ]

    if similarity_df.empty:
        similarity_df = pd.DataFrame(
            columns=similarity_columns
        )

    similarity_df.to_csv(
        OUTPUT_DIR / "similar_questions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ---------------------------------------------
    # Question clustering
    # ---------------------------------------------

    status.write("Grouping repeated questions...")

    cluster_df = create_question_clusters(
        prepared_df,
        similarity_df
    )

    cluster_columns = [
        "cluster_id",
        "question",
        "part",
        "module",
        "year",
        "marks",
        "source_file"
    ]

    if cluster_df.empty:
        cluster_df = pd.DataFrame(
            columns=cluster_columns
        )

    cluster_df.to_csv(
        OUTPUT_DIR / "question_clusters.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ---------------------------------------------
    # Frequency analysis
    # ---------------------------------------------

    status.write("Analyzing question frequency...")

    frequency_df = analyze_question_frequency(
        cluster_df
    )

    frequency_columns = [
        "cluster_id",
        "question",
        "part",
        "module",
        "total_occurrences",
        "total_papers",
        "total_years",
        "years_appeared"
    ]

    if frequency_df.empty:
        frequency_df = pd.DataFrame(
            columns=frequency_columns
        )

    frequency_df.to_csv(
        OUTPUT_DIR / "question_frequency.csv",
        index=False,
        encoding="utf-8-sig"
    )

    status.empty()

    return {
        "extracted": extracted_df,
        "prepared": prepared_df,
        "similarity": similarity_df,
        "clusters": cluster_df,
        "frequency": frequency_df,
        "errors": processing_errors
    }


# =====================================================
# 5. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("📚 Question Analyzer")

st.sidebar.caption(
    "AI-powered university exam preparation"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Upload Papers",
        "Repeated Questions",
        "Module Analysis",
        "Year Analysis",
        "Download Reports"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Upload previous university question papers "
    "to discover frequently repeated questions."
)


# =====================================================
# 6. PAGE HEADER
# =====================================================

st.markdown(
    '<div class="main-title">AI Question Paper Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Analyze previous question papers and discover important questions.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# =====================================================
# 7. UPLOAD PAPERS PAGE
# =====================================================

if page == "Upload Papers":

    st.header("Upload Question Papers")

    st.write(
        "Upload multiple university examination papers in PDF format."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} PDF file(s) selected."
        )

        for file in uploaded_files:
            st.write("📄", file.name)

        if st.button(
            "Start AI Analysis",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Analyzing your question papers. Please wait..."
                ):

                    results = run_analysis(uploaded_files)

                st.session_state["analysis_results"] = results

                st.success(
                    "AI analysis completed successfully!"
                )

                st.balloons()

                st.write(
                    "Go to the Dashboard to view your results."
                )

                if results["errors"]:
                    st.warning(
                        "Some PDFs could not be fully processed:"
                    )
                    for error in results["errors"]:
                        st.write("- " + error)

            except Exception as error:

                st.error(
                    f"Analysis failed: {str(error)}"
                )

    else:

        st.info("Upload at least one PDF to begin.")


# =====================================================
# 8. GET ANALYSIS RESULTS
# =====================================================

results = st.session_state.get(
    "analysis_results"
)


# =====================================================
# 9. DASHBOARD PAGE
# =====================================================

if page == "Dashboard":

    st.header("Dashboard")

    if results is None:

        st.info(
            "No analysis results available. "
            "Go to Upload Papers and upload your PDFs."
        )

    else:

        extracted_df = results["extracted"]
        prepared_df = results["prepared"]
        similarity_df = results["similarity"]
        cluster_df = results["clusters"]
        frequency_df = results["frequency"]

        total_papers = extracted_df["source_file"].nunique()

        total_questions = len(prepared_df)

        repeated_pairs = len(similarity_df)

        total_clusters = (
            cluster_df["cluster_id"].nunique()
            if not cluster_df.empty else 0
        )

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "PDF Papers",
            total_papers
        )

        col2.metric(
            "Questions",
            total_questions
        )

        col3.metric(
            "Similar Pairs",
            repeated_pairs
        )

        col4.metric(
            "Question Clusters",
            total_clusters
        )

        st.divider()

        # Recent frequency results
        st.subheader("Frequently Repeated Questions")

        if not frequency_df.empty:

            st.dataframe(
                frequency_df.head(10),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No repeated question clusters found at the current threshold."
            )

        # Module distribution
        st.subheader("Questions by Module")

        module_counts = (
            prepared_df["module"]
            .fillna("Unknown")
            .value_counts()
        )

        st.bar_chart(module_counts)

        # Questions by year
        st.subheader("Questions by Examination Year")

        year_counts = (
            prepared_df["year"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
        )

        st.bar_chart(year_counts)


# =====================================================
# 10. REPEATED QUESTIONS PAGE
# =====================================================

if page == "Repeated Questions":

    st.header("Repeated Questions")

    if results is None:

        st.info("Upload and analyze question papers first.")

    else:

        frequency_df = results["frequency"]
        similarity_df = results["similarity"]

        st.subheader("Frequency-ranked questions")

        if not frequency_df.empty:

            selected_part = st.selectbox(
                "Filter by question part",
                ["All", "A", "B"]
            )

            filtered_df = frequency_df.copy()

            if selected_part != "All":
                filtered_df = filtered_df[
                    filtered_df["part"] == selected_part
                ]

            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No repeated questions found.")

        st.divider()

        st.subheader("AI Similarity Matches")

        if not similarity_df.empty:

            min_score = st.slider(
                "Minimum similarity score",
                min_value=0.85,
                max_value=1.0,
                value=0.85,
                step=0.01
            )

            filtered_pairs = similarity_df[
                similarity_df["similarity_score"] >= min_score
            ]

            st.dataframe(
                filtered_pairs[
                    [
                        "similarity_score",
                        "question_1",
                        "question_2",
                        "year_1",
                        "year_2",
                        "module_1"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No AI similarity matches found.")


# =====================================================
# 11. MODULE ANALYSIS PAGE
# =====================================================

if page == "Module Analysis":

    st.header("Module-wise Analysis")

    if results is None:

        st.info("Upload and analyze question papers first.")

    else:

        prepared_df = results["prepared"]
        frequency_df = results["frequency"]

        module_counts = (
            prepared_df["module"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Module")
            .reset_index(name="Total Questions")
        )

        st.subheader("Questions in Each Module")

        st.dataframe(
            module_counts,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            module_counts.set_index("Module")
        )

        st.divider()

        st.subheader("Repeated Questions by Module")

        if not frequency_df.empty:

            repeated_by_module = (
                frequency_df.groupby("module")
                ["total_occurrences"]
                .sum()
            )

            st.bar_chart(repeated_by_module)

            st.dataframe(
                frequency_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No repeated questions available.")


# =====================================================
# 12. YEAR ANALYSIS PAGE
# =====================================================

if page == "Year Analysis":

    st.header("Examination Year Analysis")

    if results is None:

        st.info("Upload and analyze question papers first.")

    else:

        prepared_df = results["prepared"]
        frequency_df = results["frequency"]

        year_counts = (
            prepared_df["year"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("Year")
            .reset_index(name="Total Questions")
        )

        st.subheader("Questions by Year")

        st.bar_chart(
            year_counts.set_index("Year")
        )

        st.dataframe(
            year_counts,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Questions Repeated Across Years")

        if not frequency_df.empty:

            st.dataframe(
                frequency_df[
                    [
                        "question",
                        "module",
                        "total_papers",
                        "total_years",
                        "years_appeared"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No repeated questions found.")


# =====================================================
# 13. DOWNLOAD REPORTS PAGE
# =====================================================

if page == "Download Reports":

    st.header("Download Analysis Reports")

    if results is None:

        st.info("Run AI analysis before downloading reports.")

    else:

        report_files = {
            "Extracted Questions": (
                results["extracted"],
                "extracted_questions.csv"
            ),
            "Prepared Questions": (
                results["prepared"],
                "prepared_questions.csv"
            ),
            "AI Similarity Results": (
                results["similarity"],
                "similar_questions.csv"
            ),
            "Question Clusters": (
                results["clusters"],
                "question_clusters.csv"
            ),
            "Question Frequency": (
                results["frequency"],
                "question_frequency.csv"
            )
        }

        st.write(
            "Download the results of your AI question paper analysis."
        )

        for title, (dataframe, filename) in report_files.items():

            csv_data = dataframe.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label=f"Download {title}",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )


# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.caption(
    "AI Question Paper Analyzer | Built with Python, "
    "Streamlit and Sentence Transformers"
)
