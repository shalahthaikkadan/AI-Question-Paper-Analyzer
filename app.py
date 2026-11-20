
import streamlit as st
import fitz

st.set_page_config(
    page_title="AI Question Paper Analyzer",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Question Paper Analyzer")
st.write(
    "Upload university question papers "
    "to identify frequently repeated questions."
)

uploaded_files = st.file_uploader(
    "Upload Question Paper PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) uploaded!")

    if st.button("Extract PDF Text"):
        for uploaded_file in uploaded_files:

            st.subheader(uploaded_file.name)

            pdf_bytes = uploaded_file.getvalue()

            document = fitz.open(
                stream=pdf_bytes,
                filetype="pdf"
            )

            extracted_text = ""

            for page_number, page in enumerate(document):
                text = page.get_text("text")
                extracted_text += (
                    f"\n--- Page {page_number + 1} ---\n"
                    + text
                )

            st.text_area(
                "Extracted Content",
                extracted_text,
                height=300
            )

            st.download_button(
                label="Download Extracted Text",
                data=extracted_text,
                file_name=uploaded_file.name + ".txt",
                mime="text/plain"
            )

            document.close()
