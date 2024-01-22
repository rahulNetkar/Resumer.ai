from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import PyPDF2 as pdf
import json


# Returns converted pdf file to text
def pdf_to_text(file):
    loader = pdf.PdfReader(file)

    text = ""

    for i in range(len(loader.pages)):
        page = loader.pages[i]
        text += str(page.extract_text())

    return text


# Returns response generated by Gemini-pro as a string
def get_response(text, jd, api_key):
    model = GoogleGenerativeAI(
        model="models/gemini-pro",
        google_api_key=api_key,
        temperature=0.1,
    )

    # Construct prompt carefully as the response of the llm heavily depends on it
    prompt = PromptTemplate.from_template(
        """ Below given is the job description and candidates resume.
            description: {jd}
            resume:{text}
            
            Act like a skilled and experience Application Tracking System with a deep understanding of 
            tech field, software engineering, data science, machine learning, data 
            analyst, big data engineering, full stack developer. 
            
            You are to be performing these tasks with 100%/ accuracy:
            - Understand the given job description properly
            - Evaluate the given resume based on the given job description
            - Assign the percentage matching of the resume based on the job description
            - Suggest upto 10 to 20 important keywords that might be missing from the resume so as to match it better
            - Give proper and thorough reason as of why and why not the candidate is fit for the position/job
            - Give proper and thorough points to improve candidates profile and resume
            
            You must consider the job market is very competetive and you should provide best assistance for improving the
            resume.
            
            I want the response in the form of structure like this only. And make sure to only insert newlines, 
            spaces and tabs in "Reason", "Improvement" section -  
            {{"Match" : "", "MissingKeywords":[], "Reason": "", "Improvements": "",}}
            """
    )

    chain = LLMChain(llm=model, prompt=prompt)  # Making chain

    response = chain.run(text=text, jd=jd)  # invoking the chain

    return response


# Streamlit app
def main():
    st.title("Resumer.ai")
    st.header("Check and Improve Your Resume with this awesome AI bot")

    with st.sidebar:
        api_key = st.text_input("API Key", placeholder="Paste your Gemini-Pro API key")

        st.markdown(
            """
                    ### [Get your own api key](https://ai.google.dev/tutorials/setup)
                    """
        )

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            file = st.file_uploader(
                "Upload your resume", type="pdf", help="Please upload the pdf"
            )

        with col2:
            jd = st.text_area("Paste Job Description here", height=250)

    submit = st.button("Submit")

    if submit:
        try:
            text = pdf_to_text(file)
            response = get_response(text, jd, api_key)

            json_resp = json.loads(response)

            with st.container():
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Match %", "Missing keywords", "Reason", "Things to improve"]
                )

                with tab1:
                    st.progress(int(json_resp["Match"][:-1]))
                    st.subheader(f'{json_resp["Match"]}')

                with tab2:
                    for i in json_resp["MissingKeywords"]:
                        st.subheader(i)

                with tab3:
                    st.subheader(json_resp["Reason"])

                with tab4:
                    st.subheader(json_resp["Improvements"])

        except Exception as e:
            st.warning(e)


if __name__ == "__main__":
    main()
