import os
import logging
import json
import copy
import re
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import time

# LangChain imports for native LangSmith tracing
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from utils.utils import send_email

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

org = os.environ["ORG"]
env = os.environ["ENV"]


GEMINI_MODEL = os.environ["GEMINI_MODEL"]

LLM = os.environ["LLM"]
LLM_MODEL = os.environ["CHATGPT_MODEL"]
CHATGPT_MODEL = os.environ["CHATGPT_MODEL"]
temperature       = 0.7
max_tokens        = 600
top_p             = 0.85
frequency_penalty = 0.3


category_list = ['Chief Complaint', 'Diagnoses', 'Date of Surgery', 'HPI Source', 
                 'History of Present Illness', 'Interval History', 'ROS (Review of Symptoms)',
                 'ROS (Review of Systems)', 'Physical Exam', 'Impression', 'Imaging', 
                 'Assessment', 'Plan', 'Procedure', 'Follow Up', 'Patient Instructions', 
                 'Return Visit', 'Imaging and Studies', 'Referral',
                 ]

steps = """
    2. Go through the generated note and correct any grammatical errors.
    3. Go through the grammatically corrected note and ensure correct usage of medical terminology, like using 'abdomen' in place of 'stomach'. Also ensure that generic drug names are written in lowercase, while brand drug names are written in uppercase.
    4. Go through the medical terminology corrected note and correct any tenses related error.
    5. Go through the tense corrected note and check if Info any information from HPI, PE or Plan is coming into Assessment section. Correctly place the info in each section.
    6. Go through the final note and ensure only the subsections of 'ROS' where symptoms are reported or discussed are mentioned.
    7. Go through the final note and ensure that the note follows a medical logic. 
    8. Go through the complete note and ensure definitive diagnosis is given. You are required to not give differential diagnosis.
    9. Go through final note and ensure no phrasing error is present in the note. For example - 'abdomen feels fine' should be written as 'abdomen is normal'.
    10. Go through final note and ensure no symptoms are present in the 'Physical Examination' section
    11. You are required to only give the final note and not the summary or status of completion steps.
    12. You are required to act as a Quality auditor and fix all the errors in the notes which includes - 'grammar, phrasing, contradictions and medical terminologies'.
"""


system_prompt = """You are a medical scribe assistant to a Doctor. Use proper phrasing of sentences. Make sure not to keep repetitive information throughout the note. """

user_prompt = """Create a medical note capturing relevant information for the below sections, as per the descriptions given for each:

Chief Complaint : Identify the complaint/ reason of visiting the doctor. If possible, also identify if the visit is acute type or follow-up. Document the information in a brief single sentence.

History of Present Illness : Summarize in a single paragraph the conversation in detail, capturing relevant information for the complaint being discussed.Capture the information about all diagnoses discussed in the visit today.  Do not include any suggestion or recommendation or plan dictated by the doctor. Do not include any information related to treatment plan or physical exam (if performed or discussed).

Physical Exam : Summarize in a single paragraph, the information from the conversation where details of physical or medical examination related to the visit is discussed. Do not include the information already documented under 'History of Present Illness' section.

Plan : In this section, the doctor prefers to write the  information regarding suggestion/ recommendation/ treatment plan addressed for each diagnosis mentioned in the conversation. Do not include information already documented under 'History of Present Illness' and 'Physical Exam' sections.

Assessment : Summarize in bullet points, the information related to the potential diagnosis performed during the visit, related to the chief complaint. Do not include information already documented under 'History of Present Illness' , 'Physical Exam' and 'Plan' sections.

ROS (Review of Symptoms) : Based on the information documented in the report so far, identify the symptoms related to the chief complaint discussed during the visit and mention if the patient reports to or denies to experiencing the symptom, for the following subcategories : General Health, Allergy/Immunology, ENT, Respiratory, Cardiovascular, Palpitations, Gastrointestinal, Musculoskeletal, Skin Related, Neurologic, Psychiatric.

Follow Up : Based on the information analyzed from the conversation and the information documented in above sections, identify the follow-up instructions suggested by the doctor for future visit (if dictated). If no such information is found, mention 'To Be Discussed'.

Use the given doctor-patient dialog transcript to create the medical note as described above. "spk_0" is doctor and "spk_1" is patient: """


order_system_prompt = """
You are a highly skilled medical transcription assistant. Your task is to read through a doctor's transcript and accurately extract any medical orders (medications, procedures, instructions) and lab-related information (tests, results, dates). You will then create a structured report with two primary sections:
Orders - Detailed entries for each order mentioned (including medication names, dosages, frequencies, procedures, or other instructions).
Labs - Detailed entries for each lab test mentioned (including test name, result values, dates, and any other relevant information).
Your report must be clear, comprehensive, and logically organized.
"""

order_user_prompt = """
Create a well-structured report of orders and labs from the provided doctor's transcript.
 
- Extract relevant information about medical orders and lab results from the speech.
- Organize the extracted data into clear and coherent sections for orders and labs.
- Ensure that the report is detailed, complete, and easy to understand.
 
# Steps
 
1. **Identify Key Information**: Focus on parts of the transcript where the doctor discusses orders (e.g., medications, procedures) and labs (e.g., test names, results).
2. **Categorization**: Separate and categorize information into 'Orders' and 'Labs' sections.
3. **Details**: Include specific details such as medication names, dosages, test types, results, dates, and any relevant patient information.
 
# Output Format
 
Produce the output with clearly defined sections:
- **Orders**: List of all orders mentioned, with details.
- **Labs**: List of all lab tests and results, with details.
 
**Example Output:**
 
- **Orders**:
  - Medication: Amoxicillin, Dosage: 500 mg, Frequency: Twice daily
- **Labs**:
  - Test: CBC, Result: Hemoglobin at 12.5 g/dL
  - Test: Metabolic Panel, Result: [Results not provided in transcript]
 
(Note: Real examples should be more detailed, including dates and additional findings as provided in the transcript.)
 
# Notes
 
- Ensure accuracy and clarity in extracting medical terms and results.
- Pay attention to any mention of test dates or chronological information, as context may impact report content.
"""

transcript_system_prompt = """
You are a highly skilled medical language translation assistant specializing in translating Spanish transcripts into English and retaining English text as-is. Your task is to create an “English Transcript” with these rules:
 
• Read the provided conversation carefully.  
• For any portions in Spanish, produce a English translation while preserving speaker labels.  
• For any portions already in English, leave the text exactly as it is.  
• Do not omit or summarize anything. Maintain the order of the conversation.  
• Label the final output as “English Transcript” at the top.  
• Provide no extra explanation of your process—only output the fully translated transcript.
 
If the entire transcript is already in English, simply state: English Transcript: Provided transcript is in English language. In all other mixed or fully Spanish scenarios, output the entire transcript in English (including unchanged English segments) with proper speaker labels. Provide a complete translation of the transcript rather than leaving parts for the reader to infer. Ensure that all details are fully translated and clearly conveyed.
"""

transcript_user_prompt = """
Create an "English Transcript" translation document from the given doctor-patient conversation transcript:
"""

icd_response_format = {
    "format": {
        "type": "json_schema",
        "name": "icd_codes_list",
        "strict": True,
        "schema":{
            "type": "object",
            "properties": {
            "ICD 10 Code": {
                    "type": "array",
                    "description": """List of ICD codes generated from the medical transcript in following format,
                    where each line starts with ICD Code 1: '[ICD 10 code]' \n Condition or diagnosis: '[Condition/Diagnosis]'.
                    Output Format:
                        ICD Code 1: '[ICD 10 code]' \n Condition or diagnosis: '[Condition/Diagnosis]'
                        ICD Code 2: '[ICD 10 code]' \n Condition or diagnosis: '[Condition/Diagnosis]'
                    Examples:
                        ICD Code 1: 'I10' \n Condition or diagnosis: 'Essential (primary) hypertension'
                        ICD Code 2: 'J03.90' \n Condition or diagnosis: 'Acute tonsillitis, unspecified'
                    """,
                    "items": {
                    "type": "string",

                    }
                }                              
            },
            "required": [
            "ICD 10 Code",
            ],
            "additionalProperties": False
        }
    }
}

cpt_response_format = {
    "format": {
        "type": "json_schema",
        "name": "cpt_codes_list",
        "strict": True,
        "schema":{
            "type": "object",
            "properties": {
            "CPT Code": {
                    "type": "array",
                    "description": """List of CPT codes generated from the medical transcript in following format,
                    where each line starts with CPT: '[CPT code]' \n Rationale: '[Rationale]'.

                    Output Format:
                        CPT: '[CPT code]' \n Rationale: '[Rationale]'
                        CPT: '[CPT code]' \n Rationale: '[Rationale]'
                    Examples:
                        CPT: 99214 \n Rationale: 'Moderate complexity due to management of two stable chronic conditions, review of labs and medication management.'
                        CPT: 99203 \n Rationale: 'Moderate complexity due to new problem of chest pain, ordering and review of diagnostic tests and moderate risk medication management.'
                    """,
                    "items": {
                    "type": "string",

                    }
                }                              
            },
            "required": [
            "CPT Code",
            ],
            "additionalProperties": False
        }
    }
}

order_response_format = {
    "format": {
        "type": "json_schema",
        "name": "orders_labs_list",
        "strict": True,
        "schema":{
            "type": "object",
            "properties": {
            "Orders": {
                    "type": "array",
                    "description": """List of orders asked by Provider from the medical transcript in following format,
                    where each line starts with Medication: [Medication], Dosage: [Dosage], Frequency: [Medication Frequency].
                    Output Format:
                        - Medication: [Medication], Dosage: [Dosage], Frequency: [Medication Frequency]
                        - Medication: [Medication], Dosage: [Dosage], Frequency: [Medication Frequency]
                    Examples:
                        - Medication: Amoxicillin, Dosage: 500 mg, Frequency: Twice daily
                        - Medication: Ibuprofen , Dosage: 200 mg, Frequency: Once daily
                    """,
                    "items": {
                    "type": "string",

                    }
                },
            "Labs": {
                    "type": "array",
                    "description": """List of tests asked by Provider from the medical transcript in following format,
                    where each line starts with Test: '[Test]', Result: '[Result]'.
                    Output Format:
                        - Test: [Test], Result: [Result]
                        - Test: [Test], Result: [Result]
                    Examples:
                        - Test: CBC, Result: Hemoglobin at 12.5 g/dL
                        - Test: Metabolic Panel, Result: [Results not provided in transcript]
                    """,
                    "items": {
                    "type": "string",

                    }
                }  
            },
            "required": [
            "Orders",
            "Labs"
            ],
            "additionalProperties": False
        }
    }
}


ASR               = "Deepgram"
enable_icd        = "False"
enable_cpt        = "False"
enable_hcc        = "False"
cpt_system_prompt = """Evaluate Electronic Medical Records (EMR) in three roles: Risk Assessment Evaluator, Data Complexity Analyzer, and Problem Complexity Evaluator. Your evaluations will contribute to determining the complexity level of patient encounters, aiding in the Evaluation and Management (E/M) level assessment.Please enforce strict rules to just give the CPT code and a brief rationale. Risk Assessment Evaluator Identify Treatment Strategies: Review the EMR for treatment plans, interventions, or management strategies. Evaluate Associated Risks: Assess the risks linked to these management decisions, considering patient condition and potential outcomes. Assess Health Status: Evaluate the overall health of the patient, noting any comorbidities or chronic conditions. Analyze Complications: Determine the likelihood of significant complications or adverse outcomes. Document Risk Discussions: Record any discussions about potential risks and their management. Data Complexity Analyzer Identify Reviewed Data: Identify diagnostic tests, lab results, imaging studies, or other reviewed or ordered data.Complexity Evaluation: Assess the complexity based on test variety, interpretation effort, and further investigations needed.External Data Review: Determine if external data were reviewed, including consultations.Consultations Noted: Highlight consultations indicating complex data review.Problem Complexity Evaluator Identify Medical Problems: Detail medical problems, diagnoses, or conditions managed during the encounter. Classify Problems by Categories: Minimal Problem Self-limited or Minor Problem Stable, Chronic Illness Acute, Uncomplicated Illness or Injury Chronic Illness with Exacerbation, Progression, or Side Effects Undiagnosed New Problem with Uncertain Prognosis Acute Illness with Systemic Symptoms Acute, Complicated Injury Chronic Illness with Severe Exacerbation, Progression, or Side Effects Acute or Chronic Condition Posing Life/Bodily Threat Rationale for Classification: Provide detailed reasons for each assigned category. Impact on E/M Level: Explain how problem complexity informs the E/M level. Output Format CPT Code: Specify the CPT code corresponding to the medical decision-making level. Rationale: Provide a brief rationale summarizing the key factors influencing the CPT code selection. Examples CPT: [CPT Code Example] Rationale: 'High complexity due to management of chronic illness with severe exacerbation and high-risk medication management.' Notes Ensure all evaluations are based on the provided EMR and conversation transcript. Emphasize reasoning in evaluating risks, data complexity, and problem classifications. Only provide the CPT code and a brief rationale as per the strict rules."""
cpt_user_prompt   = """Below is the conversation between the doctor and patient, and the corresponding SOAP note. You are required to analyze both and extract relevant CPT code accordingly.\nDoctor-Patient Conversation: \nSOAP NOTE:\n\n"""
icd_system_prompt = """You are an AI assistant specialized in extracting ICD-10 codes from medical records. You will be provided with a doctor-patient conversation and an existing SOAP (Subjective, Objective, Assessment, and Plan) note. Using the latest available ICD-10 guidelines and resources online, follow the below instructions to accurately extract the necessary codes from the provided information. Instructions: Assess the SOAP Note: Analyze the provided SOAP note to identify all necessary information related to the diagnosis and conditions present in the 'Assessment' and 'Plan' sections. Analyze Doctor-Patient Conversation: Utilize the conversation to validate the diagnosis found in the SOAP note or identify additional conditions or details that are not covered in the SOAP note. Section Analysis to Extract ICD-10 Codes: Assessment: Check this section for direct diagnoses and assign appropriate ICD-10 codes accordingly. Plan: Extract any additional conditions treated or being monitored, including preventative care (e.g., screening for conditions like hypertension). Assign relevant ICD-10 codes based on the details. Doctor-Patient Conversation: If applicable, utilize symptoms, conditions, or other health concerns discussed to add context or enhance your understanding, contributing to the final code selection. Latest Guidelines: Ensure that you look up the latest available ICD-10 guidelines online to assign the correct and most recent codes for each condition. Make use of credible and up-to-date online medical coding resources. Output Format: The output should be in JSON format as shown below: { 'Accnt#': [Extract and insert the Accnt# value from the SOAP note], 'ICD Code_1': { 'Code': '[ICD-10 code]', 'Condition or diagnosis': '[Condition/Diagnosis]', 'Section Name': '[SOAP Section]' }, 'ICD Code_2': {'Code': '[ICD-10 code]', 'Condition or diagnosis': '[Condition/Diagnosis]', 'Section Name': '[SOAP Section]'}, ...} Make sure the output is properly formatted and validated with no missing braces or commas."""
icd_user_prompt   = """Below is the conversation between the doctor and patient, and the corresponding SOAP note. You are required to analyze both and extract relevant ICD-10 codes accordingly. Doctor-Patient Conversation: SOAP Note: """
addendum_system_prompt = """"""
addendum_user_prompt = """Create a SOAP note summary based on the addendum transcript of a medical discussion between a healthcare provider and patient. \n Use proper medical terminology (e.g., use 'abdomen' instead of 'stomach'). Only document the SOAP note sections explicitly discussed in the addendum and exclude any unrelated information. Do not infer or add details beyond those presented in the provided transcript. Maintain conciseness and avoid repeating information. \n # Steps \n 1. Review the addendum transcript. \n 2. Identify information relevant to each section of the SOAP note (Subjective, Objective, Assessment, Plan). \n 3. Document only those sections explicitly referenced in the transcript related to the SOAP note. \n # Output Format \n The output should be a concise SOAP note that follows this structure: \n - **Subjective**: Only include patient-provided details mentioned in the addendum. \n - **Objective**: Include documented provider observations or measurements. \n - **Assessment**: Provide the provider's diagnosis or evaluation as showcased in the addendum. \n - **Plan**: Only document the care plan as noted in the addendum. \n # Notes \n - If a section is not covered in the addendum, omit it. \n - Ensure concise phrasing without unnecessary repetition or elaboration."""
language_code = "en-US"
enable_secondary_language = "false"
secondary_language_code = ""
response_format   = None
enable_hearo      = 'false'
enable_memory_layer= 'false'
memory_layer_text  = ""



# LangChain clients for automatic LangSmith tracing
def get_langchain_openai(model: str, temperature: float = 0.7, top_p: float = 0.85, 
                          frequency_penalty: float = 0.3, max_tokens: int = None) -> ChatOpenAI:
    """Get LangChain ChatOpenAI instance with LangSmith tracing enabled."""
    kwargs = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "timeout": 60,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)

def get_langchain_gemini(model: str, temperature: float = 0.7, top_p: float = 0.85, 
                          max_tokens: int = None) -> ChatGoogleGenerativeAI:
    """Get LangChain ChatGoogleGenerativeAI instance with LangSmith tracing enabled."""
    kwargs = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "google_api_key": os.environ.get("GEMINI_API_KEY"),
    }
    if max_tokens:
        kwargs["max_output_tokens"] = max_tokens
    return ChatGoogleGenerativeAI(**kwargs)
    

### Without json schema ###

def call_chatgpt(assistant_info, message, model, temperature, max_tokens, top_p, frequency_penalty, 
                 ProviderName, source_path, spk_lbl_content, status_logger, 
                 providerID=None, chartType=None, VisitTypeId=None, counter = 0):
    """
    Call OpenAI ChatGPT using LangChain for automatic LangSmith tracing.
    Returns text output and preserves the same return format (output, model_used).
    """
    try:
        # Use LangChain ChatOpenAI for automatic LangSmith tracing
        llm = get_langchain_openai(
            model=model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty
        )
        
        # Add metadata for LangSmith tracing
        llm = llm.with_config({
            "run_name": "call_chatgpt",
            "metadata": {
                "LLM_Service": "openai",
                "provider_name": ProviderName,
                "providerID": providerID,
                "chartType": chartType,
                "VisitTypeId": VisitTypeId,
                "retry_attempt": counter + 1,
            }
        })
        
        messages = [
            SystemMessage(content=assistant_info),
            HumanMessage(content=message)
        ]
        
        response = llm.invoke(messages)
        chat_gpt_output = response.content
        
        return chat_gpt_output, model
        
    except Exception as gpt_error:
        if counter <= 1:
            logging.error(f"{model} failed due to: {str(gpt_error)}, Retrying: {counter+1}")
            if counter > 0:
                model = CHATGPT_MODEL
            return call_chatgpt(assistant_info, message, model, temperature, max_tokens, top_p, frequency_penalty, 
                                ProviderName, source_path, spk_lbl_content, status_logger, 
                                providerID, chartType, VisitTypeId, counter + 1)
        else:
            logging.error(f"{model} failed due to: {str(gpt_error)}")

            # Insert the file details in the FileProcessingStatus table
            Status = "CHATGPT API FAILED"
            Message = f"ChatGPT {model} failed to generate a report. Error: \n{str(gpt_error)}"
            status_logger(Status, Message)
            
            try:
                # Fallback to Gemini using LangChain
                model = GEMINI_MODEL
                llm_gemini = get_langchain_gemini(
                    model=model,
                    temperature=temperature,
                    top_p=top_p
                )
                
                llm_gemini = llm_gemini.with_config({
                    "run_name": "call_chatgpt_fallback_gemini",
                    "metadata": {
                        "LLM_Service": "google",
                        "provider_name": ProviderName,
                        "providerID": providerID,
                        "chartType": chartType,
                        "VisitTypeId": VisitTypeId,
                        "fallback": True,
                        "original_error": str(gpt_error),
                    }
                })
                
                messages = [
                    SystemMessage(content=assistant_info),
                    HumanMessage(content=f"USER PROMPT: {message}")
                ]
                
                response = llm_gemini.invoke(messages)
                gemini_output = response.content

                # Insert the file details in the FileProcessingStatus table
                Status = "REPROCESSED WITH GEMINI"
                Message = f"Gemini {model} successfully generated a report after ChatGPT failed."
                status_logger(Status, Message)

                subject = f"[ScribeRYTE PLUS]: SOAP Note Reprocessed with Gemini for Dr. {ProviderName}"
                body = f"""Gemini {model} successfully generated the SOAP note for the following file: \n{source_path} \n\nOriginally, ChatGPT failed to process this file due to the following error: \n{str(gpt_error)} \n\nTranscript: \n{spk_lbl_content}"""
                send_email(subject, body)
                
                return gemini_output, model

            except Exception as gemini_error:
                logging.error(f"{model} failed due to: {str(gemini_error)}")
                
                subject = f"URGENT [ScribeRYTE PLUS]: ChatGPT & Gemini failed to generate a report for Dr. {ProviderName}"
                body = f"""ChatGPT and Gemini failed to generate a report for the following file: \n{source_path} \n\nChatGPT Error: \n{str(gpt_error)} \n\nGemini Error: \n{str(gemini_error)} \n\nTranscript: \n{spk_lbl_content}"""
                send_email(subject, body)

                # Insert the file details in the FileProcessingStatus table
                Status = "LLMs API FAILED"
                Message = f"LLMs failed to generate a report. Error: \n{str(gpt_error)}\n\n{str(gemini_error)}"
                status_logger(Status, Message)

                raise gemini_error


def call_gemini(assistant_info, message, model, temperature, max_tokens, top_p, frequency_penalty, 
                ProviderName, source_path, spk_lbl_content, status_logger, 
                providerID=None, chartType=None, VisitTypeId=None, counter = 0):
    """
    Call Google Gemini using LangChain for automatic LangSmith tracing.
    Returns text output and preserves the same return format (output, model_used).
    """
    try:
        # Use LangChain ChatGoogleGenerativeAI for automatic LangSmith tracing
        llm = get_langchain_gemini(
            model=model,
            temperature=temperature,
            top_p=top_p
        )
        
        # Add metadata for LangSmith tracing
        llm = llm.with_config({
            "run_name": "call_gemini",
            "metadata": {
                "LLM_Service": "google",
                "provider_name": ProviderName,
                "providerID": providerID,
                "chartType": chartType,
                "VisitTypeId": VisitTypeId,
                "retry_attempt": counter + 1,
            }
        })
        
        messages = [
            SystemMessage(content=assistant_info),
            HumanMessage(content=f"USER PROMPT: {message}")
        ]
        
        response = llm.invoke(messages)
        gemini_output = response.content
        
        return gemini_output, model
        
    except Exception as gemini_error:
        if counter <= 1:
            logging.error(f"{model} failed due to: {str(gemini_error)}, Retrying: {counter+1}")
            if counter > 0:
                model = GEMINI_MODEL
            return call_gemini(assistant_info, message, model, temperature, max_tokens, top_p, frequency_penalty, 
                               ProviderName, source_path, spk_lbl_content, status_logger, 
                               providerID, chartType, VisitTypeId, counter + 1)
        else:
            logging.error(f"{model} failed due to: {str(gemini_error)}")

            # Insert the file details in the FileProcessingStatus table
            Status = "GEMINI API FAILED"
            Message = f"Gemini {model} failed to generate a report. Error: \n{str(gemini_error)}"
            status_logger(Status, Message)

            try:
                # Fallback to ChatGPT using LangChain
                model = CHATGPT_MODEL
                llm_openai = get_langchain_openai(
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty
                )
                
                llm_openai = llm_openai.with_config({
                    "run_name": "call_gemini_fallback_chatgpt",
                    "metadata": {
                        "LLM_Service": "openai",
                        "provider_name": ProviderName,
                        "providerID": providerID,
                        "chartType": chartType,
                        "VisitTypeId": VisitTypeId,
                        "fallback": True,
                        "original_error": str(gemini_error),
                    }
                })
                
                messages = [
                    SystemMessage(content=assistant_info),
                    HumanMessage(content=message)
                ]
                
                response = llm_openai.invoke(messages)
                chat_gpt_output = response.content

                # Insert the file details in the FileProcessingStatus table
                Status = "REPROCESSED WITH CHATGPT"
                Message = f"ChatGPT {model} successfully generated a report after Gemini failed."
                status_logger(Status, Message)

                subject = f"[ScribeRYTE PLUS]: SOAP Note Reprocessed with ChatGPT for Dr. {ProviderName}"
                body = f"""ChatGPT {model} successfully generated the SOAP note for the following file: \n{source_path} \n\nOriginally, Gemini failed to process this file due to the following error: \n{str(gemini_error)} \n\nTranscript: \n{spk_lbl_content}"""
                send_email(subject, body)

                return chat_gpt_output, model

            except Exception as gpt_error:
                logging.error(f"{model} failed due to: {str(gpt_error)}")
                
                subject = f"URGENT [ScribeRYTE PLUS]: ChatGPT & Gemini failed to generate a report for Dr. {ProviderName}"
                body = f"""ChatGPT and Gemini failed to generate a report for the following file: \n{source_path} \n\nChatGPT Error: \n{str(gpt_error)} \n\nGemini Error: \n{str(gemini_error)} \n\nTranscript: \n{spk_lbl_content}"""
                send_email(subject, body)

                # Insert the file details in the FileProcessingStatus table
                Status = "LLMs API FAILED"
                Message = f"LLMs failed to generate a report. Error: \n{str(gpt_error)}\n\n{str(gemini_error)}"
                status_logger(Status, Message)

                raise gpt_error


# def add_timestamp_to_html(html_string):
#     soup = BeautifulSoup(html_string, 'html.parser')

#     for tag in soup.find_all():
#         # Add the timestamp as a data attribute to the tag
#         timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
#         time.sleep(0.000001)
#         tag['class'] = f"AI_{timestamp}"
    
#     return str(soup)  


def add_timestamp_to_html(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')

    for tag in soup.find_all():
        # Generate a unique ID for each tag
        unique_id = 'e5e86b35040f1dac22709b401649b6b71' # str(uuid.uuid4())
        
        # Add the timestamp as a data attribute to the tag
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")  # Truncate milliseconds
        time.sleep(0.000001)  # Ensuring unique timestamps
        tag['id'] = unique_id
        
        # Create a new anchor tag and prepend it to the tag
        a_tag = soup.new_tag('a', href='#')
        a_tag.string = f"AI_{timestamp}"
        tag.insert(0, a_tag)
    
    return str(soup)
         


# def generate_html_string_updated(chatgpt_output, category_list):
#     chatgpt_output_lister = chatgpt_output.split("\n")
#     chatgpt_df = pd.DataFrame(chatgpt_output_lister)
#     chatgpt_df.rename(columns = {0:'content'}, inplace = True)
#     chatgpt_df['content'] = chatgpt_df['content'].str.replace(" +", " ")
#     chatgpt_df['content_length'] = chatgpt_df['content'].apply(len)
#     chatgpt_df = chatgpt_df[~(chatgpt_df['content_length'] <= 1)].reset_index(drop = True)
#     category_content_split_df = chatgpt_df['content'].str.split("-|:",expand = True)
#     category_content_split_df_columns_to_fillna = list(set(category_content_split_df.columns) - {0})
#     category_content_split_df['content_remaining'] = ''
#     for i in category_content_split_df_columns_to_fillna:
#         category_content_split_df[i] = category_content_split_df[i].fillna(value = '')
#         category_content_split_df['content_remaining'] += category_content_split_df[i]
#     category_content_split_df['content_remaining'] = category_content_split_df['content_remaining'].str.strip()
#     chatgpt_df['content_start'] = category_content_split_df[0]
#     chatgpt_df['content_remaining'] = category_content_split_df['content_remaining']
    
#     chatgpt_df['content_start'] = chatgpt_df['content_start'].str.strip()
#     chatgpt_df['category_flag'] = 0
#     for i in range(chatgpt_df.shape[0]):
#         content_start = chatgpt_df.loc[i, 'content_start']
#         if content_start in category_list:
#             chatgpt_df.loc[i, 'category_flag'] = 1
#         else:
#             pass
#     chatgpt_df.loc[chatgpt_df['category_flag'] == 1, 'category'] = chatgpt_df['content_start']
#     chatgpt_df.loc[chatgpt_df['category_flag'] == 1, 'content'] = chatgpt_df['content_remaining']
#     chatgpt_df['category'].ffill(inplace=True)
#     chatgpt_df = chatgpt_df[chatgpt_df['content']!= '']
#     chatgpt_df = chatgpt_df.reset_index(drop = True)
#     chatgpt_df_filtered = chatgpt_df[['content', 'category']]
#     chatgpt_df_filtered['content'] = chatgpt_df_filtered['content'].str.strip()

#     # chatgpt_df_filtered['last_word'] = chatgpt_df_filtered['content'].apply(lambda x: x.split()[-1])
#     # chatgpt_df_filtered['subheading'] = chatgpt_df_filtered['last_word'].apply(lambda x: True if x[-1] in [":","-"] else False)

#     chatgpt_df_filtered['last_word'] = chatgpt_df_filtered['content'].apply(lambda x: x.strip().split()[-1] if x.strip() else '')
#     chatgpt_df_filtered['subheading'] = chatgpt_df_filtered['last_word'].str[-1].isin([':', '-'])

#     chatgpt_df_filtered = chatgpt_df_filtered[~chatgpt_df_filtered['category'].isna()]
#     chatgpt_df_filtered.reset_index(drop = True)


#     category_list_filtered = [val for val in category_list if val in chatgpt_df_filtered['category'].unique()]
    
#     sectionwise_output = list(chatgpt_df_filtered.groupby(by = 'category', as_index=False).agg(
#                                                                     {'content':" ".join}).T.to_dict().values())

#     sorted_sectionwise_output = sorted(sectionwise_output, key=lambda x: category_list_filtered.index(x['category']))
    

#     html_string = """"""
#     for category in category_list_filtered:
#         temp_df = chatgpt_df_filtered[chatgpt_df_filtered['category'] == category]
#         temp_df = temp_df.reset_index(drop = True)
#         html_string = f"{html_string}<h2>{category}</h2>"
#         for index in range(temp_df.shape[0]):
#             is_subheading = temp_df.loc[index, 'subheading']
#             if(is_subheading):
#                 html_string = f"{html_string}<h4>{temp_df.loc[index, 'content']}</h4>"
#             else:
#                 html_string = f"{html_string}<p>{temp_df.loc[index, 'content']}</p>"

#     # html_string = add_timestamp_to_html(html_string)
    
#     return html_string, sorted_sectionwise_output


def generate_html_string(new_output_lister2):
    """
    Generates a string with HTML tags by parsing chatGPT output splitted in sentences in a list"""
    html_string = """"""
    for i in new_output_lister2:
        i = i.strip()
        if(len(i) == 0):
            continue
        if( ":" in i and "-" not in i):
            html_string += f"<h3>{i}</h3>"
        elif(":" in i and "-" in i):
            html_string += f"<h5>{i}</h5>"
        else:
            html_string += f"<p>{i}</p>"
    return html_string



def generate_html_string_updated(chatgpt_output, category_list):
    # Normalize category list for matching
    normalized_category_map = {c.lower().strip(): c for c in category_list}
    normalized_categories = list(normalized_category_map.keys())

    # Split input into non-empty lines
    lines = [line.strip() for line in chatgpt_output.split('\n') if line.strip()]
    section_indices = []
    section_names = []

    pre_parsed_sections = {}  # Inline key-value headers like "Chief Complaint: value"

    # Identify headers and index positions
    for idx, line in enumerate(lines):
        stripped_line = line.strip()
        # Inline section: "Chief Complaint: Fever"
        if ':' in stripped_line:
            possible_key, possible_value = map(str.strip, stripped_line.split(':', 1))
            key_norm = possible_key.lower()
            if key_norm in normalized_categories:
                if possible_value:
                    pre_parsed_sections[normalized_category_map[key_norm]] = [possible_value]
                section_indices.append(idx)
                section_names.append(normalized_category_map[key_norm])
        else:
            key_norm = stripped_line.lower().rstrip(':-')
            if key_norm in normalized_categories:
                section_indices.append(idx)
                section_names.append(normalized_category_map[key_norm])

    # Add final boundary
    if section_indices:
        section_indices.append(len(lines))

    section_content_map = {**pre_parsed_sections}

    # Fill section content between headers
    for i in range(len(section_indices) - 1):
        section_name = section_names[i]
        if section_name in pre_parsed_sections:
            continue
        start_idx = section_indices[i] + 1
        end_idx = section_indices[i + 1]
        content = lines[start_idx:end_idx]
        section_content_map[section_name] = content

    # Generate HTML
    html_string = ""
    for section in category_list:
        if section not in section_content_map:
            continue  # Skip if section isn't present in chatgpt_output

        html_string += f"<h2>{section}</h2>\n"
        for line in section_content_map[section]:
            if line.endswith(':') or line.endswith('-'):
                html_string += f"<h4>{line}</h4>\n"
            else:
                html_string += f"<p>{line}</p>\n"


    sorted_sectionwise_output = [
        {'category': sec, 'content': " ".join(section_content_map.get(sec, []))}
        for sec in category_list
    ]

    return html_string, sorted_sectionwise_output




### With json schema ###

def remove_additional_properties(schema, extraction = False):
    """
    Recursively remove additionalProperties from a JSON schema.    
    Args:
        schema (dict): The JSON schema to modify    
    Returns:
        dict: Modified schema with additionalProperties removed
    """
    # Parse JSON if string
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except (json.JSONDecodeError, Exception) as e:
            logging.info(f"Google schema parsing failed: {e}")
            return None

    if not extraction:
        if schema.get("format", {}).get("schema", None):
            schema = schema['format']['schema']
            logging.info("Google schema extracted from OpenAI JSON schema.")
        extraction = True

    # Create a deep copy to avoid modifying the original schema
    modified_schema = copy.deepcopy(schema)
    
    # Remove additionalProperties from the current level
    if isinstance(modified_schema, dict):
        # Remove additionalProperties key if it exists
        modified_schema.pop('additionalProperties', None)
        
        # Recursively process nested properties
        for key, value in modified_schema.get('properties', {}).items():
            modified_schema['properties'][key] = remove_additional_properties(value, extraction)
        
        # Recursively process nested items in arrays
        if modified_schema.get('type') == 'array' and 'items' in modified_schema:
            modified_schema['items'] = remove_additional_properties(modified_schema['items'], extraction)
    
    return modified_schema



def call_chatgpt_json(assistant_info, message, model, temperature, max_tokens, top_p, frequency_penalty, 
                 ProviderName, source_path, spk_lbl_content, status_logger, 
                 json_schema=None, schema=None, providerID=None, chartType=None, VisitTypeId=None, counter = 0):
    """
    Call OpenAI ChatGPT for JSON output using LangChain for automatic LangSmith tracing.
    Returns JSON string output and preserves the same return format (output, model_used).
    """
    try:
        # Prepare messages
        system_prompt = f"""{assistant_info}
        You are required to provide the output in the JSON format only"""

        # Use LangChain ChatOpenAI with JSON mode for automatic LangSmith tracing
        llm = get_langchain_openai(
            model=model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty
        )
        
        # Bind JSON response format
        if json_schema:
            # Extract the actual schema from the OpenAI format if present
            actual_schema = json_schema
            if isinstance(json_schema, dict) and json_schema.get("format", {}).get("schema"):
                actual_schema = json_schema["format"]["schema"]
            llm = llm.bind(response_format={"type": "json_schema", "json_schema": {"name": "response", "schema": actual_schema}})
            logging.info("Calling OpenAI with JSON schema via LangChain.")
        else:
            llm = llm.bind(response_format={"type": "json_object"})
        
        # Add metadata for LangSmith tracing
        llm = llm.with_config({
            "run_name": "call_chatgpt_json",
            "metadata": {
                "LLM_Service": "openai",
                "provider_name": ProviderName,
                "providerID": providerID,
                "chartType": chartType,
                "VisitTypeId": VisitTypeId,
                "response_format": "json",
                "retry_attempt": counter + 1,
            }
        })
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ]
        
        response = llm.invoke(messages)
        chat_gpt_output = response.content
        
        return chat_gpt_output, model
        
    except Exception as gpt_error:
        if counter <= 1:
            logging.error(f"{model} failed due to: {str(gpt_error)}, Retrying: {counter+1}")
            if counter > 0:
                model = CHATGPT_MODEL
            return call_chatgpt_json(assistant_info, message, model, temperature, max_tokens, top_p, frequency_penalty, 
                                ProviderName, source_path, spk_lbl_content, status_logger, 
                                json_schema, schema, providerID, chartType, VisitTypeId, counter + 1)
        else:
            logging.error(f"{model} failed due to: {str(gpt_error)}")

            # Insert the file details in the FileProcessingStatus table
            Status = "CHATGPT API FAILED"
            Message = f"ChatGPT {model} failed to generate a report. Error: \n{str(gpt_error)}"
            status_logger(Status, Message)
            
            try:
                # Fallback to Gemini using LangChain for LangSmith tracing
                model = GEMINI_MODEL

                system_prompt = f"""{assistant_info}
                You are required to provide the output in the JSON format only"""

                llm_gemini = get_langchain_gemini(
                    model=model,
                    temperature=temperature,
                    top_p=top_p
                )
                
                # For schema-based output, include schema in system prompt
                if schema:
                    schema_str = json.dumps(schema) if isinstance(schema, dict) else str(schema)
                    system_prompt = f"""{system_prompt}
                    
                    You MUST follow this JSON schema exactly:
                    {schema_str}"""
                    logging.info("Calling Google with Schema (fallback via LangChain).")
                
                llm_gemini = llm_gemini.with_config({
                    "run_name": "call_chatgpt_json_fallback_gemini",
                    "metadata": {
                        "LLM_Service": "google",
                        "provider_name": ProviderName,
                        "providerID": providerID,
                        "chartType": chartType,
                        "VisitTypeId": VisitTypeId,
                        "fallback": True,
                        "original_error": str(gpt_error),
                    }
                })
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"USER PROMPT: {message}")
                ]
                
                response = llm_gemini.invoke(messages)
                gemini_output = response.content
                
                # Clean up the response if it contains markdown code blocks
                if gemini_output.startswith("```json"):
                    gemini_output = gemini_output[7:]
                if gemini_output.startswith("```"):
                    gemini_output = gemini_output[3:]
                if gemini_output.endswith("```"):
                    gemini_output = gemini_output[:-3]
                gemini_output = gemini_output.strip()

                # Insert the file details in the FileProcessingStatus table
                Status = "REPROCESSED WITH GEMINI"
                Message = f"Gemini {model} successfully generated a report after ChatGPT failed."
                status_logger(Status, Message)

                subject = f"[ScribeRYTE PLUS]: SOAP Note Reprocessed with Gemini for Dr. {ProviderName}"
                body = f"""Gemini {model} successfully generated the SOAP note for the following file: \n{source_path} \n\nOriginally, ChatGPT failed to process this file due to the following error: \n{str(gpt_error)} \n\nTranscript: \n{spk_lbl_content}"""
                send_email(subject, body)

                return gemini_output, model

            except Exception as gemini_error:
                logging.error(f"{model} failed due to: {str(gemini_error)}")
                
                subject = f"URGENT [ScribeRYTE PLUS]: ChatGPT & Gemini failed to generate a report for Dr. {ProviderName}"
                body = f"""ChatGPT and Gemini failed to generate a report for the following file: \n{source_path} \n\nChatGPT Error: \n{str(gpt_error)} \n\nGemini Error: \n{str(gemini_error)} \n\nTranscript: \n{spk_lbl_content}"""
                send_email(subject, body)

                # Insert the file details in the FileProcessingStatus table
                Status = "LLMs API FAILED"
                Message = f"LLMs failed to generate a report. Error: \n{str(gpt_error)}\n\n{str(gemini_error)}"
                status_logger(Status, Message)

                raise gemini_error


def call_gemini_json(assistant_info, message, model, temperature, max_tokens, top_p, frequency_penalty, 
                ProviderName, source_path, spk_lbl_content, status_logger, 
                json_schema=None, schema=None, providerID=None, chartType=None, VisitTypeId=None, counter = 0):
    """
    Call Google Gemini for JSON output using LangChain for automatic LangSmith tracing.
    Returns JSON string output and preserves the same return format (output, model_used).
    
    Note: For JSON schema support, we use native Gemini client as LangChain's 
    Gemini integration doesn't fully support all schema features.
    """
    try:
        # Prepare messages
        system_prompt = f"""{assistant_info}
        You are required to provide the output in the JSON format only"""

        # Use LangChain for standard JSON mode
        llm = get_langchain_gemini(
            model=model,
            temperature=temperature,
            top_p=top_p
        )
        
        # Add metadata for LangSmith tracing
        llm = llm.with_config({
            "run_name": "call_gemini_json",
            "metadata": {
                "LLM_Service": "google",
                "provider_name": ProviderName,
                "providerID": providerID,
                "chartType": chartType,
                "VisitTypeId": VisitTypeId,
                "response_format": "json",
                "has_schema": schema is not None,
                "retry_attempt": counter + 1,
            }
        })
        
        # For schema-based output, enhance the system prompt with schema instructions
        if schema:
            schema_str = json.dumps(schema) if isinstance(schema, dict) else str(schema)
            system_prompt = f"""{system_prompt}
            
            You MUST follow this JSON schema exactly:
            {schema_str}"""
            logging.info("Calling Google with Schema via LangChain.")
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"USER PROMPT: {message}")
        ]
        
        response = llm.invoke(messages)
        gemini_output = response.content
        
        # Clean up the response if it contains markdown code blocks
        if gemini_output.startswith("```json"):
            gemini_output = gemini_output[7:]
        if gemini_output.startswith("```"):
            gemini_output = gemini_output[3:]
        if gemini_output.endswith("```"):
            gemini_output = gemini_output[:-3]
        gemini_output = gemini_output.strip()
        
        return gemini_output, model
        
    except Exception as gemini_error:
        if counter <= 1:
            logging.error(f"{model} failed due to: {str(gemini_error)}, Retrying: {counter+1}")
            if counter > 0:
                model = GEMINI_MODEL
            return call_gemini_json(assistant_info, message, model, temperature, max_tokens, top_p, frequency_penalty, 
                               ProviderName, source_path, spk_lbl_content, status_logger, 
                               json_schema, schema, providerID, chartType, VisitTypeId, counter + 1)
        else:
            logging.error(f"{model} failed due to: {str(gemini_error)}")

            # Insert the file details in the FileProcessingStatus table
            Status = "GEMINI API FAILED"
            Message = f"Gemini {model} failed to generate a report. Error: \n{str(gemini_error)}"
            status_logger(Status, Message)

            try:
                # Fallback to ChatGPT using LangChain
                model = CHATGPT_MODEL

                system_prompt = f"""{assistant_info}
                You are required to provide the output in the JSON format only"""

                llm_openai = get_langchain_openai(
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty
                )
                
                # Bind JSON response format
                if json_schema:
                    actual_schema = json_schema
                    if isinstance(json_schema, dict) and json_schema.get("format", {}).get("schema"):
                        actual_schema = json_schema["format"]["schema"]
                    llm_openai = llm_openai.bind(response_format={"type": "json_schema", "json_schema": {"name": "response", "schema": actual_schema}})
                    logging.info("Calling OpenAI with JSON schema (fallback).")
                else:
                    llm_openai = llm_openai.bind(response_format={"type": "json_object"})
                
                llm_openai = llm_openai.with_config({
                    "run_name": "call_gemini_json_fallback_chatgpt",
                    "metadata": {
                        "LLM_Service": "openai",
                        "provider_name": ProviderName,
                        "providerID": providerID,
                        "chartType": chartType,
                        "VisitTypeId": VisitTypeId,
                        "fallback": True,
                        "original_error": str(gemini_error),
                    }
                })
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=message)
                ]
                
                response = llm_openai.invoke(messages)
                chat_gpt_output = response.content

                # Insert the file details in the FileProcessingStatus table
                Status = "REPROCESSED WITH CHATGPT"
                Message = f"ChatGPT {model} successfully generated a report after Gemini failed."
                status_logger(Status, Message)

                subject = f"[ScribeRYTE PLUS]: SOAP Note Reprocessed with ChatGPT for Dr. {ProviderName}"
                body = f"""ChatGPT {model} successfully generated the SOAP note for the following file: \n{source_path} \n\nOriginally, Gemini failed to process this file due to the following error: \n{str(gemini_error)} \n\nTranscript: \n{spk_lbl_content}"""
                send_email(subject, body)

                return chat_gpt_output, model

            except Exception as gpt_error:
                logging.error(f"{model} failed due to: {str(gpt_error)}")
                
                subject = f"URGENT [ScribeRYTE PLUS]: ChatGPT & Gemini failed to generate a report for Dr. {ProviderName}"
                body = f"""ChatGPT and Gemini failed to generate a report for the following file: \n{source_path} \n\nChatGPT Error: \n{str(gpt_error)} \n\nGemini Error: \n{str(gemini_error)} \n\nTranscript: \n{spk_lbl_content}"""
                send_email(subject, body)

                # Insert the file details in the FileProcessingStatus table
                Status = "LLMs API FAILED"
                Message = f"LLMs failed to generate a report. Error: \n{str(gpt_error)}\n\n{str(gemini_error)}"
                status_logger(Status, Message)

                raise gpt_error
            

def json_to_html_with_sections(json_data):
    """
    Convert ChatGPT JSON output to:
    1. HTML string
    2. Section-wise dictionary
    """
    def is_numbered(item):
        return bool(re.match(r"^\s*\d+[\.\)]?\s", str(item).strip()))
    
    # Parse JSON if string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data


    def format_content(content, level=1, parent_key=None):
        html = ""
        if isinstance(content, dict):
            for key, value in content.items():
                if not value:
                    header_level = min(level + 1, 6) # 2 if level == 1 else (4 if level == 2 else 5)
                    html += f"<h{header_level}>{key}</h{header_level}>\n"
                    html += f"<p>Not mentioned.</p>\n"
                    continue

                header_level = min(level + 1, 6) # 2 if level == 1 else (4 if level == 2 else 5)
                html += f"<h{header_level}>{key}</h{header_level}>\n"
                html += format_content(value, level + 1, key)

        elif isinstance(content, list):
            for item in content:
                if isinstance(item, list):
                    sub_lister = item
                    if len(sub_lister) > 1:
                        for index, sub_lines in enumerate(sub_lister):
                            if index == 0:
                                if isinstance(sub_lines, str) and is_numbered(sub_lines):
                                    html += f"<p style='margin-left: 0em;'> {sub_lines}</p>\n"
                                else:
                                    html += f"<p style='margin-left: 0em;'>• {sub_lines}</p>\n"
                            else:
                                html += f"<p style='margin-left: 2em;'>○ {sub_lines}</p>\n"
                    else:
                        if isinstance(item, str) and is_numbered(item):
                            html += f"<p style='margin-left: 0em;'> {item}</p>\n"
                        else:
                            html += f"<p style='margin-left: 0em;'>• {item}</p>\n"
                else:
                    html += format_content(item, level, parent_key)

        else:
            text_lister = (str(content).strip()).split('\n')
            for text in text_lister:
                html += f"<p>{text}</p>\n"

        return html

    html_output = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Medical Note</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
    h1, h2, h3, h4, h5, h6 {{ color: #333; margin-top: 20px; }}
    p {{ margin-bottom: 10px; }}
</style>
</head>
<body>
{format_content(data)}
</body>
</html>"""

    def format_value(value):
        """Convert dict or list to a newline-joined string."""
        if isinstance(value, dict):
            # Flatten dict to "key: value" lines
            return "\n".join(f"{k}: \n{format_value(v)}" for k, v in value.items())
        elif isinstance(value, list):
            formatted_items = []
            for item in value:
                if isinstance(item, dict):
                    formatted_items.append("\n".join(f"{k}: \n{format_value(v)}" for k, v in item.items()))
                else:
                    formatted_items.append(str(item))
            return "\n".join(formatted_items)
        else:
            return f"{str(value)}"

    # Create new dictionary with flattened values
    flattened_dict = {key: format_value(value) for key, value in data.items()}

    sectionwise_output = []
    for key, value in flattened_dict.items():
        sectionwise_output.append({'category': f"{key}", 'content': f"{value}"})

    return html_output, sectionwise_output
    
####################### Wound-Care ##############################


def json_to_html_with_sections_for_wound_care(json_data, schema_name=None):
    """
    Convert ChatGPT JSON output to:
    1. HTML string
    2. Section-wise dictionary
    """
    def is_numbered(item):
        return bool(re.match(r"^\s*\d+[\.\)]?\s", str(item).strip()))

    # Parse JSON if string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = copy.deepcopy(json_data)
    
    # Unwrap top-level key if it's the only key (sometimes LLM wraps output)
    top_level_keys = ["Multi Wound Chart Details", "Follow-Up - Wound Chart Details", "wound_care_visit_schema", "mist_documentation"]
    if isinstance(data, dict) and len(data) == 1:
        key = list(data.keys())[0]
        if key in top_level_keys:
            data = data[key]

    # Use the (potentially unwrapped) data for section-wise output as well
    orig_data_for_sections = data

    # Auto-detect schema if not provided
    if not schema_name:
        if any(k in data for k in ["Wound Entries", "Patient Wound Entries", "Multi Wound Chart Details"]):
            schema_name = "mist_documentation"
        elif any(k in data for k in ["Wound Assessment Table", "wounds", "Follow-Up - Wound Chart Details"]):
            schema_name = "wound_care_visit_schema"


    def render_special_chart(d):
        html = ""
        # 1. Patient Info Table
        if 'patient_information' in d or 'Patient Information' in d:
            pi = d.pop('patient_information', d.pop('Patient Information', {}))
            if not pi: pi = {} # Ensure pi is a dict not None
            p_name = pi.get("patient_name", pi.get("Patient Name", "______________________________"))
            if p_name == "-" or not p_name: p_name = "______________________________"
            
            html += f"<h1>Follow-Up - Wound Chart Details &ndash; {p_name}</h1>\n"
            formatted_pi = {
                "Patient Name": pi.get("patient_name", pi.get("Patient Name", "-")),
                "DOB": pi.get("dob", pi.get("DOB", "-")),
                "Date of Service": pi.get("date_of_service", pi.get("Date of Service", "-")),
                "Physician": pi.get("physician", pi.get("Physician", "-")),
                "Scribe": pi.get("scribe", pi.get("Transcriptionist", "-")),
                "Facility": pi.get("facility", pi.get("Facility", "-"))
            }
            cols = list(formatted_pi.items())
            html += "<h2 class='wc-h2'>Patient Information</h2>\n<table class='patient-info'>\n"
            for i in range(0, len(cols), 2):
                html += "  <tr>\n"
                html += f"    <th>{cols[i][0]}</th><td>{cols[i][1]}</td>\n"
                if i + 1 < len(cols): html += f"    <th>{cols[i+1][0]}</th><td>{cols[i+1][1]}</td>\n"
                else: html += "    <th></th><td></td>\n"
                html += "  </tr>\n"
            html += "</table>\n"
 
        # 2. Wound Assessment Table
        wounds_data = d.pop('wounds', [])
        wat = d.pop('Wound Assessment Table', {})
        if wat and 'headers' in wat and 'rows' in wat:
            html += "<h2 class='wc-h2'>Wound Assessment Table</h2>\n<table class='wound-table'>\n"
            html += "  <thead><tr><th>Field</th>" + "".join(f"<th>{h}</th>" for h in wat['headers'][1:]) + "</tr></thead>\n<tbody>\n"
            for row in wat['rows']:
                if row: html += f"  <tr><th>{row[0]}</th>" + "".join(f"<td>{v}</td>" for v in row[1:]) + "</tr>\n"
            html += "</tbody></table>\n"
        elif wounds_data:
            ATTRS = [
                ("MIST Therapy", "mist_therapy"),
                ("Wound Location", "location"),
                ("Outcome", "outcome"),
                ("Wound Type", "type"),
                ("Wound Status", "status"),
                ("Measurements (L x W x D)", "measurements"),
                ("Area (sq cm)", "area_sq_cm"),
                ("Volume (cm³)", "volume_cu_cm"),
                ("Tunnels", "tunnels"),
                ("Max Depth (cm)", "max_depth"),
                ("Undermining (cm)", "undermining"),
                ("Stage / Grade", "stage_grade"),
                ("Exudate Amount", "exudate_amount"),
                ("Exudate Type", "exudate_type"),
                ("Odor", "odor"),
                ("Wound Margin", "wound_margin"),
                ("Periwound", "periwound"),
                ("Necrotic Material (%)", "necrotic_material"),
                ("Granulation (%)", "granulation"),
                ("Tissue Exposed", "tissue_exposed"),
                ("Debridement", "debridement"),
                ("Primary Dressing", "primary_dressing"),
                ("Secondary Dressing", "secondary_dressing"),
                ("Frequency", "frequency"),
                ("Special Equipment", "special_equipment")
            ]
            wound_headers = [f"<th>Wound {w.get('number', i+1)}</th>" for i, w in enumerate(wounds_data)]
            html += "<h2 class='wc-h2'>Wound Assessment Table</h2>\n<table class='wound-table'>\n"
            html += "  <thead><tr><th>Field</th>" + "".join(wound_headers) + "</tr></thead>\n<tbody>\n"
            for label, key in ATTRS:
                html += f"  <tr><th>{label}</th>" + "".join(f"<td>{w.get(key, '-')}</td>" for w in wounds_data) + "</tr>\n"
            html += "</tbody></table>\n"

        # 3. Detailed Summaries
        summaries = d.pop('Detailed Visit Summaries', d.pop('summaries', []))
        if not summaries and wounds_data:
            summaries = []
            for i, w in enumerate(wounds_data, start=1):
                num = w.get('number', str(i))
                summaries.append({
                    "header": f"🩹 Wound {num}: {w.get('type', '-')}",
                    "narrative": w.get('clinical_summary', w.get('clinical_summary_notes', '-')),
                    "location": w.get('location', '-'),
                    "stage_grade": w.get('stage_grade', '-'),
                    "debridement_sharp": w.get('debridement_sharp'),
                    "debridement_none": w.get('debridement_none'),
                    "debridement_details": w.get('debridement_details', '-'),
                    "primary_dressing": w.get('primary_dressing', '-'),
                    "secondary_dressing": w.get('secondary_dressing', '-'),
                    "offloading_equipment": w.get('offloading_equipment', '-')
                })

        if summaries:
            html += "<h2 class='wc-h2'>Detailed Visit Summaries</h2>\n"
            for s in summaries:
                header = s.get('header', 'Wound Summary')
                narrative = s.get('narrative', s.get('content', '-'))
                html += f"<div class='summary-block'><div class='summary-header'>{header}</div>"
                html += "<div>"
                if 'location' in s: html += f"  <p><strong>Location:</strong> {s['location']}</p>"
                if 'stage_grade' in s: html += f"  <p><strong>Stage:</strong> {s['stage_grade']}</p>"
                if 'primary_dressing' in s and s['primary_dressing'] != "-": html += f"  <p><strong>Primary Dressing:</strong> {s['primary_dressing']}</p>"
                if 'secondary_dressing' in s and s['secondary_dressing'] != "-": html += f"  <p><strong>Secondary Dressing:</strong> {s['secondary_dressing']}</p>"
                if 'offloading_equipment' in s and s['offloading_equipment'] != "-": html += f"  <p><strong>Offloading Equipment:</strong> {s['offloading_equipment']}</p>"
                
                if any(k in s for k in ["debridement_sharp", "debridement_none"]):
                    d_sharp_checked = 'checked' if s.get("debridement_sharp") else ''
                    d_none_checked = 'checked' if s.get("debridement_none") else ''
                    html += f"""
                    <ul style='list-style-type:none; padding-left:15px; margin-top:5px;'>
                        <li><strong>Debridement:</strong>
                            <ul style='list-style-type:none; padding-left:15px; margin-top:5px;'>
                                <li><input type='checkbox' {d_sharp_checked} class='wc-check' onclick='return false;'> Sharp debridement</li>
                                <li><input type='checkbox' {d_none_checked} class='wc-check' onclick='return false;'> No debridement</li>
                                <li><strong>Details:</strong> {s.get('debridement_details', '-')}</li>
                            </ul>
                        </li>
                    </ul>"""

                if narrative and narrative != "-":
                    html += "<div style='margin-top: 10px;'><strong>Clinical Narrative:</strong>"
                    clean_narrative = str(narrative).replace('\\n', '\n')
                    clean_narrative = re.sub(r'(?<=[.\s])\s*(Procedure:)', r'\n\1', clean_narrative)
                    clean_narrative = re.sub(r'(?<=[.\s])\s*(Plan:)', r'\n\1', clean_narrative)
                    for line in clean_narrative.split('\n'):
                        if line.strip(): html += f"<p style='margin: 3px 0; padding-left: 10px;'>{line.strip()}</p>"
                    html += "</div>"
                html += "</div></div>\n"
        
        # 4. Comments & EM
        if 'Provider Comment' in d or 'comments' in d:
            comm = d.pop('Provider Comment', d.pop('comments', "-"))
            if comm and comm != "-":
                clean_val = str(comm).replace('\\n', '<br>')
                html += f"<h2 class='wc-h2'>Provider Comments</h2><div style='padding-left:10px;'>{clean_val}</div>"
        
        if 'EM Justification' in d or 'em_justification' in d:
            em = d.pop('EM Justification', d.pop('em_justification', {}))
            html += "<h2 class='wc-h2'>E/M Justification</h2>\n<div style='padding-left: 10px;'>"
            em_fields = [
                ('Time Spent Examining/Evaluating', 'time_spent_examining'),
                ('Time Spent Documenting', 'time_spent_documenting'),
                ('Time Spent Coordinating', 'time_spent_coordinating'),
                ('Resolved Wound Sign Off', 'resolved_wound_sign_off'),
                ('Total Time', 'total_time')
            ]
            for label, key in em_fields:
                val = em.get(label, em.get(key, '______'))
                html += f"<p>{label}: {val}</p>"
            html += "</div>"

        if d:
            html += "<h2 class='wc-h2'>Additional Information</h2>\n"
            html += format_content(d)
        return html

    def render_mist_chart(data, level=1, parent_key=None):
        html = ""
        # Handle parsed JSON layout - use pop to track rendered keys
        pi = data.pop("Patient Information", {}) or {}
        entries = data.pop("Wound Entries", data.pop("Patient Wound Entries", []))
    
        if not entries and not pi:
            # Fallback
            return "<p>No MIST wound entries found.</p>"
    
        p_name = pi.get("Patient Name", pi.get("patient_name", "______________________________"))
        if p_name == "-": p_name = "______________________________"
        
        html += "<h1 style='text-align:center;'>Multi Wound Chart Details</h1>\n"
        
        # Static Patient Information block matching PDF text layout exactly
        html += "<table class='patient-info-mist' style='width:100%; margin-bottom: 25px;'>\n"
        html += f"  <tr><th>Patient Name:</th><td>{pi.get('Patient Name', '-')}</td><th>Date:</th><td>{pi.get('Date', '-')}</td></tr>\n"
        html += f"  <tr><th>Patient Date of Birth:</th><td>{pi.get('Patient Date of Birth', '-')}</td><th>Physician/Extender:</th><td>{pi.get('Physician/Extender', '-')}</td></tr>\n"
        html += f"  <tr><th>Transcriptionist:</th><td>{pi.get('Transcriptionist', '-')}</td><th>Facility:</th><td>{pi.get('Facility', '-')}</td></tr>\n"
        html += "</table>\n"
    
        # MIST specific attributes requested in PDF template that map to our JSON Schema
        ATTRS = [
            ("Wound Number", "Wound Number"),
            ("MIST Therapy", "MIST Therapy"),
            ("Wound Location", "Wound Location"),
            ("Outcome", "Outcome"),
            ("Wound Type", "Wound Type"),
            ("Wound Status", "Wound Status"),
            ("Measurements L x W x D", "Measurements L x W x D"),
            ("Area (sq cm)", "Area (sq cm)"),
            ("Volume (cm3)", "Volume (cm3)"),
            ("Treatment No.", "Treatment No."),
            ("Time", "Time"),
            ("Tunnels", "Tunnels"),
            ("Max depth of deepest tunnel (cm)", "Max depth of deepest tunnel (cm)"),
            ("Undermining (cm)", "Undermining (cm)"),
            ("Stage or grade if applicable", "Stage or grade if applicable"),
            ("Exudate Amount", "Exudate Amount"),
            ("Exudate Type", "Exudate Type"),
            ("Odor", "Odor"),
            ("Wound Margin", "Wound Margin"),
            ("Periwound", "Periwound"),
            ("Necrotic Material", "Necrotic Material"),
            ("Granulation", "Granulation"),
            ("Tissue Exposed", "Tissue Exposed"),
            ("Debridement", "Debridement"),
            ("MIST indication", "MIST indication"),
            ("Benchmark Justification", "Benchmark Justification"),
            ("NCF", "NCF"),
            ("TO Pre", "TO Pre"),
            ("TO post", "TO post"),
            ("Treatment performed", "Treatment performed"),
            ("PT specific comments/documentation", "PT specific comments/documentation"),
        ]
        
        wound_headers = [f"<th>Wound {w.get('Wound Number', i+1) if isinstance(w, dict) else i+1}</th>" for i, w in enumerate(entries)]
        html += "<table class='wound-table'>\n"
        html += "  <thead><tr><th>Field</th>" + "".join(wound_headers) + "</tr></thead>\n<tbody>\n"
        
        for label, key in ATTRS:
            html += f"  <tr><th>{label}</th>"
            for w in entries:
                val = w.get(key, "-") if isinstance(w, dict) else "-"
                if val is None: val = "-"
                html += f"<td>{val}</td>"
            html += "</tr>\n"
            
        html += "</tbody></table>\n"
        
        physician_rec_raw = data.pop("Physician Recommendation Details", "-")
        if isinstance(physician_rec_raw, list):
            physician_rec = "\n\n".join(str(i) for i in physician_rec_raw)
        else:
            physician_rec = str(physician_rec_raw) if physician_rec_raw else "-"
        
        if physician_rec != "-":
            html += "<h2 class='wc-h2'>Physician Recommendation Details</h2>\n"
            # Split by double newline to handle different wounds
            blocks = physician_rec.split('\n\n')
            for block in blocks:
                if not block.strip(): continue
                block_lines = block.strip().split('\n')
                if not block_lines: continue
                
                # Header: Wound #...
                html += f"<div style='margin-bottom: 20px;'>\n"
                html += f"  <p><strong>{block_lines[0]}</strong></p>\n"
                
                # Sub-headers and content
                for i in range(1, len(block_lines)):
                    line = block_lines[i]
                    if "Procedure:" in line:
                        html += f"  <p style='margin-bottom: 5px;'><strong>Procedure</strong></p>\n"
                        # Use line content if it contains more than just the label
                        line_parts = line.split("Procedure:", 1)
                        line_content = line_parts[1].strip() if len(line_parts) > 1 else ""
                        if line_content and line_content not in ["-", "not mentioned"]:
                            proc_val = line_content
                        else:
                            # Explicitly handle block as string for type inference
                            block_str = str(block)
                            proc_val = "• MIST Therapy" if "mist" in block_str.lower() else "• procedure not mentioned"
                        html += f"  <p style='margin-top: 0; padding-left: 20px;'>{proc_val}</p>\n"
                    elif "mist" in line.lower() or "• procedure not mentioned" in line.lower():
                        continue
                    else:
                        if line.strip():
                            html += f"  <p style='margin-top: 0; padding-left: 20px;'>{line.strip()}</p>\n"
                html += f"</div>\n"
        
        # Append Provider Comments
        provider_comments = data.pop("Provider Comments", "-")
        if provider_comments == "-":
            # Fallback: check if any wound entries have PT specific comments
            all_pt_comments: list[str] = []
            for entry in entries:
                if not isinstance(entry, dict): continue
                cmt = entry.get("PT specific comments/documentation", "-")
                if cmt != "-":
                    all_pt_comments.append(f"Wound #{entry.get('Wound Number', '?')}: {cmt}")
            if all_pt_comments:
                provider_comments = "\n".join(all_pt_comments)

        if provider_comments != "-":
            html += "<h2 class='wc-h2'>Provider Comments</h2>\n"
            clean_val = str(provider_comments).replace('\\n', '<br>').replace('\n', '<br>')
            html += f"<div style='padding-left:10px; font-size: 14px;'>{clean_val}</div>\n"
        
        if data:
            html += "<h2 class='wc-h2'>Additional Information</h2>\n"
            html += format_content(data)
        return html

    def format_content(content, level=1, parent_key=None):
        html = ""
        if isinstance(content, dict):
            for key, value in content.items():
                header_level = min(level + 1, 6)
                html += f"<h{header_level}>{key}</h{header_level}>\n"
                if not value: html += "<p>Not mentioned.</p>\n"
                else: html += format_content(value, level + 1, key)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, list):
                    for idx, line in enumerate(item):
                        margin = "0em" if idx == 0 else "2em"
                        bullet = "• " if idx == 0 and not is_numbered(line) else ("○ " if idx > 0 else "")
                        html += f"<p style='margin-left: {margin};'>{bullet}{line}</p>\n"
                elif isinstance(item, dict):
                    html += "<div class='summary-block'>\n" + format_content(item, level) + "</div>\n"
                else: 
                     # For a standard list of strings, add bullets
                     bullet = "• " if not is_numbered(item) else ""
                     html += f"<p style='margin-left: 0em;'>{bullet}{item}</p>\n"
        else:
            for text in str(content).strip().split('\n'):
                if text.strip(): html += f"<p>{text.strip()}</p>\n"
        return html
 
    # Execute conversion
    if schema_name == "mist_documentation" or "Patient Wound Entries" in data or "Multi Wound Chart Details" in data:
        body_content = render_mist_chart(data)
        title_html = ""
    elif schema_name == "wound_care_visit_schema" or "Wound Assessment Table" in data or "wounds" in data or "Follow-Up - Wound Chart Details" in data:
        body_content = render_special_chart(data)
        title_html = ""
    else:
        body_content = format_content(data)
        title_html = "<h1>MEDICAL NOTE</h1>"
 
    html_output = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><style>body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }} h1, h2, h3, h4, h5, h6 {{ color: #000; margin-top: 25px; }} p {{ margin-bottom: 10px; }} table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; table-layout: fixed; }} th, td {{ border: 1px solid #999; padding: 8px; text-align: left; word-wrap: break-word; font-size: 13px; }} .patient-info th {{ width: 25%; background-color: #f8f9fa; }} .patient-info-mist th {{ width: 22%; text-align: left; border: 1px solid #999; background-color: #fff; font-weight: bold; color: #000; }} .patient-info-mist td {{ width: 28%; border: 1px solid #999; }} .wound-table th {{ background-color: #f8f9fa; width: 30%; }} .summary-block {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 4px; }} .summary-header {{ font-weight: bold; margin-bottom: 8px; font-size: 1.1em; border-bottom: 1px solid #ccc; }} .wc-h2 {{ border-bottom: 1px solid #000; margin-bottom: 15px; font-size: 1.25em; padding-bottom: 5px; }} .wc-check {{ accent-color: #333333; width: 14px; height: 14px; vertical-align: middle; cursor: default; }}</style></head><body>{title_html}{body_content}</body></html>"
 
    def format_value(value):
        if isinstance(value, dict): return "\n".join(f"{k}: \n{format_value(v)}" for k, v in value.items())
        elif isinstance(value, list): return "\n".join(format_value(i) for i in value)
        else: return str(value)
 
    if isinstance(orig_data_for_sections, dict):
        sectionwise_output = [{'category': f"{k}", 'content': f"{format_value(v)}"} for k, v in orig_data_for_sections.items()]
    else:
        sectionwise_output = [{'category': 'Wound Care Report', 'content': f"{format_value(orig_data_for_sections)}"}]
    return html_output, sectionwise_output

####################### Wound-Care ##############################


### HEARO ###


def clean_medical_transcript_json(transcription_string, original_transcript= None, secondary_language_code= None, providerID=None, providerName=None, VisitTypeId=None):
    """
    Clean and correct a medical transcript (speaker-labeled text) using LangChain Gemini 
    for automatic LangSmith tracing.

    This function processes medical transcripts in two scenarios:
    1. **English-only transcripts** (no `secondary_language_code` provided) – 
       The transcript is cleaned for transcription errors, grammar, spelling, 
       punctuation, and formatting, while preserving speaker labels and order.
    2. **Multi-language transcripts** (`secondary_language_code` provided) – 
       Both the translated transcript (English) and the original transcript (possibly 
       multilingual) are compared. The function identifies translation errors, 
       transcription issues, and rephrases poorly translated sentences into 
       clear, natural English while preserving full medical accuracy.

    The output is always a cleaned transcript in **plain text format** with the 
    original speaker labels (`Speaker-1:`, `Speaker-2:`, etc.) and order preserved.

    Args:
        transcription_string (str): The translated or raw speaker-labeled transcript 
            in plain text format. Example:
                "Speaker-1: I thought it was one of the medicine.\n
                 Speaker-2: OK, so he was."
        original_transcript (str, optional): The original multilingual transcript text 
            used as reference when `secondary_language_code` is provided. Defaults to None.
        secondary_language_code (str, optional): Language code of the secondary language 
            present in the original transcript (e.g., "es" for Spanish, "fr" for French). 
            If None, assumes the input transcript is already in English. Defaults to None.

    Returns:
        str: A cleaned and corrected transcript in plain text format, with the same 
        speaker labels and chronological order preserved.

    Raises:
        Exception: Logs and returns the original `transcription_string` if an error 
        occurs during Gemini model processing.
    """
    try:
        logging.info(f"Receieved transcript: {transcription_string[:200]}")
        if secondary_language_code is None:
            logging.info(f"English transcript receieved.")

            system_instruction = """You are an expert medical transcriptionist specializing in cleaning and correcting medical documentation. You will receive a transcript in text format, where each line is labeled with a speaker (e.g., 'Speaker-1:' or 'Speaker-2:')or (e.g., 'spk_0:' or 'spk_1:'). Your task is to meticulously clean and correct the spoken content after the speaker labels.

            CRITICAL INSTRUCTIONS:
            - Preserve the exact speaker labels and order of turns.
            - Correct obvious transcription errors, misspellings, and grammatical mistakes.
            - Maintain all critical medical information, terminology, and formatting.
            - Preserve the chronological flow and natural conversational style of the medical encounter.
            - Do not reorganize, restructure, or summarize the content. Keep it as a cleaned narrative, turn by turn.
            - Use proper punctuation and capitalization for sentences and proper nouns.
            - Convert all numbers, including dates and measurements, into numerals (e.g., 'twenty twenty-five' should be written as 2025).
            - Ensure numbers are transcribed with perfect accuracy.
            - Output must remain in plain text format with the same speaker labels and order."""

            user_message = f"""Please analyze this speaker-labeled medical transcript:

            {transcription_string}

            Your task is to correct the spoken content for each speaker according to the critical instructions. Return the complete cleaned transcript in plain text format, preserving the same speaker labels and order."""

        else:
            logging.info(f"Multi-Language transcript receieved. original_transcript: {original_transcript[:200] if original_transcript else 'None'}")
            
            system_instruction = f"""
            You are an expert medical transcriptionist specializing in cleaning and correcting medical documentation.  
            You will receive a transcript in plain text format, where each line is labeled with a speaker (e.g., 'Speaker-1:' or 'Speaker-2:') or (e.g., 'spk_0:' or 'spk_1:').  

            TYPES OF INPUT:  
            1. **Translated transcript** – some sentences may have been automatically translated from another language (language code: {secondary_language_code}) into English with poor accuracy. These must be rewritten into clear, natural English.  
            2. **Original transcript text** – the raw transcript, which may contain sentences in multiple languages. This transcript may include transcription errors that need correction.  

            YOUR TASK:  
            - First, carefully compare the **Original transcript text** with the **Translated transcript** across the entire conversation (from the first sentence to the last).  
            - Identify transcription errors in the original text and translation errors in the translated version, always keeping the full medical context in mind.  
            - Using both inputs as reference, clean and correct the transcript while preserving the original **speaker labels** and order of dialogue.  
            - Ensure each corrected sentence is fluent, grammatically correct, medically accurate, and professionally formatted.  
            - For poorly translated sentences, rephrase them into clear, natural English while strictly preserving medical meaning, terminology, procedures, and medication names.  
            - Remember that sentences may span multiple speaker turns: you must maintain context and consistency across the entire transcript.  

            CRITICAL INSTRUCTIONS:  
            - Output must remain in **plain text format** with the same speaker labels and order preserved.  
            - Do not remove, rename, or modify speaker labels (e.g., keep "Speaker-1:" as is).  
            - Correct only the spoken content after the speaker labels.  
            - Correct spelling, transcription errors, and grammar issues.  
            - Preserve all medical details, including terminology, procedures, diagnoses, and medication names.  
            - Maintain the chronological sequence and conversational flow exactly as given.  
            - Do not summarize, merge, expand, or shorten content.  
            - Apply correct punctuation, capitalization, and formatting consistently.  
            - Convert all numbers (e.g., dates, ages, and measurements) into numerals (e.g., "twenty twenty-five" → "2025").  
            - Ensure absolute accuracy in recording numbers.  
            - If a translation or transcription is unclear, correct grammar and structure but stay as close as possible to the original meaning without inventing new information.  

            FINAL REQUIREMENT:  
            Return the fully corrected transcript in plain text, with the original **speaker labels and order** fully preserved.
            """


            user_message = f"""
            Please analyze this medical transcript and clean it according to the system instructions.  

            Translated transcript:  
            {transcription_string}  

            Original transcript text:  
            {original_transcript}  

            Your task:  
            - Correct the spoken content for each line while preserving the same speaker labels and order.  
            - Follow all critical instructions provided in the system prompt.  
            - Return the fully corrected transcript in plain text format, keeping speaker labels unchanged.  
            """

        model = GEMINI_MODEL

        # Use LangChain ChatGoogleGenerativeAI for automatic LangSmith tracing
        llm = get_langchain_gemini(
            model=model,
            temperature=temperature,
            top_p=top_p
        )
        
        # Add metadata for LangSmith tracing
        llm = llm.with_config({
            "run_name": "HEARO_transcription_cleaning",
            "metadata": {
                "LLM_Service": "google",
                "Functionality": "HEARO",
                "providerID": providerID,
                "providerName": providerName,
                "VisitTypeId": VisitTypeId,
                "secondary_language_code": secondary_language_code,
            }
        })
        
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"USER PROMPT: {user_message}")
        ]
        
        response = llm.invoke(messages)
        gemini_output = response.content

        if gemini_output:
            logging.info("Successfully cleaned transcript with Gemini via LangChain.")
            return gemini_output
        else:
            logging.info(f"Problem in Gemini output, sending back original transcript. Response text: {gemini_output}")
            return transcription_string

    except Exception as e:
        error_msg = f"An unexpected error occurred during Gemini processing: {str(e)}"
        logging.error(error_msg)
        return transcription_string



# def clean_medical_transcript_json(transcription_string, original_transcript= None, secondary_language_code= None):
#     """
#     Clean and correct a medical transcript (speaker-labeled text) using an LLM (ChatGPT).

#     This function processes medical transcripts in two scenarios:
#     1. **English-only transcripts** (no `secondary_language_code` provided) – 
#        The transcript is cleaned for transcription errors, grammar, spelling, 
#        punctuation, and formatting, while preserving speaker labels and order.
#     2. **Multi-language transcripts** (`secondary_language_code` provided) – 
#        Both the translated transcript (English) and the original transcript (possibly 
#        multilingual) are compared. The function identifies translation errors, 
#        transcription issues, and rephrases poorly translated sentences into 
#        clear, natural English while preserving full medical accuracy.

#     The output is always a cleaned transcript in **plain text format** with the 
#     original speaker labels (`Speaker-1:`, `Speaker-2:`, etc.) and order preserved.

#     Args:
#         transcription_string (str): The translated or raw speaker-labeled transcript 
#             in plain text format. Example:
#                 "Speaker-1: I thought it was one of the medicine.\n
#                  Speaker-2: OK, so he was."
#         original_transcript (str, optional): The original multilingual transcript text 
#             used as reference when `secondary_language_code` is provided. Defaults to None.
#         secondary_language_code (str, optional): Language code of the secondary language 
#             present in the original transcript (e.g., "es" for Spanish, "fr" for French). 
#             If None, assumes the input transcript is already in English. Defaults to None.

#     Returns:
#         str: A cleaned and corrected transcript in plain text format, with the same 
#         speaker labels and chronological order preserved.

#     Raises:
#         Exception: Logs and returns the original `transcription_string` if an error 
#         occurs during ChatGPT model processing.
#     """
#     try:
        
#         if secondary_language_code is None:
#             logging.info(f"English transcript receieved.")

#             system_instruction = """You are an expert medical transcriptionist specializing in cleaning and correcting medical documentation. You will receive a transcript in text format, where each line is labeled with a speaker (e.g., 'Speaker-1:' or 'Speaker-2:'). Your task is to meticulously clean and correct the spoken content after the speaker labels.

#             CRITICAL INSTRUCTIONS:
#             - Preserve the exact speaker labels and order of turns.
#             - Correct obvious transcription errors, misspellings, and grammatical mistakes.
#             - Maintain all critical medical information, terminology, and formatting.
#             - Preserve the chronological flow and natural conversational style of the medical encounter.
#             - Do not reorganize, restructure, or summarize the content. Keep it as a cleaned narrative, turn by turn.
#             - Use proper punctuation and capitalization for sentences and proper nouns.
#             - Convert all numbers, including dates and measurements, into numerals (e.g., 'twenty twenty-five' should be written as 2025).
#             - Ensure numbers are transcribed with perfect accuracy.
#             - Output must remain in plain text format with the same speaker labels and order."""

#             user_message = f"""Please analyze this speaker-labeled medical transcript:

#             {transcription_string}

#             Your task is to correct the spoken content for each speaker according to the critical instructions. Return the complete cleaned transcript in plain text format, preserving the same speaker labels and order."""

#         else:
#             logging.info(f"Multi-Language transcript receieved.")
            
#             system_instruction = f"""
#             You are an expert medical transcriptionist specializing in cleaning and correcting medical documentation.  
#             You will receive a transcript in plain text format, where each line is labeled with a speaker (e.g., 'Speaker-1:' or 'Speaker-2:').  

#             TYPES OF INPUT:  
#             1. **Translated transcript** – some sentences may have been automatically translated from another language (language code: {secondary_language_code}) into English with poor accuracy. These must be rewritten into clear, natural English.  
#             2. **Original transcript text** – the raw transcript, which may contain sentences in multiple languages. This transcript may include transcription errors that need correction.  

#             YOUR TASK:  
#             - First, carefully compare the **Original transcript text** with the **Translated transcript** across the entire conversation (from the first sentence to the last).  
#             - Identify transcription errors in the original text and translation errors in the translated version, always keeping the full medical context in mind.  
#             - Using both inputs as reference, clean and correct the transcript while preserving the original **speaker labels** and order of dialogue.  
#             - Ensure each corrected sentence is fluent, grammatically correct, medically accurate, and professionally formatted.  
#             - For poorly translated sentences, rephrase them into clear, natural English while strictly preserving medical meaning, terminology, procedures, and medication names.  
#             - Remember that sentences may span multiple speaker turns: you must maintain context and consistency across the entire transcript.  

#             CRITICAL INSTRUCTIONS:  
#             - Output must remain in **plain text format** with the same speaker labels and order preserved.  
#             - Do not remove, rename, or modify speaker labels (e.g., keep "Speaker-1:" as is).  
#             - Correct only the spoken content after the speaker labels.  
#             - Correct spelling, transcription errors, and grammar issues.  
#             - Preserve all medical details, including terminology, procedures, diagnoses, and medication names.  
#             - Maintain the chronological sequence and conversational flow exactly as given.  
#             - Do not summarize, merge, expand, or shorten content.  
#             - Apply correct punctuation, capitalization, and formatting consistently.  
#             - Convert all numbers (e.g., dates, ages, and measurements) into numerals (e.g., "twenty twenty-five" → "2025").  
#             - Ensure absolute accuracy in recording numbers.  
#             - If a translation or transcription is unclear, correct grammar and structure but stay as close as possible to the original meaning without inventing new information.  

#             FINAL REQUIREMENT:  
#             Return the fully corrected transcript in plain text, with the original **speaker labels and order** fully preserved.
#             """


#             user_message = f"""
#             Please analyze this medical transcript and clean it according to the system instructions.  

#             Translated transcript:  
#             {transcription_string}  

#             Original transcript text:  
#             {original_transcript}  

#             Your task:  
#             - Correct the spoken content for each line while preserving the same speaker labels and order.  
#             - Follow all critical instructions provided in the system prompt.  
#             - Return the fully corrected transcript in plain text format, keeping speaker labels unchanged.  
#             """


#         model = CHATGPT_MODEL

#         # Prepare messages
#         system_prompt = f"""{system_instruction}
#         You are required to provide the output in the JSON format only"""

#         input_msg = [
#             {"role": "developer", "content": system_prompt},
#             {"role": "user", "content": user_message}
#         ]

#         # Prepare the API call arguments
#         api_args = {
#             "temperature": temperature,
#             "model": model,
#             "input": input_msg,
#             "top_p": top_p,
#             "store": False,
#             "truncation": "disabled",
#             # "reasoning": {"effort": "medium"},
#             # "text": response_format
#         }        
       
#         # Make the API call
#         response = openai_client.responses.create(**api_args)
#         chat_gpt_output = response.output_text 
#         logging.info("Successfully cleaned transcript with ChatGPT.")
#         return chat_gpt_output

#     except Exception as e:
#         error_msg = f"An unexpected error occurred during ChatGPT processing: {str(e)}"
#         logging.error(error_msg)
#         return transcription_string


# def clean_medical_transcript_json(transcript_json, original_transcript_json= None, secondary_language_code= None):
#     """
#     Clean and correct a medical transcript in JSON format using the ChatGPT LLM.

#     This function processes JSON-based medical transcripts and returns a cleaned version
#     with corrected 'sentence' values while preserving the original structure and speaker labels.

#     It supports two modes of operation:
    
#     1. **English-only transcripts** (no `secondary_language_code` provided):
#        - Cleans transcription errors, grammar, spelling, and formatting in the transcript.
#        - Preserves the exact JSON structure, including speaker labels and chronological order.

#     2. **Multi-language transcripts** (`secondary_language_code` provided):
#        - Takes both a translated JSON transcript (English) and the original transcript text.
#        - Compares the original and translated transcripts to identify transcription 
#          and translation errors.
#        - Rewrites poorly translated sentences into clear, natural English while strictly
#          preserving medical accuracy, terminology, and context.

#     The output is a fully corrected JSON array that preserves the original input structure.

#     Args:
#         transcript_json (list[dict]): The input transcript as a JSON array of objects,
#             where each object contains:
#                 {
#                     "speaker": str,
#                     "sentence": str
#                 }
#         original_transcript_json (dict, optional): The original transcript (may contain 
#             multiple languages), typically structured with a key such as 
#             `original_transcript_json['results']['transcripts'][0]['transcript']`.
#             Used only when `secondary_language_code` is provided. Defaults to None.
#         secondary_language_code (str, optional): Language code of the secondary language 
#             in the original transcript (e.g., "es" for Spanish, "fr" for French). 
#             If None, assumes transcript is English-only. Defaults to None.

#     Returns:
#         list[dict]: A cleaned transcript in JSON array format, with the same structure 
#         and speaker labels as the input. Example:
#             [
#                 {"speaker": "Speaker-1", "sentence": "I thought it was one of the medicines."},
#                 {"speaker": "Speaker-2", "sentence": "OK, so he was."}
#             ]

#     Raises:
#         json.JSONDecodeError: If ChatGPT returns an invalid or unparsable JSON response.
#         Exception: For any unexpected errors during ChatGPT processing. In both cases, 
#         the function logs the error and returns the original `transcript_json`.

#     Notes:
#         - All corrections preserve medical details, chronology, and conversational flow.
#         - Numbers are consistently converted to numerals (e.g., "twenty twenty-five" → "2025").
#         - The schema enforces that the output is always a JSON array of objects with
#           'speaker' and 'sentence' keys.
#     """
#     try:
#         transcript_json_string = json.dumps(transcript_json, indent=2)
        
#         if secondary_language_code is None:
#             logging.info(f"English transcript receieved.")

#             system_instruction = """You are an expert medical transcriptionist specializing in cleaning and correcting medical documentation. You will receive a JSON array of objects, where each object represents a part of a conversation with a 'speaker' and a 'sentence'. Your task is to meticulously clean and correct the 'sentence' value for each object.

#             CRITICAL INSTRUCTIONS:
#             - Return the entire JSON array in the exact same structure as the input. Only the 'sentence' values should be modified.
#             - Correct obvious transcription errors, misspellings, and grammatical mistakes.
#             - Maintain all critical medical information, terminology, and formatting.
#             - Preserve the chronological flow and natural conversational style of the medical encounter.
#             - Do not reorganize, restructure, or summarize the content. Keep it as a cleaned narrative, turn by turn.
#             - Use proper punctuation and capitalization for sentences and proper nouns.
#             - Convert all numbers, including dates and measurements, into numerals (e.g., 'twenty twenty-five' should be written as 2025).
#             - Ensure numbers are transcribed with perfect accuracy."""

#             user_message = f"""Please analyze this JSON transcript of a medical audio encounter:
#             {transcript_json_string}

#             Your task is to correct the 'sentence' for each object in the JSON array according to the critical instructions. Return the complete, corrected JSON array. Do not alter the structure or the 'speaker' keys."""

#         else:
#             logging.info(f"Multi-Language transcript receieved.")
            
#             system_instruction = f"""
#             You are an expert medical transcriptionist specializing in cleaning and correcting medical documentation.  
#             You will receive a JSON array of objects, where each object contains two keys: 'speaker' and 'sentence'.  

#             TYPES OF INPUT:  
#             1. **Translated JSON transcription** – some sentences may have been automatically translated from another language (language code: {secondary_language_code}) into English with poor accuracy. These must be rewritten into clear, natural English.  
#             2. **Original transcript text** – the raw transcript, which may contain sentences in multiple languages. This transcript may include transcription errors that need correction.  

#             YOUR TASK:  
#             - First, carefully compare the **Original transcript text** with the **Translated JSON transcription** across the entire conversation (from the first sentence to the last).  
#             - Identify transcription errors in the original text and translation errors in the JSON, always keeping the full medical context in mind.  
#             - Using both inputs as reference, clean and correct only the values under the 'sentence' key in the JSON array.  
#             - Ensure each corrected sentence is fluent, grammatically correct, medically accurate, and professionally formatted.  
#             - For poorly translated sentences, rephrase them into clear, natural English while strictly preserving medical meaning, terminology, procedures, and medication names.  
#             - Remember that sentences are split across multiple objects: you must maintain context and consistency across the entire transcript, not just within individual objects.  

#             CRITICAL INSTRUCTIONS:  
#             - Output must be the **exact same JSON array structure** as the input. Do not add, remove, or reorder objects.  
#             - Modify only the 'sentence' values.  
#             - Correct spelling, transcription errors, and grammar issues.  
#             - Preserve all medical details, including terminology, procedures, diagnoses, and medication names.  
#             - Maintain the chronological sequence and conversational flow exactly as given.  
#             - Do not summarize, merge, expand, or shorten content.  
#             - Apply correct punctuation, capitalization, and formatting consistently.  
#             - Convert all numbers (e.g., dates, ages, and measurements) into numerals (e.g., "twenty twenty-five" → "2025").  
#             - Ensure absolute accuracy in recording numbers.  
#             - If a translation or transcription is unclear, correct grammar and structure but stay as close as possible to the original meaning without inventing new information.  

#             FINAL REQUIREMENT:  
#             Return only the corrected JSON array with the original structure fully preserved.  
#             """

#             user_message = f"""
#             Please analyze this medical transcript and clean it according to the system instructions.  

#             Translated JSON transcript:  
#             {transcript_json_string}  

#             Original transcript text:  
#             {original_transcript_json['results']['transcripts'][0]['transcript']}  

#             Your task:  
#             - Correct the 'sentence' field for each object in the JSON array.  
#             - Follow all critical instructions provided in the system prompt.  
#             - Return the fully corrected JSON array without altering its structure or the 'speaker' keys.  
#             """


#         model = CHATGPT_MODEL

#         # Prepare messages
#         system_prompt = f"""{system_instruction}
#         You are required to provide the output in the JSON format only"""

#         input_msg = [
#             {"role": "developer", "content": system_prompt},
#             {"role": "user", "content": user_message}
#         ]

#         # if json_schema:
#         #     response_format = json_schema
#         #     logging.info("Calling OpenAI with JSON schema.")
#         # else:
#         response_format = {"format": {"type": "json_object"}}

#         # Prepare the API call arguments
#         api_args = {
#             "temperature": temperature,
#             "model": model,
#             "input": input_msg,
#             "top_p": top_p,
#             "store": False,
#             "truncation": "disabled",
#             # "reasoning": {"effort": "medium"},
#             "text": response_format
#         }        
       
#         # Make the API call
#         response = openai_client.responses.create(**api_args)
#         chat_gpt_output = response.output_text
       
#         if chat_gpt_output:
#             cleaned_transcript = json.loads(chat_gpt_output)
#             logger.info("Successfully cleaned transcript with ChatGPT.")
#             return cleaned_transcript
#         else:
#             logger.info(f"Problem in ChatGPT output, sending back original transcript. Response text: {chat_gpt_output}")
#             return transcript_json

#     except json.JSONDecodeError as e:
#         logger.error(f"Failed to parse ChatGPT JSON response. Error: {str(e)}. Response text: {chat_gpt_output}")
#         return transcript_json
#     except Exception as e:
#         logger.error(f"An unexpected error occurred during ChatGPT processing: {str(e)}", exc_info=True)
#         return transcript_json



# def clean_medical_transcript_json(transcript_json, original_transcript_json= None, secondary_language_code= None):
#     """
#     Clean and correct a medical transcript in JSON format using the Gemini LLM.

#     This function processes JSON-based medical transcripts and returns a cleaned version
#     with corrected 'sentence' values while preserving the original structure and speaker labels.

#     It supports two modes of operation:
    
#     1. **English-only transcripts** (no `secondary_language_code` provided):
#        - Cleans transcription errors, grammar, spelling, and formatting in the transcript.
#        - Preserves the exact JSON structure, including speaker labels and chronological order.

#     2. **Multi-language transcripts** (`secondary_language_code` provided):
#        - Takes both a translated JSON transcript (English) and the original transcript text.
#        - Compares the original and translated transcripts to identify transcription 
#          and translation errors.
#        - Rewrites poorly translated sentences into clear, natural English while strictly
#          preserving medical accuracy, terminology, and context.

#     The output is a fully corrected JSON array that preserves the original input structure.

#     Args:
#         transcript_json (list[dict]): The input transcript as a JSON array of objects,
#             where each object contains:
#                 {
#                     "speaker": str,
#                     "sentence": str
#                 }
#         original_transcript_json (dict, optional): The original transcript (may contain 
#             multiple languages), typically structured with a key such as 
#             `original_transcript_json['results']['transcripts'][0]['transcript']`.
#             Used only when `secondary_language_code` is provided. Defaults to None.
#         secondary_language_code (str, optional): Language code of the secondary language 
#             in the original transcript (e.g., "es" for Spanish, "fr" for French). 
#             If None, assumes transcript is English-only. Defaults to None.

#     Returns:
#         list[dict]: A cleaned transcript in JSON array format, with the same structure 
#         and speaker labels as the input. Example:
#             [
#                 {"speaker": "Speaker-1", "sentence": "I thought it was one of the medicines."},
#                 {"speaker": "Speaker-2", "sentence": "OK, so he was."}
#             ]

#     Raises:
#         json.JSONDecodeError: If Gemini returns an invalid or unparsable JSON response.
#         Exception: For any unexpected errors during Gemini processing. In both cases, 
#         the function logs the error and returns the original `transcript_json`.

#     Notes:
#         - All corrections preserve medical details, chronology, and conversational flow.
#         - Numbers are consistently converted to numerals (e.g., "twenty twenty-five" → "2025").
#         - The schema enforces that the output is always a JSON array of objects with
#           'speaker' and 'sentence' keys.
#     """
#     try:
#         transcript_json_string = json.dumps(transcript_json, indent=2)
        
#         if secondary_language_code is None:
#             logging.info(f"English transcript receieved.")

#             system_instruction = """You are an expert medical transcriptionist specializing in cleaning and correcting medical documentation. You will receive a JSON array of objects, where each object represents a part of a conversation with a 'speaker' and a 'sentence'. Your task is to meticulously clean and correct the 'sentence' value for each object.

#             CRITICAL INSTRUCTIONS:
#             - Return the entire JSON array in the exact same structure as the input. Only the 'sentence' values should be modified.
#             - Correct obvious transcription errors, misspellings, and grammatical mistakes.
#             - Maintain all critical medical information, terminology, and formatting.
#             - Preserve the chronological flow and natural conversational style of the medical encounter.
#             - Do not reorganize, restructure, or summarize the content. Keep it as a cleaned narrative, turn by turn.
#             - Use proper punctuation and capitalization for sentences and proper nouns.
#             - Convert all numbers, including dates and measurements, into numerals (e.g., 'twenty twenty-five' should be written as 2025).
#             - Ensure numbers are transcribed with perfect accuracy."""

#             user_message = f"""Please analyze this JSON transcript of a medical audio encounter:
#             {transcript_json_string}

#             Your task is to correct the 'sentence' for each object in the JSON array according to the critical instructions. Return the complete, corrected JSON array. Do not alter the structure or the 'speaker' keys."""

#         else:
#             logging.info(f"Multi-Language transcript receieved.")
            
#             system_instruction = f"""
#             You are an expert medical transcriptionist specializing in cleaning and correcting medical documentation.  
#             You will receive a JSON array of objects, where each object contains two keys: 'speaker' and 'sentence'.  

#             TYPES OF INPUT:  
#             1. **Translated JSON transcription** – some sentences may have been automatically translated from another language (language code: {secondary_language_code}) into English with poor accuracy. These must be rewritten into clear, natural English.  
#             2. **Original transcript text** – the raw transcript, which may contain sentences in multiple languages. This transcript may include transcription errors that need correction.  

#             YOUR TASK:  
#             - First, carefully compare the **Original transcript text** with the **Translated JSON transcription** across the entire conversation (from the first sentence to the last).  
#             - Identify transcription errors in the original text and translation errors in the JSON, always keeping the full medical context in mind.  
#             - Using both inputs as reference, clean and correct only the values under the 'sentence' key in the JSON array.  
#             - Ensure each corrected sentence is fluent, grammatically correct, medically accurate, and professionally formatted.  
#             - For poorly translated sentences, rephrase them into clear, natural English while strictly preserving medical meaning, terminology, procedures, and medication names.  
#             - Remember that sentences are split across multiple objects: you must maintain context and consistency across the entire transcript, not just within individual objects.  

#             CRITICAL INSTRUCTIONS:  
#             - Output must be the **exact same JSON array structure** as the input. Do not add, remove, or reorder objects.  
#             - Modify only the 'sentence' values.  
#             - Correct spelling, transcription errors, and grammar issues.  
#             - Preserve all medical details, including terminology, procedures, diagnoses, and medication names.  
#             - Maintain the chronological sequence and conversational flow exactly as given.  
#             - Do not summarize, merge, expand, or shorten content.  
#             - Apply correct punctuation, capitalization, and formatting consistently.  
#             - Convert all numbers (e.g., dates, ages, and measurements) into numerals (e.g., "twenty twenty-five" → "2025").  
#             - Ensure absolute accuracy in recording numbers.  
#             - If a translation or transcription is unclear, correct grammar and structure but stay as close as possible to the original meaning without inventing new information.  

#             FINAL REQUIREMENT:  
#             Return only the corrected JSON array with the original structure fully preserved.  
#             """

#             user_message = f"""
#             Please analyze this medical transcript and clean it according to the system instructions.  

#             Translated JSON transcript:  
#             {transcript_json_string}  

#             Original transcript text:  
#             {original_transcript_json['results']['transcripts'][0]['transcript']}  

#             Your task:  
#             - Correct the 'sentence' field for each object in the JSON array.  
#             - Follow all critical instructions provided in the system prompt.  
#             - Return the fully corrected JSON array without altering its structure or the 'speaker' keys.  
#             """


#         model = GEMINI_MODEL

#         # Prepare messages
#         system_prompt = f"""{system_instruction}"""

#         contents_for_api = f"""USER PROMPT: {user_message}"""        

#         # Define the exact output schema Gemini must follow
#         gemini_output_schema = types.Schema(
#             type=types.Type.ARRAY,
#             items=types.Schema(
#                 type=types.Type.OBJECT,
#                 properties={
#                     'speaker': types.Schema(type=types.Type.STRING),
#                     'sentence': types.Schema(type=types.Type.STRING)
#                 },
#                 required=['speaker', 'sentence']
#             )
#         )

#         generation_config = types.GenerateContentConfig(
#             response_mime_type='application/json',
#             response_schema=gemini_output_schema,
#             temperature=temperature,
#             max_output_tokens=max_tokens*10,
#             top_p=top_p,
#             system_instruction=system_prompt
#         )          

#         response = gemini_client.models.generate_content(
#             model=model,
#             contents=contents_for_api,
#             config=generation_config
#         )

#         gemini_output = response.text
       
#         if gemini_output:
#             cleaned_transcript = json.loads(gemini_output)
#             logger.info("Successfully cleaned transcript with Gemini.")
#             return cleaned_transcript
#         else:
#             logger.info(f"Problem in Gemini output, sending back original transcript. Response text: {gemini_output}")
#             return transcript_json

#     except json.JSONDecodeError as e:
#         logger.error(f"Failed to parse Gemini JSON response. Error: {str(e)}. Response text: {gemini_output}")
#         return transcript_json
#     except Exception as e:
#         logger.error(f"An unexpected error occurred during Gemini processing: {str(e)}")
#         return transcript_json
