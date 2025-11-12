import streamlit as st
import httpx
import asyncio
import json
import PyPDF2
import docx
import io
import time
from typing import List, Dict, Any

# --- 1. AI CONFIGURATION (Google Gemini) ---
# The `API_KEY` is where you paste your key from Google AI Studio
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key="
API_KEY = "AIzaSyCLMmmA49qite4dPR0lTqOMEBwh_oKx3Iw" 

# --- AI PROMPTS AND SCHEMAS (NOW UPGRADED) ---

# NEW: Job schema now includes education requirements
JD_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "job_title": {"type": "STRING"},
        "experience_level": {"type": "STRING", "description": "e.g., Entry-Level, Senior"},
        "min_experience_years": {"type": "NUMBER", "description": "The minimum number of years of experience required."},
        "education_requirement": {
            "type": "STRING",
            "description": "e.g., 'None', 'Bachelor''s', 'Master''s', 'PhD'"
        },
        "required_skills": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List of core, must-have skills."
        },
        "nice_to_have_skills": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List of 'plus' or 'bonus' skills."
        }
    },
    "required": ["job_title", "min_experience_years", "education_requirement", "required_skills"]
}

JD_SYSTEM_PROMPT = """
You are an expert HR analyst. Extract key information from this job description.
Return a JSON object *only* following this schema. Do not add any extra text.
'education_requirement' is the *minimum* level required (e.g., Bachelor's).
"""

# NEW: Resume schema now includes projects, certs, education, and the AI Signal
RESUME_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "suggested_role": {"type": "STRING", "description": "A likely job title, e.g., Senior Software Engineer"},
        "total_experience_years": {"type": "NUMBER", "description": "The total number of years of professional experience."},
        "education_level": {
            "type": "STRING",
            "description": "The highest level of education achieved (e.g., 'None', 'Bachelor''s', 'Master''s', 'PhD')."
        },
        "skills": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "A comprehensive list of all skills found."
        },
        "certifications": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List of professional certifications."
        },
        "projects": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "description": {"type": "STRING"}
                }
            },
            "description": "List of personal or professional projects."
        },
        "ai_signal": {
            "type": "OBJECT",
            "properties": {
                "score_adjustment": {"type": "NUMBER", "description": "A score from -10 (red flags) to +10 (green flags)."},
                "reasoning": {"type": "STRING", "description": "Brief 1-2 sentence reasoning for the score."}
            },
            "description": "Your (Gemini's) expert analysis of the resume's quality, clarity, career progression, and any red/green flags."
        }
    },
    "required": ["suggested_role", "total_experience_years", "education_level", "skills", "certifications", "projects", "ai_signal"]
}

RESUME_SYSTEM_PROMPT = """
You are an expert HR analyst. Extract key information from this resume.
Pay close attention to all fields.
For 'ai_signal', analyze the resume's overall quality. Is it well-written? 
Are there large unexplained gaps? Does the career show clear progression? 
Provide a 'score_adjustment' (a number between -10 and +10) and brief 'reasoning'.
Return a JSON object *only* following this schema. Do not add any extra text.
"""

CHATBOT_SYSTEM_PROMPT = """
You are an expert HR recruitment assistant. Your goal is to help a recruiter analyze
a job and a set of candidates.

Using the provided JSON data (context) ONLY:
1. The 'job_description' JSON shows the role's requirements.
2. The 'candidates' JSON shows the parsed data for all applicants, including skills,
   experience, projects, certifications, and your AI Signal analysis.

Answer the user's question based *strictly* on this data.
If the user asks for an opinion (e.g., "Who is the best?"), use the data 
to make a recommendation (e.g., "Based on the 7-factor Efficiency Score, [Candidate X] 
is the strongest. They have a high skill match and a positive AI Signal.").
If the data is insufficient, say so.
"""

# --- 2. ASYNC API CALL FUNCTIONS ---

async def call_gemini_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to call the Gemini API with retry logic."""
    headers = {'Content-Type': 'application/json'}
    url = f"{API_URL}{API_KEY}"
    
    # Check if API_KEY is set
    if not API_KEY:
        raise ValueError("API_KEY is not set. Please add your Google AI API key at the top of the script.")

    async with httpx.AsyncClient(timeout=90.0) as client:
        retries = 3
        delay = 2
        for attempt in range(retries):
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                # Check for empty or malformed response
                if not result.get("candidates") or not result["candidates"][0].get("content"):
                    raise Exception("Invalid AI response format.")

                json_text = result["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(json_text)
            
            except httpx.HTTPStatusError as e:
                # Handle 403 Forbidden specifically
                if e.response.status_code == 403:
                    st.error("ERROR: 403 Forbidden. Your API key is likely invalid or has not been enabled. Please check your Google AI Studio settings.")
                    raise
                
                print(f"API call error (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay * (2 ** attempt))
                else:
                    raise
            except Exception as e:
                print(f"Non-HTTP error: {e}")
                raise

async def parse_with_ai(text_content: str, parser_type: str) -> Dict[str, Any]:
    """Parses text using the AI (Job or Resume)."""
    if parser_type == 'job':
        system_prompt = JD_SYSTEM_PROMPT
        json_schema = JD_JSON_SCHEMA
    else:
        system_prompt = RESUME_SYSTEM_PROMPT
        json_schema = RESUME_JSON_SCHEMA

    payload = {
        "contents": [{"parts": [{"text": text_content}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": json_schema
        }
    }
    return await call_gemini_api(payload)

async def get_ai_chat_response(question: str, context: str) -> str:
    """Gets a chat-based answer from the AI."""
    payload = {
        "contents": [{"parts": [{"text": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"}]}],
        "systemInstruction": {"parts": [{"text": CHATBOT_SYSTEM_PROMPT}]}
    }
    
    headers = {'Content-Type': 'application/json'}
    url = f"{API_URL}{API_KEY}"
    
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"Chatbot API error: {e}")
            return f"Error: Could not get a response from the AI. {e}"

# --- 3. FILE READING FUNCTIONS (Unchanged) ---

def read_pdf(file_stream: io.BytesIO) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def read_docx(file_stream: io.BytesIO) -> str:
    try:
        document = docx.Document(file_stream)
        text = ""
        for para in document.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

# --- 4. CORE APP LOGIC (NEW SCORING) ---

def education_level_to_int(level_str: str) -> int:
    """Converts education string to a comparable integer."""
    level = (level_str or "none").lower()
    if "phd" in level:
        return 3
    if "master" in level:
        return 2
    if "bachelor" in level:
        return 1
    return 0

def calculate_efficiency_score(job_data: Dict[str, Any], cand_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates the new 7-Factor Efficiency Score.
    Weights:
    - 40%: Required Skills
    - 20%: Experience (Years)
    - 10%: Project Work
    - 10%: AI Signal
    - 10%: Education
    - 5%: Certifications
    - 5%: Bonus Skills
    """
    
    # --- 1. Required Skills (40%) ---
    job_req_skills = set([s.lower() for s in job_data.get('required_skills', [])])
    cand_skills = set([s.lower() for s in cand_data.get('skills', [])])
    matched_req_skills = job_req_skills.intersection(cand_skills)
    missing_req_skills = job_req_skills.difference(cand_skills)
    skill_match_score = 0
    if len(job_req_skills) > 0:
        skill_match_score = len(matched_req_skills) / len(job_req_skills)
    
    # --- 2. Experience (20%) ---
    job_exp = job_data.get('min_experience_years', 0)
    cand_exp = cand_data.get('total_experience_years', 0)
    exp_match_score = 0
    if job_exp > 0:
        exp_match_score = min(cand_exp / job_exp, 1.0) # Cap at 100%
    elif cand_exp >= 0:
        exp_match_score = 1.0 # If 0 years required, any experience is 100%
        
    # --- 3. Project Work (10%) ---
    cand_projects = cand_data.get('projects', [])
    # Score capped at 3 projects (0=0%, 1=33%, 2=66%, 3+=100%)
    project_score = min(len(cand_projects) / 3.0, 1.0)
    
    # --- 4. AI Signal (10%) ---
    ai_signal = cand_data.get('ai_signal', {})
    ai_adjustment = ai_signal.get('score_adjustment', 0) # -10 to +10
    ai_reasoning = ai_signal.get('reasoning', 'No AI signal.')
    # Convert -10..+10 range to 0..1 score
    # -10 -> 0, 0 -> 0.5, +10 -> 1.0
    ai_signal_score = (ai_adjustment + 10) / 20.0
    
    # --- 5. Education (10%) ---
    job_edu_req = education_level_to_int(job_data.get('education_requirement', 'None'))
    cand_edu_level = education_level_to_int(cand_data.get('education_level', 'None'))
    education_score = 1.0 if cand_edu_level >= job_edu_req else 0.0
    
    # --- 6. Certifications (5%) ---
    cand_certs = cand_data.get('certifications', [])
    # Score capped at 2 certifications
    certification_score = min(len(cand_certs) / 2.0, 1.0)
    
    # --- 7. Bonus Skills (5%) ---
    job_bonus_skills = set([s.lower() for s in job_data.get('nice_to_have_skills', [])])
    matched_bonus_skills = job_bonus_skills.intersection(cand_skills)
    bonus_skill_score = 0
    if len(job_bonus_skills) > 0:
        bonus_skill_score = len(matched_bonus_skills) / len(job_bonus_skills)
    
    # --- Final Score Calculation ---
    total_score = (
        (skill_match_score * 0.40) +
        (exp_match_score * 0.20) +
        (project_score * 0.10) +
        (ai_signal_score * 0.10) +
        (education_score * 0.10) +
        (certification_score * 0.05) +
        (bonus_skill_score * 0.05)
    )
    
    return {
        "score": int(total_score * 100),
        "breakdown": {
            "Required Skills": int(skill_match_score * 100),
            "Experience Match": int(exp_match_score * 100),
            "Project Work": int(project_score * 100),
            "AI Signal": int(ai_signal_score * 100),
            "Education Match": int(education_score * 100),
            "Certifications": int(certification_score * 100),
            "Bonus Skills": int(bonus_skill_score * 100)
        },
        "matched_req_skills": list(matched_req_skills),
        "missing_req_skills": list(missing_req_skills),
        "matched_bonus_skills": list(matched_bonus_skills),
        "experience_note": f"Candidate: {cand_exp} yrs, Required: {job_exp} yrs",
        "education_note": f"Candidate: {cand_data.get('education_level', 'N/A')}, Required: {job_data.get('education_requirement', 'N/A')}",
        "ai_signal_reasoning": ai_reasoning,
        "projects": cand_projects,
        "certifications": cand_certs
    }

def calculate_skill_parity(cand_A_skills: List[str], cand_B_skills: List[str]) -> int:
    """Calculates skill similarity between two candidates (Unchanged)."""
    set_a = set([s.lower() for s in cand_A_skills])
    set_b = set([s.lower() for s in cand_B_skills])
    
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    
    if len(union) == 0:
        return 0
        
    parity = len(intersection) / len(union)
    return int(parity * 100)

# --- 5. STREAMLIT APP (UI) ---

def initialize_session_state():
    """Initializes the session state variables."""
    if 'job_description' not in st.session_state:
        st.session_state.job_description = {}
    if 'candidates' not in st.session_state:
        st.session_state.candidates = {}

# --- Async Runner Functions (Unchanged) ---
async def run_job_parser(jd_text):
    st.session_state.job_description = {}
    st.session_state.candidates = {}
    with st.spinner("🤖 Calling AI to analyze job description..."):
        try:
            st.session_state.job_description = await parse_with_ai(jd_text, 'job')
            st.success("Job description parsed successfully!")
        except Exception as e:
            st.error(f"Failed to parse job description: {e}")

async def run_resume_parser(uploaded_files):
    st.session_state.candidates = {}
    progress_bar = st.progress(0, text="Starting resume parsing...")
    
    for i, file in enumerate(uploaded_files):
        filename = file.name
        progress_text = f"Parsing {filename}... ({i+1}/{len(uploaded_files)})"
        progress_bar.progress((i+1) / len(uploaded_files), text=progress_text)

        file_stream = io.BytesIO(file.getvalue())
        text = ""
        if filename.endswith(".pdf"):
            text = read_pdf(file_stream)
        elif filename.endswith(".docx"):
            text = read_docx(file_stream)
        
        if text:
            try:
                ai_data = await parse_with_ai(text, 'resume')
                st.session_state.candidates[filename] = ai_data
            except Exception as e:
                st.error(f"Failed to parse {filename}: {e}")
        else:
            st.warning(f"Could not extract text from {filename}. Skipping.")
    
    progress_bar.empty()
    st.success(f"Successfully parsed {len(st.session_state.candidates)} resumes!")

async def run_ai_chat(question, context):
    with st.spinner("🤖 AI is thinking..."):
        try:
            response = await get_ai_chat_response(question, context)
            st.markdown(response)
        except Exception as e:
            st.error(f"Failed to get AI response: {e}")

# --- Main App Function (NEW UI) ---
def main():
    st.set_page_config(
        page_title="Project Helios - AI Hiring",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    initialize_session_state()
    
    st.title("🚀 Project Helios - Multi-Vector AI Hiring")
    st.caption("Using a 7-factor score to find the *true* best-fit candidates.")

    # Check for API Key
    if not API_KEY:
        st.error("Hold on! You need to add your Google AI API key to the 'API_KEY' variable at the top of the 'helios_app.py' script.", icon="🚨")
        return

    tab_list = [
        "1. Set Job Description",
        "2. Upload Resumes",
        "3. Candidate Ranking (Efficiency Score)",
        "4. Candidate Swapping",
        "5. Ask AI"
    ]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

    # --- TAB 1: JOB DESCRIPTION (Unchanged) ---
    with tab1:
        st.header("Set the Job Description")
        st.markdown("Paste a job description below. The AI will parse it to establish the 'ideal candidate' profile. **This will clear all existing candidates.**")
        
        jd_text = st.text_area("Job Description", height=300, placeholder="Paste your job description here (make sure it mentions experience and education requirements)...")
        
        if st.button("Parse Job Description", type="primary"):
            if jd_text:
                asyncio.run(run_job_parser(jd_text))
            else:
                st.warning("Please paste a job description first.")
        
        if st.session_state.job_description:
            st.subheader("AI-Parsed Job Requirements")
            st.json(st.session_state.job_description)

    # --- TAB 2: UPLOAD RESUMES (Unchanged) ---
    with tab2:
        st.header("Upload Candidate Resumes")
        st.markdown("Upload one or more resumes (PDF or DOCX). The AI will parse each one for skills, experience, projects, certs, and more.")
        
        uploaded_files = st.file_uploader(
            "Choose resumes...",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )
        
        if st.button("Parse Uploaded Resumes", type="primary"):
            if uploaded_files:
                asyncio.run(run_resume_parser(uploaded_files))
            else:
                st.warning("Please upload at least one resume.")
        
        if st.session_state.candidates:
            st.subheader(f"Parsed Candidates ({len(st.session_state.candidates)})")
            for filename, data in st.session_state.candidates.items():
                with st.expander(f"**{filename}** - {data.get('suggested_role', 'N/A')}"):
                    st.json(data)
    
    # --- TAB 3: CANDIDATE RANKING (NEW UI) ---
    with tab3:
        st.header("Candidate Ranking (New 7-Factor Score)")
        
        if not st.session_state.job_description:
            st.warning("Please parse a Job Description in Tab 1 first.")
        elif not st.session_state.candidates:
            st.warning("Please upload and parse resumes in Tab 2 first.")
        else:
            st.markdown("Candidates are ranked by the **Efficiency Score**, a blend of 7 different metrics for a complete, objective analysis.")
            
            job_data = st.session_state.job_description
            candidate_list = st.session_state.candidates
            
            scored_candidates = []
            for filename, cand_data in candidate_list.items():
                try:
                    score_data = calculate_efficiency_score(job_data, cand_data)
                    scored_candidates.append((filename, cand_data, score_data))
                except Exception as e:
                    st.error(f"Error scoring {filename}: {e}")
            
            # Sort by the score
            scored_candidates.sort(key=lambda x: x[2]['score'], reverse=True)
            
            # Display ranked list
            for i, (filename, cand_data, score_data) in enumerate(scored_candidates):
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    st.metric(label=f"Rank #{i+1} - {filename}", value=f"{score_data['score']}%")
                
                with col2:
                    with st.expander(f"**{cand_data.get('suggested_role')}** | {score_data['experience_note']}"):
                        
                        st.subheader("AI Signal")
                        if score_data['ai_signal_reasoning'] != "No AI signal.":
                            st.info(f"**AI Analysis:** {score_data['ai_signal_reasoning']}")
                        else:
                            st.warning("AI signal was not generated for this candidate.")

                        st.subheader("Score Breakdown (7 Factors)")
                        breakdown = score_data['breakdown']
                        cols = st.columns(4)
                        cols[0].metric("Required Skills (40%)", f"{breakdown['Required Skills']}%")
                        cols[1].metric("Experience (20%)", f"{breakdown['Experience Match']}%")
                        cols[2].metric("Project Work (10%)", f"{breakdown['Project Work']}%")
                        cols[3].metric("AI Signal (10%)", f"{breakdown['AI Signal']}%")
                        cols = st.columns(4)
                        cols[0].metric("Education (10%)", f"{breakdown['Education Match']}%")
                        cols[1].metric("Certifications (5%)", f"{breakdown['Certifications']}%")
                        cols[2].metric("Bonus Skills (5%)", f"{breakdown['Bonus Skills']}%")
                        
                        st.subheader("Details")
                        cols = st.columns(2)
                        with cols[0]:
                            st.markdown(f"**Education:** {score_data['education_note']}")
                            st.markdown("**Matched Required Skills**")
                            st.multiselect("Matched", score_data['matched_req_skills'], score_data['matched_req_skills'], disabled=True, key=f"match_req_{filename}")
                            st.markdown("**Matched Bonus Skills**")
                            st.multiselect("Bonus", score_data['matched_bonus_skills'], score_data['matched_bonus_skills'], disabled=True, key=f"match_bonus_{filename}")
                        with cols[1]:
                            st.markdown("**Missing Required Skills**")
                            st.multiselect("Missing", score_data['missing_req_skills'], score_data['missing_req_skills'], disabled=True, key=f"miss_req_{filename}")
                            st.markdown("**Certifications Found**")
                            st.dataframe(score_data['certifications'], use_container_width=True)
                        
                        st.subheader("Projects Found")
                        if score_data['projects']:
                            st.dataframe(score_data['projects'], use_container_width=True)
                        else:
                            st.text("No projects listed.")
                        
                        st.subheader("AI Summary")
                        st.info(cand_data.get('summary', 'No summary available.'))

    # --- TAB 4: CANDIDATE SWAPPING (Unchanged) ---
    with tab4:
        st.header("Candidate Swapping (Skills Parity)")
        st.markdown("If your top candidate drops out, select them here to find the next best fit based on *skill-set similarity*.")
        
        if not st.session_state.candidates:
            st.warning("Please upload and parse resumes in Tab 2 first.")
        else:
            candidate_names = list(st.session_state.candidates.keys())
            withdrawn_candidate = st.selectbox("Select the withdrawn candidate:", candidate_names)
            
            if withdrawn_candidate:
                withdrawn_data = st.session_state.candidates[withdrawn_candidate]
                withdrawn_skills = withdrawn_data.get('skills', [])
                
                parity_scores = []
                for filename, cand_data in st.session_state.candidates.items():
                    if filename == withdrawn_candidate:
                        continue
                    
                    cand_skills = cand_data.get('skills', [])
                    parity = calculate_skill_parity(withdrawn_skills, cand_skills)
                    parity_scores.append((filename, parity, cand_data.get('suggested_role')))
                
                parity_scores.sort(key=lambda x: x[1], reverse=True)
                
                st.subheader(f"Top Skill Matches for {withdrawn_candidate}:")
                
                for i, (filename, parity, role) in enumerate(parity_scores[:3]):
                    st.metric(
                        label=f"**Rank #{i+1}**: {filename} ({role})",
                        value=f"{parity}% Skill Parity"
                    )

    # --- TAB 5: ASK AI (Unchanged) ---
    with tab5:
        st.header("Ask AI About Your Candidates")
        st.markdown("Ask a question in plain English. The AI will use all the new parsed data as context.")
        
        if not st.session_state.job_description or not st.session_state.candidates:
            st.warning("Please parse a Job Description and at least one Resume first.")
        else:
            question = st.text_input("Your question", placeholder="e.g., Who has the most projects? or 'Who has a positive AI Signal?'")
            
            if st.button("Ask AI", type="primary"):
                if question:
                    context_data = {
                        "job_description": st.session_state.job_description,
                        "candidates": st.session_state.candidates
                    }
                    context_str = json.dumps(context_data, indent=2)
                    
                    asyncio.run(run_ai_chat(question, context_str))
                else:
                    st.warning("Please type a question.")

# --- Main entry point ---
if __name__ == "__main__":
    main()