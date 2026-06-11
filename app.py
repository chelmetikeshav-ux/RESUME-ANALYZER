import streamlit as st
import pandas as pd
import pypdf
import spacy
import re
import time
import sys
import subprocess
from skills import extract_skills, calculate_jaccard_similarity

# Set up page configurations
st.set_page_config(
    page_title="QuantumATS // Resume Matcher & Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom SpaCy model loader with caching and auto-download
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Fallback to direct wheel download in case standard download installer is missing
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"], check=True)
            return spacy.load("en_core_web_sm")
        except Exception as e:
            st.error("Could not load spaCy model automatically. Please install it with: pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")
            raise e

# PDF Text Extraction Function
def extract_text_from_pdf(uploaded_file):
    try:
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Action Verb Analyzer
STRONG_ACTION_VERBS = {
    "lead", "direct", "manage", "develop", "architect", "engineer", "implement", "design", "create", 
    "execute", "optimize", "accelerate", "deliver", "modernize", "scale", "streamline", "integrate", 
    "orchestrate", "author", "pioneer", "formulate", "spearhead", "oversee", "catalyze", "build",
    "launch", "coordinate", "supervise", "revamp", "resolve", "diagnose"
}

def analyze_resume_structure(text, nlp_doc):
    suggestions = []
    structure_score = 100
    
    # 1. Contact Information check
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    
    has_email = bool(re.search(email_pattern, text))
    has_phone = bool(re.search(phone_pattern, text))
    
    if not has_email:
        suggestions.append({
            "status": "critical",
            "title": "Email Missing",
            "desc": "No email address was detected. Ensure contact details are clearly formatted at the top."
        })
        structure_score -= 20
    if not has_phone:
        suggestions.append({
            "status": "warning",
            "title": "Phone Number Missing",
            "desc": "No phone number was detected. Recruiters often require phone contact info."
        })
        structure_score -= 10
        
    # 2. Section Headings check
    sections = {
        "Experience": ["experience", "work history", "employment", "professional background"],
        "Education": ["education", "academic", "university", "degree"],
        "Skills": ["skills", "technologies", "technical skills", "expertise"],
        "Projects": ["projects", "personal projects", "portfolio"]
    }
    
    text_lower = text.lower()
    for section_name, keywords in sections.items():
        found = False
        for kw in keywords:
            if re.search(rf"\b{kw}\b", text_lower):
                found = True
                break
        if not found:
            suggestions.append({
                "status": "warning",
                "title": f"'{section_name}' Section Not Found",
                "desc": f"Consider adding a clear heading for '{section_name}' to help ATS parsing indexers."
            })
            structure_score -= 15

    # 3. Action Verbs audit
    verbs_used = set()
    for token in nlp_doc:
        if token.pos_ == "VERB" and token.lemma_ in STRONG_ACTION_VERBS:
            verbs_used.add(token.lemma_)
            
    if len(verbs_used) < 4:
        suggestions.append({
            "status": "info",
            "title": "Enhance Action Verbs",
            "desc": f"Detected only {len(verbs_used)} strong action verbs. Use verbs like 'Spearheaded', 'Orchestrated', 'Optimized' at the beginning of experience points."
        })
        structure_score -= 10
    else:
        suggestions.append({
            "status": "success",
            "title": "Strong Action Verbs",
            "desc": f"Great job! Found {len(verbs_used)} strong action verbs in your experience descriptions."
        })

    # 4. Word Count Check
    word_count = len(text.split())
    if word_count < 250:
        suggestions.append({
            "status": "warning",
            "title": "Resume Too Short",
            "desc": f"Your resume has {word_count} words. ATS systems favor resumes with 400-800 words containing detailed accomplishments."
        })
        structure_score -= 15
    elif word_count > 1200:
        suggestions.append({
            "status": "info",
            "title": "Resume is Long",
            "desc": f"Your resume is {word_count} words. Aim for under 1000 words (1-2 pages) to maintain focus."
        })
        structure_score -= 5
        
    return max(structure_score, 10), suggestions

# Highlight terms in text
def highlight_terms(text, matched_set, missing_set):
    highlighted = text
    # Clean text to prevent regex break
    # Sort terms by length descending to match larger phrases first (e.g., 'machine learning' before 'machine')
    all_terms = list(matched_set) + list(missing_set)
    all_terms.sort(key=len, reverse=True)
    
    for term in all_terms:
        # Create a safe pattern for regex
        escaped_term = re.escape(term)
        if term in matched_set:
            color = "#10b981" # Emerald
            bg = "rgba(16, 185, 129, 0.15)"
            cls = "matched-hl"
        else:
            color = "#f43f5e" # Rose
            bg = "rgba(244, 63, 94, 0.15)"
            cls = "missing-hl"
            
        # Compile match with word boundaries
        if any(char in term.lower() for char in ['+', '#', '.', '-']):
            pattern = rf"(?i)(?:^|[^a-zA-Z0-9+#.-])({escaped_term})(?:$|[^a-zA-Z0-9+#.-])"
        else:
            pattern = rf"(?i)\b({escaped_term})\b"
            
        def repl(match):
            val = match.group(1) if len(match.groups()) > 0 else match.group(0)
            # Find the surrounding non-word characters if any, and keep them outside the span
            matched_str = match.group(0)
            start_offset = matched_str.find(val)
            prefix = matched_str[:start_offset]
            suffix = matched_str[start_offset + len(val):]
            return f'{prefix}<span class="{cls}" style="background:{bg}; border-bottom:2px solid {color}; padding:1px 4px; border-radius:4px; color:{color}; font-weight:500;">{val}</span>{suffix}'
            
        try:
            highlighted = re.sub(pattern, repl, highlighted)
        except Exception:
            pass # Avoid breaking on parsing errors
            
    return highlighted

# Predefined Demo Datasets
DEMO_RESUME = """
Keshav Chelmeti
Software Engineer
Email: keshav.c@email.com | Phone: +1 555-019-2834 | GitHub: github.com/keshavc

SUMMARY
Highly motivated and results-driven Software Engineer with 3+ years of experience building scalable web applications. Expert in Python, React, and database optimization, with a strong foundation in cloud deployment and agile methodologies.

PROFESSIONAL EXPERIENCE
Software Engineer | TechQuantum Solutions (2024 - Present)
- Developed and optimized a customer data platform using Python, FastAPI, and PostgreSQL, reducing query latency by 42%.
- Spearheaded the frontend migration from Angular to React and Redux, improving core web vitals and overall page load speed.
- Configured CI/CD pipelines using GitHub Actions to automate unit testing, linting, and staging deployments on AWS (EC2/S3).
- Implemented REST APIs and integrated Redis caching, boosting platform throughput by 2.5x.
- Mentored 2 junior engineers and actively participated in Agile/Scrum planning sprints.

Junior Developer | ByteCraft Studio (2022 - 2024)
- Built internal tooling and microservices using Node.js, Express, and MongoDB.
- Wrote clean, test-driven code (TDD) using Jest and PyTest.
- Designed responsive layouts using Tailwind CSS and HTML5/CSS3.
- Utilized Git for version control and collaborated with product teams in system design reviews.

EDUCATION
Bachelor of Science in Computer Science
State University (Graduated 2022)

TECHNICAL SKILLS
- Languages: Python, JavaScript, TypeScript, SQL, HTML, CSS
- Frameworks: React, Node.js, Express, FastAPI, Django, Redux
- Cloud & DevOps: AWS (EC2, S3), Git, CI/CD (GitHub Actions), Docker
- Databases: PostgreSQL, MongoDB, Redis, MySQL
- Methodologies: Agile, Scrum, REST APIs, Test-Driven Development (TDD), System Design
"""

DEMO_JD = """
Job Title: Full-Stack Software Engineer
Location: Remote

We are looking for a skilled Full-Stack Software Engineer to join our growing development team. You will be responsible for developing backend services, creating responsive user interfaces, and ensuring cloud deployment pipelines run smoothly.

Key Requirements:
- 3+ years of professional software development experience.
- Deep expertise with Python (FastAPI/Django) and JavaScript/TypeScript (React).
- Strong experience working with PostgreSQL and MongoDB databases.
- Hands-on experience deploying scalable systems on AWS (EC2, Lambda).
- Experience setting up Docker containers and CI/CD pipelines.
- Knowledge of system design, REST APIs, and microservices architecture.
- Understanding of Agile/Scrum workflows and Test-Driven Development (TDD).
- Familiarity with Kubernetes, TypeScript, and Redis caching is a big plus.
"""

# Theme Styling Injections
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Main body overrides */
    .stApp {
        background-color: #090d16;
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title and Subtitle */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Header Area */
    .app-header {
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .logo-container {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .logo-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, #00e5ff, #9d5cff);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.25);
    }
    .logo-brand {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .logo-brand span {
        background: linear-gradient(135deg, #00e5ff, #9d5cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Cards and Glassmorphism */
    .glass-card {
        background: rgba(13, 20, 35, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(20px);
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Step Indicators styling */
    .pipeline-container {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
    }
    .step-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        transition: all 0.3s ease;
    }
    .step-item.pending {
        opacity: 0.4;
    }
    .step-item.active {
        opacity: 1;
        background: rgba(0, 229, 255, 0.05);
        border-color: rgba(0, 229, 255, 0.25);
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.08);
    }
    .step-item.done {
        opacity: 0.85;
        background: rgba(16, 185, 129, 0.05);
        border-color: rgba(16, 185, 129, 0.2);
    }
    .step-dot {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
    }
    .step-dot.pending { background: #475569; }
    .step-dot.active { background: #00e5ff; color: #090d16; }
    .step-dot.done { background: #10b981; color: #ffffff; }
    
    /* Circular progress gauge */
    .gauge-container {
        position: relative;
        width: 170px;
        height: 170px;
        margin: 1.5rem auto;
    }
    .gauge-svg {
        transform: rotate(-90deg);
        width: 100%;
        height: 100%;
    }
    .gauge-bg {
        fill: none;
        stroke: rgba(255, 255, 255, 0.04);
        stroke-width: 8;
    }
    .gauge-fill {
        fill: none;
        stroke: #00e5ff;
        stroke-width: 8;
        stroke-linecap: round;
        stroke-dasharray: 251.2;
        transition: stroke-dashoffset 1s ease-out;
    }
    .gauge-text {
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .gauge-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
    }
    .gauge-lbl {
        font-size: 0.75rem;
        color: #94a3b8;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: -2px;
    }
    
    /* Skill badges */
    .skill-tag {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        padding: 0.3rem 0.65rem;
        border-radius: 6px;
        margin: 0.25rem;
        border: 1px solid transparent;
        font-weight: 500;
    }
    .tag-matched {
        background: rgba(16, 185, 129, 0.08);
        border-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
    }
    .tag-missing {
        background: rgba(244, 63, 94, 0.08);
        border-color: rgba(244, 63, 94, 0.2);
        color: #f43f5e;
    }
    .tag-extra {
        background: rgba(0, 229, 255, 0.08);
        border-color: rgba(0, 229, 255, 0.2);
        color: #00e5ff;
    }
    
    /* Suggestions card items */
    .suggestion-card {
        display: flex;
        gap: 1rem;
        padding: 0.85rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background: rgba(255, 255, 255, 0.01);
        margin-bottom: 0.75rem;
    }
    .sug-status {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.75rem;
        flex-shrink: 0;
    }
    .status-c { background: rgba(244, 63, 94, 0.15); color: #f43f5e; }
    .status-w { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .status-i { background: rgba(0, 229, 255, 0.15); color: #00e5ff; }
    .status-s { background: rgba(16, 185, 129, 0.15); color: #10b981; }
    
    /* Text viewer */
    .text-viewer {
        background: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        height: 380px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
    
    /* Streamlit overrides to make it feel custom */
    div[data-testid="stBlock"] {
        padding: 0 !important;
    }
    .stTextArea textarea {
        background-color: rgba(13, 20, 35, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
    }
    .stTextArea textarea:focus {
        border-color: #00e5ff !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.15) !important;
    }
    
    /* Hide default streamlit margins */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<header class="app-header">
    <div class="logo-container">
        <div class="logo-icon">⚡</div>
        <div class="logo-brand">Quantum<span>ATS</span> // Alignment Engine</div>
    </div>
</header>
""", unsafe_allow_html=True)

# Initialize Session States
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "resume_filename" not in st.session_state:
    st.session_state.resume_filename = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

# Demo Data Trigger
if st.button("✨ Load Demo Data"):
    st.session_state.resume_text = DEMO_RESUME.strip()
    st.session_state.resume_filename = "demo_developer_resume.pdf"
    st.session_state.job_desc = DEMO_JD.strip()
    # Rerun to update values
    st.rerun()

# ----------------- INPUT PHASE -----------------
st.markdown("<h2 style='text-align: center; margin-bottom: 0.5rem;'>Optimize Your Resume for ATS Algorithms</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size:1.05rem; max-width:800px; margin: 0 auto 2.5rem;'>Upload your PDF resume and paste the target job description to analyze skills gaps, match percentages, and receive direct suggestions for improvement.</p>", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="glass-card" style="height: 480px; overflow: hidden; display:flex; flex-direction:column;">', unsafe_allow_html=True)
    st.markdown("<h3>📋 Job Description Requirements</h3>", unsafe_allow_html=True)
    
    # Textarea for Job Description
    jd_input = st.text_area(
        "Paste the Job Description here:",
        value=st.session_state.job_desc,
        placeholder="Requirements, responsibilities, skills, tools required...",
        height=320,
        label_visibility="collapsed"
    )
    st.session_state.job_desc = jd_input
    
    # Word count
    jd_word_count = len(jd_input.split())
    st.markdown(f"<span style='color: #64748b; font-size: 0.8rem;'>Word Count: {jd_word_count} words</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card" style="height: 480px; display:flex; flex-direction:column; justify-content:space-between;">', unsafe_allow_html=True)
    st.markdown("<h3>📄 Resume PDF Upload</h3>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload your resume in PDF format:",
        type=["pdf"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        if st.session_state.resume_filename != uploaded_file.name:
            # New file uploaded
            st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
            st.session_state.resume_filename = uploaded_file.name
            st.rerun()
            
    # Display current file status
    if st.session_state.resume_filename:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.05); border: 1px dashed rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 1.5rem; text-align: center; margin-top: 1rem;">
            <span style="color:#10b981; font-size:2.5rem; display:block; margin-bottom:0.5rem;">✓</span>
            <span style="font-weight: 600; display:block; color:#f8fafc;">{st.session_state.resume_filename}</span>
            <span style="font-family:'JetBrains Mono'; font-size: 0.75rem; color:#64748b;">({len(st.session_state.resume_text.split())} words extracted)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.03); border: 1px dashed rgba(245, 158, 11, 0.2); border-radius: 10px; padding: 2.5rem; text-align: center; margin-top: 1rem; color: #94a3b8;">
            <span style="font-size:2.2rem; display:block; margin-bottom:0.5rem; filter: grayscale(1);">⚡</span>
            <span>Awaiting PDF Upload or Demo Data</span>
            <span style="display:block; font-size:0.75rem; color:#64748b; margin-top:0.4rem;">PDF files are processed entirely client-side.</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# Submit button container
st.markdown("<br>", unsafe_allow_html=True)
col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])

analyze_clicked = False
with col_btn_c:
    is_ready = bool(st.session_state.resume_text.strip()) and bool(st.session_state.job_desc.strip())
    analyze_clicked = st.button(
        "Analyze Compatibility", 
        disabled=not is_ready, 
        use_container_width=True,
        type="primary"
    )

# ----------------- PIPELINE AND ANALYSIS PHASE -----------------
if analyze_clicked or ("analysis_done" in st.session_state and st.session_state.analysis_done):
    st.session_state.analysis_done = True
    
    # Load NLP Model
    nlp = load_nlp_model()
    
    # Pipeline status steps
    steps = [
        ("Resume PDF", "Loading document structure..."),
        ("Extract Text", f"Reading document character maps ({len(st.session_state.resume_text)} chars)"),
        ("Skill Extraction", "Analyzing technical skills taxonomy..."),
        ("Compare with Job Description", "Calculating overlapping keyword weights..."),
        ("Calculate Match %", "Formulating scoring matrices..."),
        ("Generate Suggestions", "Running heuristic suggestions audit..."),
        ("Display Results", "Finalizing visualizations...")
    ]
    
    # If the analysis was just clicked, show the visual progress pipeline
    if analyze_clicked:
        st.markdown("<h3 style='margin-top:2rem;'>Executing Pipeline Engine...</h3>", unsafe_allow_html=True)
        progress_placeholders = [st.empty() for _ in range(len(steps))]
        
        # Initialize steps view
        for idx, (label, desc) in enumerate(steps):
            progress_placeholders[idx].markdown(f"""
            <div class="step-item pending">
                <div class="step-dot pending">○</div>
                <div style="flex:1;">
                    <span style="font-weight:600; font-size:0.95rem; color:#64748b;">{label}</span>
                </div>
                <span style="font-size:0.8rem; color:#475569;">Awaiting preceding steps</span>
            </div>
            """, unsafe_allow_html=True)
            
        time.sleep(0.3)
        
        # Execute stages sequentially with animation
        for idx in range(len(steps)):
            # Mark active
            progress_placeholders[idx].markdown(f"""
            <div class="step-item active">
                <div class="step-dot active">⚡</div>
                <div style="flex:1;">
                    <span style="font-weight:600; font-size:0.95rem; color:#00e5ff;">{steps[idx][0]}</span>
                    <span style="font-size:0.8rem; color:#94a3b8; display:block;">{steps[idx][1]}</span>
                </div>
                <span style="font-size:0.8rem; color:#00e5ff;">Processing...</span>
            </div>
            """, unsafe_allow_html=True)
            
            time.sleep(0.5) # Simulate workload
            
            # Mark done
            progress_placeholders[idx].markdown(f"""
            <div class="step-item done">
                <div class="step-dot done">✓</div>
                <div style="flex:1;">
                    <span style="font-weight:600; font-size:0.95rem; color:#ffffff;">{steps[idx][0]}</span>
                    <span style="font-size:0.8rem; color:#64748b; display:block;">{steps[idx][1]}</span>
                </div>
                <span style="font-size:0.8rem; color:#10b981;">Completed</span>
            </div>
            """, unsafe_allow_html=True)
            
        time.sleep(0.2)
        
        # Clear progress indicators
        for ph in progress_placeholders:
            ph.empty()
            
    # --- PROCESS PIPELINE LOGIC ---
    resume_text = st.session_state.resume_text
    job_desc = st.session_state.job_desc
    
    # 1. Skill Extraction
    resume_cats, resume_skills = extract_skills(resume_text)
    jd_cats, jd_skills = extract_skills(job_desc)
    
    # Compute intersections
    matched_skills = resume_skills.intersection(jd_skills)
    missing_skills = jd_skills - resume_skills
    extra_skills = resume_skills - jd_skills
    
    # 2. Similarity
    jaccard_score = calculate_jaccard_similarity(resume_text, job_desc)
    
    # 3. Structure & SpaCy analysis
    doc = nlp(resume_text)
    structure_score, formatting_suggs = analyze_resume_structure(resume_text, doc)
    
    # 4. Final Match Calculation
    # - Skill match ratio: matched / requested in JD
    skill_ratio = len(matched_skills) / len(jd_skills) if jd_skills else 1.0
    
    # Overall formula
    # 60% skills coverage, 30% word-similarity matching, 10% structural health
    overall_score = (skill_ratio * 60) + (jaccard_score * 300) + (structure_score * 0.1)
    overall_score = min(max(int(overall_score), 10), 100)
    
    # Determine rating
    if overall_score >= 80:
        rating = "Excellent"
        rating_color = "#10b981"
    elif overall_score >= 60:
        rating = "Good Match"
        rating_color = "#f59e0b"
    else:
        rating = "Low Match"
        rating_color = "#f43f5e"
        
    # --- RENDER RESULTS DASHBOARD ---
    st.markdown("<h2 style='margin-top:2.5rem; margin-bottom:1.5rem;'>⚡ Analysis Dashboard</h2>", unsafe_allow_html=True)
    
    dash_col_left, dash_col_mid, dash_col_right = st.columns([1, 1.2, 1.2])
    
    with dash_col_left:
        st.markdown('<div class="glass-card" style="height: 520px; display:flex; flex-direction:column; justify-content:space-between; align-items:center;">', unsafe_allow_html=True)
        st.markdown("<h3 style='margin-bottom:0; width:100%; text-align:left;'>Overall Compatibility</h3>", unsafe_allow_html=True)
        
        # Custom SVG Progress Dial
        stroke_dashoffset = 251.2 - (251.2 * (overall_score / 100))
        st.markdown(f"""
        <div class="gauge-container">
            <svg class="gauge-svg" viewBox="0 0 100 100">
                <circle class="gauge-bg" cx="50" cy="50" r="40"></circle>
                <circle class="gauge-fill" cx="50" cy="50" r="40" 
                        style="stroke-dashoffset: {stroke_dashoffset}; stroke: {rating_color}; filter: drop-shadow(0 0 6px {rating_color}80);"></circle>
            </svg>
            <div class="gauge-text">
                <span class="gauge-val">{overall_score}%</span>
                <span class="gauge-lbl" style="color:{rating_color};">{rating}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sub-score Progress bars
        skills_percent = int(skill_ratio * 100)
        keywords_percent = int(jaccard_score * 300) # Jaccard ranges up to ~0.33 usually for similar profiles, scaling to 100%
        keywords_percent = min(keywords_percent, 100)
        
        st.markdown(f"""
        <div style="width:100%;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:0.25rem;">
                <span style="color:#94a3b8;">Skills Match</span>
                <span style="font-family:'JetBrains Mono'; font-weight:600;">{skills_percent}%</span>
            </div>
            <div style="height:6px; background:rgba(255,255,255,0.04); border-radius:10px; margin-bottom:1rem; overflow:hidden;">
                <div style="width:{skills_percent}%; height:100%; background:linear-gradient(90deg, #00e5ff, #9d5cff); border-radius:10px;"></div>
            </div>
            
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:0.25rem;">
                <span style="color:#94a3b8;">Keyword Density</span>
                <span style="font-family:'JetBrains Mono'; font-weight:600;">{keywords_percent}%</span>
            </div>
            <div style="height:6px; background:rgba(255,255,255,0.04); border-radius:10px; margin-bottom:1rem; overflow:hidden;">
                <div style="width:{keywords_percent}%; height:100%; background:linear-gradient(90deg, #00e5ff, #9d5cff); border-radius:10px;"></div>
            </div>
            
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:0.25rem;">
                <span style="color:#94a3b8;">Structural Health</span>
                <span style="font-family:'JetBrains Mono'; font-weight:600;">{structure_score}%</span>
            </div>
            <div style="height:6px; background:rgba(255,255,255,0.04); border-radius:10px; overflow:hidden;">
                <div style="width:{structure_score}%; height:100%; background:linear-gradient(90deg, #00e5ff, #9d5cff); border-radius:10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with dash_col_mid:
        st.markdown('<div class="glass-card" style="height: 520px; display:flex; flex-direction:column; overflow:hidden;">', unsafe_allow_html=True)
        st.markdown("<h3>🏷️ Skill Coverage Audit</h3>", unsafe_allow_html=True)
        
        # Tabs for filtering skills
        skill_tab_all, skill_tab_match, skill_tab_miss, skill_tab_extra = st.tabs([
            f"All ({len(jd_skills.union(resume_skills))})", 
            f"Matched ({len(matched_skills)})", 
            f"Missing ({len(missing_skills)})", 
            f"Extra ({len(extra_skills)})"
        ])
        
        # Compile tags HTML helper
        def get_tags_html(skills, tag_class):
            if not skills:
                return "<p style='color:#64748b; font-size:0.85rem; padding: 0.5rem;'>No skills in this category.</p>"
            tags = ""
            for s in sorted(skills):
                tags += f'<span class="skill-tag {tag_class}">{s}</span>'
            return f'<div style="margin-top:0.5rem;">{tags}</div>'
            
        with skill_tab_all:
            st.markdown('<div style="overflow-y:auto; height:340px; padding-right:5px;">', unsafe_allow_html=True)
            st.markdown("<h5>Matched Job Requirements:</h5>", unsafe_allow_html=True)
            st.markdown(get_tags_html(matched_skills, "tag-matched"), unsafe_allow_html=True)
            st.markdown("<h5 style='margin-top:1.25rem;'>Missing Required Skills:</h5>", unsafe_allow_html=True)
            st.markdown(get_tags_html(missing_skills, "tag-missing"), unsafe_allow_html=True)
            st.markdown("<h5 style='margin-top:1.25rem;'>Additional Resume Skills:</h5>", unsafe_allow_html=True)
            st.markdown(get_tags_html(extra_skills, "tag-extra"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with skill_tab_match:
            st.markdown('<div style="overflow-y:auto; height:340px;">', unsafe_allow_html=True)
            st.markdown(get_tags_html(matched_skills, "tag-matched"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with skill_tab_miss:
            st.markdown('<div style="overflow-y:auto; height:340px;">', unsafe_allow_html=True)
            st.markdown(get_tags_html(missing_skills, "tag-missing"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with skill_tab_extra:
            st.markdown('<div style="overflow-y:auto; height:340px;">', unsafe_allow_html=True)
            st.markdown(get_tags_html(extra_skills, "tag-extra"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with dash_col_right:
        st.markdown('<div class="glass-card" style="height: 520px; display:flex; flex-direction:column; overflow:hidden;">', unsafe_allow_html=True)
        st.markdown("<h3>🎯 Optimization Checklist</h3>", unsafe_allow_html=True)
        
        # Combine structural suggestions with dynamic skills recommendations
        all_suggs = []
        
        # Critical skills recommendations
        if missing_skills:
            top_miss = list(missing_skills)[:4]
            miss_list_str = ", ".join([f"**{m}**" for m in top_miss])
            if len(missing_skills) > 4:
                miss_list_str += f", and {len(missing_skills)-4} other skills"
            all_suggs.append({
                "status": "critical",
                "title": f"Integrate Missing Core Skills",
                "desc": f"Your resume lacks these requested qualifications: {miss_list_str}. Weave them into your projects or experience items."
            })
            
        # Add formatting / structural suggestions
        all_suggs.extend(formatting_suggs)
        
        # Render Suggestions
        st.markdown('<div style="overflow-y:auto; height:410px; padding-right:5px;">', unsafe_allow_html=True)
        for sug in all_suggs:
            status_cls = {
                "critical": "status-c",
                "warning": "status-w",
                "info": "status-i",
                "success": "status-s"
            }.get(sug["status"], "status-i")
            
            icon = {"critical": "✗", "warning": "⚠", "info": "ℹ", "success": "✓"}.get(sug["status"], "•")
            
            st.markdown(f"""
            <div class="suggestion-card">
                <div class="sug-status {status_cls}">{icon}</div>
                <div>
                    <h5 style="margin:0; font-weight:600; color:#ffffff;">{sug["title"]}</h5>
                    <p style="margin:0.25rem 0 0; font-size:0.8rem; color:#94a3b8; line-height:1.4;">{sug["desc"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- HEATMAP / DETAILS VIEW ---
    st.markdown("<h3 style='margin-top:1.5rem;'>🔍 Interactive Text Match Audit</h3>", unsafe_allow_html=True)
    
    col_text_resume, col_text_jd = st.columns(2)
    
    with col_text_resume:
        st.markdown('<div class="glass-card" style="margin-bottom: 0.5rem;">', unsafe_allow_html=True)
        st.markdown("<h4>Resume Content Analysis</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.75rem; color:#64748b; margin-top:-0.5rem;'>Highlights matching keywords (green) and general skills found in the text.</p>", unsafe_allow_html=True)
        
        hl_resume = highlight_terms(resume_text, matched_skills, set())
        st.markdown(f'<div class="text-viewer">{hl_resume}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_text_jd:
        st.markdown('<div class="glass-card" style="margin-bottom: 0.5rem;">', unsafe_allow_html=True)
        st.markdown("<h4>Job Description Audit</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.75rem; color:#64748b; margin-top:-0.5rem;'>Highlights where your skills match (green) and gaps where requirements are missing (red).</p>", unsafe_allow_html=True)
        
        hl_jd = highlight_terms(job_desc, matched_skills, missing_skills)
        st.markdown(f'<div class="text-viewer">{hl_jd}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("""
    <div style="text-align:center; padding:1.5rem; margin-top:2rem; font-size:0.75rem; color:#64748b; border-top:1px solid rgba(255,255,255,0.05);">
        QuantumATS • All analysis processing is handled locally within memory buffers. No document files are saved.
    </div>
    """, unsafe_allow_html=True)
