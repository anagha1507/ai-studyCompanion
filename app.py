"""
AI Study Companion - Complete Implementation
Features: PDF/YouTube Summarizer, Quiz Generator, Context-Aware Chatbot
Uses yt-dlp for YouTube (no caption API needed)
"""

import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import json
import os
import re
from dotenv import load_dotenv
from datetime import datetime
import time
from gtts import gTTS
from sklearn.cluster import KMeans
import base64
from io import BytesIO
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION & INITIALIZATION
# ============================================

# Page configuration
st.set_page_config(
    page_title="AI Study Companion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better readability
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        cursor: pointer;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Make all text dark and readable */
    p, li, div, span, label {
        color: #1a1a1a !important;
    }
    
    h1, h2, h3, h4 {
        color: #2c3e50 !important;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .user-message strong {
        color: #0d47a1 !important;
        font-size: 1.1em;
    }
    
    .bot-message {
        background-color: #ffffff;
        border: 1px solid #d4d4d4;
        border-left: 4px solid #764ba2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .bot-message, 
    .bot-message p, 
    .bot-message li, 
    .bot-message strong,
    .bot-message span {
        color: #1a1a1a !important;
    }
    
    .bot-message strong {
        color: #4a148c !important;
        font-size: 1.1em;
    }
    
    /* Quiz card */
    .quiz-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .quiz-card p, .quiz-card label {
        color: #1a1a1a !important;
    }
    
    /* Summary container */
    .summary-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .summary-container, 
    .summary-container p, 
    .summary-container li {
        color: #1a1a1a !important;
        line-height: 1.8;
    }
    
    .summary-container h1, 
    .summary-container h2, 
    .summary-container h3 {
        color: #2c3e50 !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    /* Messages */
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
    }
    
    .stError {
        background-color: #f8d7da !important;
        color: #721c24 !important;
    }
    
    .stInfo {
        background-color: #d1ecf1 !important;
        color: #0c5460 !important;
    }
    
    .stWarning {
        background-color: #fff3cd !important;
        color: #856404 !important;
    }
    
    /* Chat input */
    .stTextInput input {
        border: 2px solid #cccccc;
        border-radius: 10px;
        padding: 0.75rem;
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #2c3e50 !important;
        font-weight: 500 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

# Session state initialization
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = []
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'quiz_history' not in st.session_state:
    st.session_state.quiz_history = []
if 'xp' not in st.session_state:
    st.session_state.xp = 0
if 'streak' not in st.session_state:
    st.session_state.streak = 0
if 'badges' not in st.session_state:
    st.session_state.badges = []

# ============================================
# TEXT EXTRACTION FUNCTIONS
# ============================================


def extract_pdf_text(uploaded_file):
    """Extract text from uploaded PDF"""
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            return None, "Could not extract text from PDF. The file may be scanned images."
        
        return text, None
    except Exception as e:
        return None, f"Error extracting PDF text: {str(e)}"

# ============================================
# SUMMARY GENERATION
# ============================================

def generate_summary(text):
    """Generate comprehensive summary using Gemini"""
    try:
        prompt = f"""
        You are an expert educator creating a study summary.
        
        Please provide a comprehensive yet concise summary of the following text.
        Structure your summary with:
        1. Main Topic (1 sentence)
        2. Key Concepts (bullet points)
        3. Important Details (paragraph)
        4. Key Takeaways (3-5 bullet points)
        
        Make it clear and study-friendly.
        
        Text to summarize:
        {text[:50000]}
        """
        
        response = model.generate_content(prompt)
        return response.text, None
    except Exception as e:
        return None, f"Error generating summary: {str(e)}"

# ============================================
# QUIZ GENERATION
# ============================================

def generate_quiz(text, num_questions=5):
    """Generate multiple choice quiz using Gemini with randomization"""
    try:
        import random
        
        # Randomize parameters for variety
        focus_areas = [
            "key concepts and definitions",
            "important details and facts",
            "relationships between concepts",
            "practical applications",
            "compare and contrast different ideas",
            "cause and effect relationships",
            "examples and case studies mentioned",
            "the most challenging concepts"
        ]
        
        question_styles = [
            "direct factual questions",
            "scenario-based questions",
            "fill-in-the-blank style questions",
            "which of the following is true/false",
            "identify the correct statement"
        ]
        
        # Randomly select focus and style
        selected_focus = random.choice(focus_areas)
        selected_style = random.choice(question_styles)
        
        # Randomize number of questions slightly
        num_questions = random.randint(4, 6)
        
        prompt = f"""
        You are an expert educator creating a multiple-choice quiz.
        
        Based on the following text, generate {num_questions} multiple-choice questions
        that test understanding of the material.
        
        IMPORTANT RANDOMIZATION INSTRUCTIONS:
        - Focus especially on: {selected_focus}
        - Use this question style: {selected_style}
        - Make each quiz unique by focusing on different parts of the text
        - Vary the difficulty level
        
        Return ONLY a valid JSON array in this exact format:
        [
            {{
                "question": "Question text here?",
                "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                "correct": "A) Option 1",
                "explanation": "Brief explanation of why this is correct"
            }}
        ]
        
        Make questions clear and challenging but fair.
        Do not include any text outside the JSON array.
        
        Text:
        {text[:30000]}
        """
        
        response = model.generate_content(prompt)
        
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.startswith("```"):
            json_text = json_text[3:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        
        quiz_data = json.loads(json_text.strip())
        return quiz_data, None
    except Exception as e:
        return None, f"Error generating quiz: {str(e)}"
    
# ============================================
# CHATBOT FUNCTIONS
# ============================================

def search_relevant_text(query, full_text, top_k=3):
    """TF-IDF based semantic search for chatbot context"""
    try:
        # Split text into sentences
        sentences = re.split(r'[.!?]+', full_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return [full_text[:1000]]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        
        # Fit on all sentences + query
        all_texts = sentences + [query]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Get query vector (last row)
        query_vector = tfidf_matrix[-1]
        
        # Get sentence vectors (all except last)
        sentence_vectors = tfidf_matrix[:-1]
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, sentence_vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Get top sentences with context
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                # Get surrounding context (2 sentences before and after)
                start = max(0, idx - 2)
                end = min(len(sentences), idx + 3)
                context = " ".join(sentences[start:end])
                results.append(context)
        
        return results if results else [full_text[:1000]]
        
    except Exception as e:
        print(f"Search error: {e}")
        return [full_text[:1000]]

def chat_with_context(full_text, user_question, chat_history):
    """Chat with context from the study material"""
    try:
        relevant_chunks = search_relevant_text(user_question, full_text)
        context = "\n\n".join(relevant_chunks)
        
        history_text = ""
        for msg in chat_history[-6:]:
            history_text += f"{msg['role']}: {msg['content']}\n"
        
        prompt = f"""
        You are StudyBuddy, a helpful and patient AI tutor.
        
        IMPORTANT RULES:
        1. Answer ONLY using information from the provided context below.
        2. If the answer isn't in the context, say: "I don't see that specific information in your study material. Would you like me to explain a related concept that's mentioned?"
        3. Be encouraging and use simple, clear language.
        
        CONVERSATION HISTORY:
        {history_text}
        
        CONTEXT FROM STUDY MATERIAL:
        {context}
        
        USER QUESTION: {user_question}
        
        YOUR RESPONSE (as StudyBuddy):
        """
        
        response = model.generate_content(prompt)
        return response.text, None
    except Exception as e:
        return None, f"Error in chat: {str(e)}"
    
# ============================================
# NEW FEATURE 1: AUDIO SUMMARY
# ============================================

def generate_audio(text):
    """Generate audio from text using gTTS"""
    try:
        # Take first 2000 characters for audio (to keep it brief)
        audio_text = text[:2000]
        
        tts = gTTS(text=audio_text, lang='en', slow=False)
        
        # Save to BytesIO object
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        return audio_bytes, None
    except Exception as e:
        return None, f"Error generating audio: {str(e)}"

# ============================================
# NEW FEATURE 2: FLASHCARD EXPORT
# ============================================

def generate_flashcards(quiz_data):
    """Generate Anki-compatible flashcards from quiz data"""
    cards = []
    for q in quiz_data:
        # Format: question;answer (Anki format)
        question = q['question'].replace(';', ',')
        answer = q['correct'].replace(';', ',') + " - " + q['explanation'].replace(';', ',')
        cards.append(f"{question};{answer}")
    
    return "\n".join(cards)

# ============================================
# NEW FEATURE 3: BADGES & ACHIEVEMENTS
# ============================================

def check_badges():
    """Check and award badges based on user activity"""
    
    # First quiz completed
    if len(st.session_state.quiz_history) >= 1 and "First Quiz" not in st.session_state.badges:
        st.session_state.badges.append("First Quiz")
        st.balloons()
        st.success("🏆 Badge Earned: First Quiz Completed!")
    
    # Score of 80% or higher
    if st.session_state.quiz_history and st.session_state.quiz_history[-1]['score'] >= 80:
        if "High Scorer" not in st.session_state.badges:
            st.session_state.badges.append("High Scorer")
            st.balloons()
            st.success("🏆 Badge Earned: High Scorer (80%+)!")
    
    # Perfect score
    if st.session_state.quiz_history and st.session_state.quiz_history[-1]['score'] == 100:
        if "Perfect Score" not in st.session_state.badges:
            st.session_state.badges.append("Perfect Score")
            st.balloons()
            st.success("🏆 Badge Earned: Perfect Score!")
    
    # Completed 5 quizzes
    if len(st.session_state.quiz_history) >= 5 and "Quiz Master" not in st.session_state.badges:
        st.session_state.badges.append("Quiz Master")
        st.balloons()
        st.success("🏆 Badge Earned: Quiz Master (5 Quizzes)!")

def calculate_xp(score, total_questions):
    """Calculate XP based on quiz performance"""
    percentage = (score / total_questions) * 100
    
    if percentage == 100:
        return 100  # Perfect score bonus
    elif percentage >= 80:
        return 75
    elif percentage >= 60:
        return 50
    elif percentage >= 40:
        return 25
    else:
        return 10  # Participation XP

# ============================================
# UI COMPONENTS
# ============================================

def render_sidebar():
    """Render sidebar with upload options"""
    with st.sidebar:
        st.markdown("## 📤 Upload Study Material")
        
        upload_option = st.radio(
            "Choose input method:",
            ["📄 Upload PDF", "📝 Paste Text"]
        )
        
        if upload_option == "📄 Upload PDF":
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            if uploaded_file and st.button("Process PDF", use_container_width=True):
                with st.spinner("Extracting text from PDF..."):
                    text, error = extract_pdf_text(uploaded_file)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.extracted_text = text
                        st.success(f"✅ Extracted {len(text)} characters!")
                        process_all_features(text)
                        
        else:  # Paste Text
            pasted_text = st.text_area("Paste your notes here:", height=200)
            if pasted_text and st.button("Process Text", use_container_width=True):
                st.session_state.extracted_text = pasted_text
                st.success(f"✅ Received {len(pasted_text)} characters!")
                process_all_features(pasted_text)
        
        st.markdown("---")
        st.markdown("### 📊 Material Stats")
        if st.session_state.extracted_text:
            word_count = len(st.session_state.extracted_text.split())
            st.metric("Words", f"{word_count:,}")
            st.metric("Characters", f"{len(st.session_state.extracted_text):,}")
            est_read_time = max(1, word_count // 200)
            st.metric("Est. Reading Time", f"{est_read_time} min")

def train_educational_classifier():
    """Train a simple classifier to detect educational content"""
    
    # Training data
    educational_texts = [
        "The process of photosynthesis converts light energy into chemical energy in plants",
        "Newton's laws of motion describe the relationship between force and acceleration",
        "The quadratic formula is used to solve quadratic equations in algebra",
        "Machine learning algorithms can be categorized as supervised and unsupervised learning",
        "The French Revolution began in 1789 and ended in 1799",
        "DNA replication is the process of copying genetic information",
        "Supply and demand determine market equilibrium in economics",
    ]
    
    non_educational_texts = [
        "OMG! Did you see what happened at the party last night?",
        "Buy now and get 50% off on all products! Limited time offer",
        "I had pizza for dinner and it was delicious",
        "The movie was so boring I fell asleep halfway through",
        "My cat scratched the furniture again today",
    ]
    
    # Create labels
    texts = educational_texts + non_educational_texts
    labels = [1] * len(educational_texts) + [0] * len(non_educational_texts)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    # Train Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X, labels)
    
    return vectorizer, classifier

def is_educational_content(text):
    """Check if content is educational using ML classifier + Gemini"""
    try:
        # Method 1: Local ML Classifier (fast, no API call)
        try:
            vectorizer, classifier = train_educational_classifier()
            text_vector = vectorizer.transform([text[:500]])
            prediction = classifier.predict(text_vector)[0]
            confidence = classifier.predict_proba(text_vector)[0]
            
            # If high confidence, use local prediction
            if max(confidence) > 0.8:
                return bool(prediction)
        except:
            pass
        
        # Method 2: Gemini API (fallback for accuracy)
        sample = text[:500]
        prompt = f"""
        Is this text educational/study-related content?
        Reply ONLY "YES" or "NO".
        Text: {sample}
        """
        response = model.generate_content(prompt)
        return "YES" in response.text.strip().upper()
        
    except Exception as e:
        print(f"Validation error: {e}")
        return True  # Allow content on error
    
def get_adapted_quiz_difficulty():
    """Determine quiz difficulty based on learner's performance history"""
    
    if not st.session_state.quiz_history:
        return "medium"
    
    # Get average score
    avg_score = sum(q['score'] for q in st.session_state.quiz_history) / len(st.session_state.quiz_history)
    
    if avg_score >= 85:
        return "hard"
    elif avg_score >= 65:
        return "medium"
    else:
        return "easy"

def generate_adaptive_quiz(text, num_questions=5):
    """Generate quiz with difficulty adapted to learner"""
    try:
        import random
        
        difficulty = get_adapted_quiz_difficulty()
        
        difficulty_instructions = {
            "easy": "Create basic recall questions about fundamental concepts. Use simple language.",
            "medium": "Create questions that test understanding and application. Use moderate complexity.",
            "hard": "Create challenging questions requiring analysis and synthesis. Test deeper understanding."
        }
        
        prompt = f"""
        You are an expert educator. Create a {difficulty.upper()} difficulty quiz.
        
        {difficulty_instructions[difficulty]}
        
        Generate {num_questions} multiple-choice questions based on the text.
        
        Return ONLY valid JSON array:
        [
            {{
                "question": "Question text?",
                "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                "correct": "A) Option 1",
                "explanation": "Why this is correct"
            }}
        ]
        
        Text: {text[:30000]}
        """
        
        response = model.generate_content(prompt)
        
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        
        return json.loads(json_text.strip()), difficulty
        
    except Exception as e:
        return None, f"Error: {str(e)}"
    
def cluster_topics(text, n_clusters=3):
    """Cluster sentences into topic groups"""
    try:
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        
        if len(sentences) < n_clusters * 3:
            return None
        
        # Vectorize sentences
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(sentences)
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Group sentences by cluster
        topic_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in topic_groups:
                topic_groups[cluster_id] = []
            topic_groups[cluster_id].append(sentences[i])
        
        # Get top terms for each cluster
        feature_names = vectorizer.get_feature_names_out()
        cluster_keywords = {}
        
        for cluster_id in range(n_clusters):
            centroid = kmeans.cluster_centers_[cluster_id]
            top_indices = centroid.argsort()[-5:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            cluster_keywords[cluster_id] = keywords
        
        return topic_groups, cluster_keywords
        
    except Exception as e:
        print(f"Clustering error: {e}")
        return None
    
def process_all_features(text):
    """Process all features with content validation"""
    
    if not text or len(text.strip()) < 100:
        st.error("❌ Text is too short. Please provide more content.")
        return
    
    # ============================================
    # CONTENT VALIDATION
    # ============================================
    with st.spinner("🔍 Checking if content is study-related..."):
        is_educational = is_educational_content(text)
        
        if not is_educational:
            st.error("❌ This doesn't appear to be study-related content.")
            st.warning("📚 Please upload educational materials like textbooks, articles, tutorials, or study notes.")
            
            # Clear the extracted text so it doesn't show stats
            st.session_state.extracted_text = ""
            return
    
    st.success("✅ Content verified as educational material!")
    
    # Generate Summary
    with st.spinner("📝 Generating summary..."):
        summary, error = generate_summary(text)
        if error:
            st.error(f"Summary failed: {error}")
        else:
            st.session_state.summary = summary
            st.success("✅ Summary generated!")
    
    # Generate Quiz
    with st.spinner("🎯 Creating quiz questions..."):
        quiz_data, error = generate_quiz(text)
        if error:
            st.error(f"Quiz failed: {error}")
        else:
            st.session_state.quiz_data = quiz_data
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.success("✅ Quiz ready!")
    
    # Store text for chatbot
    st.session_state.text_chunks = text
    
    st.success("✨ All features ready! Navigate through tabs above.")
    


def render_summary_tab():
    """Render Summary tab"""
    st.markdown("## 📝 Study Summary")
    
    if not st.session_state.summary:
        st.info("👈 Upload material in the sidebar to generate a summary!")
        return
    
    # Display summary
    with st.container():
        st.markdown("""
        <div class='summary-container'>
        """, unsafe_allow_html=True)
        st.markdown(st.session_state.summary)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Buttons row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download summary
        st.download_button(
            label="📥 Download Summary",
            data=st.session_state.summary,
            file_name="study_summary.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # Generate audio
        if st.button("🔊 Listen to Summary", use_container_width=True):
            with st.spinner("Generating audio..."):
                audio_bytes, error = generate_audio(st.session_state.summary)
                if error:
                    st.error(error)
                else:
                    st.audio(audio_bytes, format='audio/mp3')
                    st.success("✅ Audio ready!")
    
    with col3:
        # Export flashcards
        if st.session_state.quiz_data:
            flashcards = generate_flashcards(st.session_state.quiz_data)
            st.download_button(
                label="🃏 Export Flashcards",
                data=flashcards,
                file_name="study_flashcards.csv",
                mime="text/csv",
                use_container_width=True
            )
    

def render_quiz_tab():
    """Render Quiz tab"""
    st.markdown("## 🎯 Knowledge Check Quiz")
    
    if not st.session_state.quiz_data:
        st.info("👈 Upload material in the sidebar to generate a quiz!")
        return
    
    if not st.session_state.quiz_submitted:
        for i, q in enumerate(st.session_state.quiz_data):
            with st.container():
                st.markdown(f'<div class="quiz-card">', unsafe_allow_html=True)
                st.markdown(f"**Question {i+1}:** {q['question']}")
                
                answer = st.radio(
                    f"Select your answer for Q{i+1}:",
                    q['options'],
                    key=f"q_{i}",
                    index=None
                )
                
                if answer:
                    st.session_state.quiz_answers[i] = answer
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("✅ Submit Quiz", use_container_width=True):
                if len(st.session_state.quiz_answers) == len(st.session_state.quiz_data):
                    st.session_state.quiz_submitted = True
                    st.rerun()
                else:
                    st.warning("Please answer all questions!")
    
    else:
        score = 0
        for i, q in enumerate(st.session_state.quiz_data):
            user_answer = st.session_state.quiz_answers.get(i)
            is_correct = user_answer == q['correct']
            if is_correct:
                score += 1
            
            with st.container():
                if is_correct:
                    st.success(f"✅ **Q{i+1}:** {q['question']}")
                else:
                    st.error(f"❌ **Q{i+1}:** {q['question']}")
                
                st.markdown(f"**Your answer:** {user_answer}")
                st.markdown(f"**Correct answer:** {q['correct']}")
                st.markdown(f"**Explanation:** {q['explanation']}")
                st.markdown("---")
        
        score_percentage = (score / len(st.session_state.quiz_data)) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", f"{score}/{len(st.session_state.quiz_data)}")
        with col2:
            st.metric("Percentage", f"{score_percentage:.1f}%")
        with col3:
            if score_percentage >= 80:
                st.success("🌟 Excellent!")
            elif score_percentage >= 60:
                st.warning("📚 Good effort!")
            else:
                st.info("💪 Keep studying!")
        
        # ============================================
        # SAVE RESULTS & AWARD XP
        # ============================================
        
        # Calculate and award XP
        xp_earned = calculate_xp(score, len(st.session_state.quiz_data))
        st.session_state.xp += xp_earned
        
        # Save to history
        st.session_state.quiz_history.append({
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'score': score_percentage,
            'questions': len(st.session_state.quiz_data)
        })
        
        # Check for badges
        check_badges()
        
        # Show XP earned
        st.success(f"⭐ +{xp_earned} XP earned!")
        
        # Show fireworks for high scores
        if score_percentage >= 80:
            st.balloons()
        
        # Try Again button - generates NEW questions
        if st.button("🔄 Try Again with New Questions", use_container_width=True):
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            
            # Generate NEW quiz questions
            with st.spinner("🎯 Creating new questions..."):
                quiz_data, error = generate_quiz(st.session_state.extracted_text)
                if error:
                    st.error(f"Quiz failed: {error}")
                else:
                    st.session_state.quiz_data = quiz_data
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.success("✅ New questions generated!")
            
            st.rerun()

def render_progress_tab():
    """Render Progress & Achievements tab"""
    st.markdown("## 📊 Your Learning Dashboard")
    
    # Check if user has any activity
    if not st.session_state.quiz_history:
        st.info("📝 Complete your first quiz to see your progress!")
        
        # Show sample dashboard
        st.markdown("""
        ### What you'll see here:
        - 📈 Quiz score trends
        - 🏆 Badges and achievements
        - ⭐ XP points
        - 📊 Performance analytics
        """)
        return
    
    # ============================================
    # XP & STATS SECTION
    # ============================================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_quizzes = len(st.session_state.quiz_history)
        st.metric("📝 Total Quizzes", total_quizzes)
    
    with col2:
        avg_score = sum(q['score'] for q in st.session_state.quiz_history) / total_quizzes
        st.metric("📊 Average Score", f"{avg_score:.1f}%")
    
    with col3:
        st.metric("⭐ Total XP", st.session_state.xp)
    
    with col4:
        best_score = max(q['score'] for q in st.session_state.quiz_history)
        st.metric("🏅 Best Score", f"{best_score:.1f}%")
    
    st.markdown("---")
    
    # ============================================
    # PROGRESS CHART
    # ============================================
    st.markdown("### 📈 Score History")
    
    if len(st.session_state.quiz_history) > 1:
        import plotly.express as px
        
        df = pd.DataFrame(st.session_state.quiz_history)
        df['quiz_number'] = range(1, len(df) + 1)
        
        fig = px.line(
            df, 
            x='quiz_number', 
            y='score',
            title='Your Quiz Performance Over Time',
            labels={'quiz_number': 'Quiz Number', 'score': 'Score (%)'},
            markers=True
        )
        
        # Add a target line at 80%
        fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Target: 80%")
        
        fig.update_layout(
            yaxis_range=[0, 105],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Complete more quizzes to see your progress chart!")
    
    # ============================================
    # RECENT QUIZ RESULTS
    # ============================================
    st.markdown("### 📋 Recent Quiz Results")
    
    for i, quiz in enumerate(reversed(st.session_state.quiz_history[-5:])):
        with st.container():
            score_color = "🟢" if quiz['score'] >= 80 else "🟡" if quiz['score'] >= 60 else "🔴"
            st.markdown(f"{score_color} **Quiz {len(st.session_state.quiz_history) - i}** - {quiz['date']} - Score: {quiz['score']:.1f}%")
    
    st.markdown("---")
    
    # ============================================
    # BADGES SECTION
    # ============================================
    st.markdown("### 🏆 Your Achievements")
    
    if st.session_state.badges:
        badge_cols = st.columns(len(st.session_state.badges))
        
        badge_icons = {
            "First Quiz": "🎯",
            "High Scorer": "🌟",
            "Perfect Score": "💎",
            "Quiz Master": "👑"
        }
        
        for i, badge in enumerate(st.session_state.badges):
            with badge_cols[i]:
                icon = badge_icons.get(badge, "🏆")
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
                    <h1>{icon}</h1>
                    <p style='color: white !important;'><strong>{badge}</strong></p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Complete quizzes to earn badges!")
    
    # Show locked badges
    all_badges = ["First Quiz", "High Scorer", "Perfect Score", "Quiz Master"]
    locked_badges = [b for b in all_badges if b not in st.session_state.badges]
    
    if locked_badges:
        st.markdown("#### 🔒 Locked Badges")
        lock_cols = st.columns(len(locked_badges))
        for i, badge in enumerate(locked_badges):
            with lock_cols[i]:
                st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 10px; opacity: 0.6;'>
                    <p>🔒</p>
                    <p><small>{badge}</small></p>
                </div>
                """, unsafe_allow_html=True)

def render_chatbot_tab():
    """Render AI Tutor Chatbot tab"""
    st.markdown("## 🤖 AI Study Tutor")
    
    if not st.session_state.text_chunks:
        st.info("👈 Upload material in the sidebar to start chatting with your AI tutor!")
        return
    
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 StudyBuddy:</strong><br>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask a question about your study material:",
            key="chat_input",
            placeholder="e.g., Can you explain this concept in simpler terms?"
        )
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    if send_button and user_input:
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        with st.spinner("Thinking..."):
            response, error = chat_with_context(
                st.session_state.text_chunks,
                user_input,
                st.session_state.chat_history
            )
            
            if error:
                st.error(error)
            else:
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
        
        st.rerun()
    
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ============================================
# MAIN APP
# ============================================

def main():
    """Main application entry point"""
    
    # Header
    st.markdown('<h1 class="main-header">📚 AI Study Companion</h1>', unsafe_allow_html=True)
    st.markdown("### Your Personal AI-Powered Learning Assistant")
    st.markdown("---")
    
    # Render sidebar
    render_sidebar()
    
    # Main content area with tabs (NOW 4 TABS)
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🎯 Quiz", "🤖 AI Tutor", "📊 Progress"])
    
    with tab1:
        render_summary_tab()
    
    with tab2:
        render_quiz_tab()
    
    with tab3:
        render_chatbot_tab()
    
    with tab4:
        render_progress_tab()  # NEW TAB
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Built with Streamlit & Google Gemini</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()