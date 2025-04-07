import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Custom CSS with modern, clean design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #ff4b4b;
        --primary-light: #ff4b4b;
        --primary-dark: #ff4b4b;
        --text: #1f2937;
        --text-light: #6b7280;
        --bg: #f9fafb;
        --card: #ffffff;
        --border: #e5e7eb;
        --success: #10b981;
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    body {
        background-color: var(--bg);
        color: var(--text);
    }
    
    .stApp {
        background-color: var(--bg);
    }
    
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .title {
        color: var(--text);
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: var(--text-light);
        font-weight: 400;
        font-size: 1rem;
        margin-bottom: 0;
    }
    
    .divider {
        height: 1px;
        background-color: var(--border);
        
        margin: 1.5rem 0;
        opacity: 0.5;
    }
    
    .tool-name {
        color: var(--primary);
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .tool-name::before {
        content: "✨";
    }
    
    .input-label {
        font-weight: 500;
        color: var(--text);
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .generated-title {
        background-color: var(--card);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--border);
        transition: all 0.2s ease;
    }
    
    .generated-title:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-color: var(--primary-light);
    }
    
    .title-number {
        display: inline-block;
        background-color: var(--primary);
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 6px;
        text-align: center;
        line-height: 24px;
        margin-right: 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .stTextArea textarea {
        border-radius: 8px !important;
        padding: 12px !important;
        border: 1px solid var(--border) !important;
        min-height: 150px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1) !important;
    }
    
    .stSlider .style {
        color: var(--text) !important;
        
    }
    
    .stSlider .st-ag {
    background-color: var(--primary) !important;
}

.stSlider .st-af {
    color: var(--primary) !important;
}

.stButton button {
    background-color: var(--primary) !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    border: none !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    width: 100%;
}

.stButton button:hover {
    background-color: var(--primary-dark) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.3) !important;
}
  .stFormSubmitButton > button {
            background-color:var(--primary) !important; /* Example: Green background */
            color: white;
            padding: 10px 20px; /* Example: Padding */
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .stFormSubmitButton > button:hover {
            background-color:var(--primary-dark) !important; /* Example: Darker green on hover */
        }  
    .stSpinner > div {
        border-color: var(--primary) transparent transparent transparent !important;
    }
    
    .success-message {
        background-color: rgba(16, 185, 129, 0.1);
        color: var(--success);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid rgba(16, 185, 129, 0.2);
        font-weight: 500;
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: var(--text-light);
        font-size: 0.875rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border);
    }
    
    .powered-by {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        color: var(--text-light);
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    folder_path = "Model_1_80"  # Update this path
    tokenizer = T5Tokenizer.from_pretrained(folder_path)
    model = T5ForConditionalGeneration.from_pretrained(folder_path)
    return model, tokenizer

# Generate titles function
def generate_titles_sampling(text, num_titles=3):
    input_text = "generate title: " + text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    
    outputs = model.generate(
        input_ids, 
        max_length=32, 
        num_return_sequences=num_titles, 
        do_sample=True, 
        top_k=50, 
        top_p=0.95, 
        temperature=0.7
    )

    titles = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return titles

# Load model
try:
    model, tokenizer = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App layout
st.markdown("""
    <div class="header">
        <h1 class="title">Story Title Generator</h1>
        <p class="subtitle">Create compelling titles for your stories with AI</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="tool-name">Title Wizard</div>', unsafe_allow_html=True)

# Input section
with st.form("title_form"):
    st.markdown('<div class="input-label">Enter your story content</div>', unsafe_allow_html=True)
    story_description = st.text_area(
        "",
        placeholder="Paste your story content here to generate creative titles...",
        height=200,
        label_visibility="collapsed"
    )
    
    num_titles = st.slider("Number of titles to generate", 1, 5, 3, help="Select how many title options you'd like to generate")
    
    submitted = st.form_submit_button("Generate Titles", type="primary")

# Generation and output
if submitted and story_description:
    with st.spinner("Generating creative titles..."):
        try:
            titles = generate_titles_sampling(story_description, num_titles)
            
            st.markdown("""
                <div class="success-message">
                    ✨ Your generated titles are ready!
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="input-label">Generated titles</div>', unsafe_allow_html=True)
            
            for i, title in enumerate(titles, 1):
                clean_title = title.replace("generate title: ", "")
                st.markdown(
                    f"""
                    <div class='generated-title'>
                        <span class='title-number'>{i}</span>
                        <strong>{clean_title}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        except Exception as e:
            st.error(f"Error generating titles: {e}")

elif submitted and not story_description:
    st.warning("Please enter some story content to generate titles.")

# Footer
st.markdown("""
    <div class='footer'>
        <p>Transform your story ideas into captivating titles</p>
       
    </div>
""", unsafe_allow_html=True)