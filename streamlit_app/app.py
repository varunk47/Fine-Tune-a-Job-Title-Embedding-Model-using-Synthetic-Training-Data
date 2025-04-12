import os
# Set Streamlit configuration to disable file watcher
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'

import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import pandas as pd

# Set up page configuration and CSS.
st.set_page_config(page_title="Job Posting Search Engine", layout="centered")
st.markdown(
    """
    <style>
    .block-container {
        padding: 2rem 2rem !important;
        max-width: 1200px;
    }
    .section-spacing {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .header-container > div {
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper: detect device.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Callback function to handle search term changes
def on_change():
    # Detect if the search term has changed and force a rerun
    if "user_input" in st.session_state and st.session_state.user_input != st.session_state.saved_search:
        st.session_state.saved_search = st.session_state.user_input
        st.session_state.search_submitted = True
        st.session_state.app_state = "results"

st.title('Job Posting Search Engine')
device = get_device()

# Initialize session state variables.
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "saved_search" not in st.session_state:
    st.session_state.saved_search = ""
if "search_submitted" not in st.session_state:
    st.session_state.search_submitted = False
if "app_state" not in st.session_state:
    # "search": initial search input form,
    # "results": search results are available,
    # "similar_jobs": a job has been selected to view similar jobs.
    st.session_state.app_state = "search"

# ----- Functions for loading resources -----
@st.cache_resource
def load_fine_tuned_embeddings():
    embeddings = np.load(os.path.join('data', 'fine_tuned_embeddings.npy'))
    return embeddings

@st.cache_resource
def load_default_embeddings():
    embeddings = np.load(os.path.join('data', 'default_embeddings.npy'))
    return embeddings

@st.cache_resource
def load_job_postings():
    job_postings_df = pd.read_parquet(os.path.join('data', 'job_postings.parquet'))
    job_postings_df['posting'] = job_postings_df['job_posting_title'] + ' @ ' + job_postings_df['company']
    return job_postings_df['posting'].to_list()

@st.cache_resource
def load_fine_tuned_model():
    fine_tuned_model_path = os.path.join('data', 'fine_tuned_model')
    model = SentenceTransformer(fine_tuned_model_path, device=device)
    return model

@st.cache_resource
def load_default_model():
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
    return model

# ----- Load Resources -----
# For demonstration, limit to the first 5000 job postings.
fine_tuned_embeddings = torch.tensor(load_fine_tuned_embeddings()[:5000], device=device)
default_embeddings = torch.tensor(load_default_embeddings()[:5000], device=device)
job_postings = load_job_postings()[:5000]
fine_tuned_model = load_fine_tuned_model()
default_model = load_default_model()

# =============================================================================
# State Machine:
#
# app_state: "search" -> enter query, "results" -> display search results,
# "similar_jobs": a job has been selected to view similar jobs.
#
# Transitions:
# - When user types a query and submits, set app_state = "results"
# - When user clicks a "Show most similar jobs" button,
#       set st.session_state.selected_job and app_state = "similar_jobs"
# - When user clicks "Back to search",
#       clear selected_job, set app_state = "results" (or "search" if you prefer)
# =============================================================================

if st.session_state.app_state == "similar_jobs" and st.session_state.selected_job is not None:
    # Similar-jobs view.
    selected_index = st.session_state.selected_job
    st.header("Similar Jobs for:")
    st.write(f"**{job_postings[selected_index]}**")
    st.markdown("<hr>", unsafe_allow_html=True)
    # Compute similar jobs for both models.
    with torch.inference_mode():
        # Default model similar jobs.
        default_embedding = default_embeddings[selected_index]
        default_sim = torch.inner(default_embedding, default_embeddings)
        default_sim[selected_index] = -1  # Exclude the job itself.
        default_top_indices = torch.argsort(default_sim, descending=True)[:5]
        # Fine-tuned model similar jobs.
        finetuned_embedding = fine_tuned_embeddings[selected_index]
        finetuned_sim = torch.inner(finetuned_embedding, fine_tuned_embeddings)
        finetuned_sim[selected_index] = -1
        finetuned_top_indices = torch.argsort(finetuned_sim, descending=True)[:5]
    st.markdown(
        """
        <div class="section-spacing">
            <h3 style="margin-bottom:1rem;">Similar Jobs (Default vs. Fine-Tuned)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Display headers.
    col_rank, col_default, col_finetuned = st.columns([0.5, 4, 4])
    with col_rank:
        st.write("")  # placeholder for rank header.
    with col_default:
        st.markdown("<div class='header-container'><div>Default Model</div></div>", unsafe_allow_html=True)
    with col_finetuned:
        st.markdown("<div class='header-container'><div>Fine-Tuned Model</div></div>", unsafe_allow_html=True)
    # Show similar jobs result rows.
    for i in range(5):
        col_rank, col_default, col_finetuned = st.columns([0.5, 4, 4])
        with col_rank:
            st.markdown(f"<h4>{i+1}.</h4>", unsafe_allow_html=True)
        with col_default:
            idx = default_top_indices[i].item()
            st.write(f"**{job_postings[idx]}**")
            st.write(f"Score: {default_sim[idx]:.4f}")
        with col_finetuned:
            idx = finetuned_top_indices[i].item()
            st.write(f"**{job_postings[idx]}**")
            st.write(f"Score: {finetuned_sim[idx]:.4f}")
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Back to search", key="clear_selection"):
        st.session_state.selected_job = None
        # Transition back to "results" without wiping the search query.
        st.session_state.app_state = "results"
        st.rerun()
else:
    # Either in the initial search mode or in search results mode.
    user_input = st.text_input(
        "Enter a job title:",
        value=st.session_state.saved_search,
        key="user_input",
        on_change=on_change
    )
    
    # Create a button to explicitly submit the search
    search_button = st.button("Search", key="search_button")
    if search_button and user_input:
        st.session_state.saved_search = user_input
        st.session_state.search_submitted = True
        st.session_state.app_state = "results"
        st.rerun()
    
    # Process search if user input is provided and search was submitted
    if st.session_state.search_submitted and st.session_state.saved_search:
        # Reset the submitted flag so we don't keep searching on every rerun
        st.session_state.search_submitted = False
        
        with torch.inference_mode():
            default_query_embedding = default_model.encode(
                [st.session_state.saved_search],
                normalize_embeddings=True,
                convert_to_tensor=True,
            )[0]
            finetuned_query_embedding = fine_tuned_model.encode(
                [st.session_state.saved_search],
                normalize_embeddings=True,
                convert_to_tensor=True,
            )[0]
            default_sim = torch.inner(default_query_embedding, default_embeddings)
            finetuned_sim = torch.inner(finetuned_query_embedding, fine_tuned_embeddings)
            top10_default = torch.argsort(default_sim, descending=True)[:10]
            top10_finetuned = torch.argsort(finetuned_sim, descending=True)[:10]
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-spacing">
                <h3 style="margin-bottom:1rem;">Top Matches (Default vs. Fine-Tuned)</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Column headers above search results.
        col_rank, col_default, col_finetuned = st.columns([0.5, 4, 4])
        with col_rank:
            st.write("")  # empty header for rank.
        with col_default:
            st.markdown("<div class='header-container'><div>Default Model</div></div>", unsafe_allow_html=True)
        with col_finetuned:
            st.markdown("<div class='header-container'><div>Fine-Tuned Model</div></div>", unsafe_allow_html=True)
        # Build search results rows.
        for i in range(len(top10_default)):
            col_rank, col_default, col_finetuned = st.columns([0.5, 4, 4])
            with col_rank:
                st.markdown(f"<h4>{i+1}.</h4>", unsafe_allow_html=True)
            with col_default:
                job_index = top10_default[i].item()
                st.write(f"**{job_postings[job_index]}**")
                st.write(f"Score: {default_sim[job_index]:.4f}")
                if st.button("Show most similar jobs", key=f"default_{job_index}"):
                    st.session_state.selected_job = job_index
                    st.session_state.app_state = "similar_jobs"
                    st.rerun()
            with col_finetuned:
                job_index = top10_finetuned[i].item()
                st.write(f"**{job_postings[job_index]}**")
                st.write(f"Score: {finetuned_sim[job_index]:.4f}")
                if st.button("Show most similar jobs", key=f"finetuned_{job_index}"):
                    st.session_state.selected_job = job_index
                    st.session_state.app_state = "similar_jobs"
                    st.rerun()
    else:
        st.info("Please enter a job title to start searching.")