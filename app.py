import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import ast
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="🏋️ Fitness Workout Recommender",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .workout-card {
        background-color: #ffffff; /* Tidak perlu diubah jika putih diinginkan */
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #FF6B6B;
        color: #333333; /* Tambahkan warna teks agar lebih jelas */
    }
    .metric-card {
        background-color: #ffffff; /* Mengubah menjadi putih untuk memastikan kejelasan */
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: #333333; /* Tambahkan warna teks agar lebih jelas */
        border: 1px solid #d3d3d3; /* Menambahkan border untuk memisahkan area yang berbeda */
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess data"""
    df = pd.read_csv('./data/fitness_and_workout_dataset.csv')
    df.dropna(inplace=True)
    
    # Extract first values from lists
    def extract_first_value(text):
        if pd.isna(text):
            return 'Unknown'
        try:
            values = ast.literal_eval(text)
            return values[0] if values else 'Unknown'
        except:
            return 'Unknown'
    
    df['primary_level'] = df['level'].apply(extract_first_value)
    df['primary_goal'] = df['goal'].apply(extract_first_value)
    
    return df

@st.cache_data
def create_features(df):
    """Create feature matrices for recommendation"""
    # Combine categorical features
    df['content_text'] = (df['primary_level'] + ' ' + 
                         df['primary_goal'] + ' ' + 
                         df['equipment'])
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    content_matrix = tfidf.fit_transform(df['content_text'])
    
    # Normalize numeric features
    scaler = StandardScaler()
    numeric_features = ['program_length', 'time_per_workout', 'total_exercises']
    numeric_scaled = scaler.fit_transform(df[numeric_features])
    
    return content_matrix, numeric_scaled, tfidf, scaler

def recommend_workouts(target_idx, df, content_matrix, numeric_scaled, 
                      top_k=5, content_weight=0.7, numeric_weight=0.3):
    """Generate workout recommendations"""
    # Content similarity
    content_sim = cosine_similarity(content_matrix[target_idx], content_matrix).flatten()
    
    # Numeric similarity
    target_numeric = numeric_scaled[target_idx].reshape(1, -1)
    numeric_distances = np.linalg.norm(numeric_scaled - target_numeric, axis=1)
    numeric_sim = 1 / (1 + numeric_distances)
    
    # Combined similarity
    combined_sim = content_weight * content_sim + numeric_weight * numeric_sim
    combined_sim[target_idx] = -1  # Exclude target
    
    # Get top recommendations
    top_indices = combined_sim.argsort()[-top_k:][::-1]
    top_scores = combined_sim[top_indices]
    
    return top_indices, top_scores

def main():
    # Header
    st.markdown('<h1 class="main-header">🏋️ Fitness Workout Recommender 💪</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading workout data...'):
        df = load_data()
        content_matrix, numeric_scaled, tfidf, scaler = create_features(df)
    
    # Sidebar for filters
    st.sidebar.header("🔍 Filter Workouts")
    
    # Filter options
    levels = df['primary_level'].unique()
    goals = df['primary_goal'].unique()
    equipment_types = df['equipment'].unique()
    
    selected_level = st.sidebar.selectbox("Fitness Level", ['All'] + list(levels))
    selected_goal = st.sidebar.selectbox("Fitness Goal", ['All'] + list(goals))
    selected_equipment = st.sidebar.selectbox("Equipment", ['All'] + list(equipment_types))
    
    # Filter dataframe
    filtered_df = df.copy()
    if selected_level != 'All':
        filtered_df = filtered_df[filtered_df['primary_level'] == selected_level]
    if selected_goal != 'All':
        filtered_df = filtered_df[filtered_df['primary_goal'] == selected_goal]
    if selected_equipment != 'All':
        filtered_df = filtered_df[filtered_df['equipment'] == selected_equipment]
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📋 Select a Workout")
        
        if len(filtered_df) == 0:
            st.warning("No workouts found with selected filters!")
            return
        
        # Display filtered workouts
        workout_options = {}
        for idx, row in filtered_df.head(20).iterrows():
            title = row['title'][:50] + "..." if len(row['title']) > 50 else row['title']
            workout_options[f"{title} ({row['primary_level']}, {row['time_per_workout']:.0f}min)"] = idx
        
        selected_workout_key = st.selectbox("Choose a workout:", list(workout_options.keys()))
        selected_idx = workout_options[selected_workout_key]
        
        # Display selected workout details
        selected_workout = df.iloc[selected_idx]
        
        st.markdown(f"""
        <div class="workout-card">
            <h4>🎯 Selected Workout</h4>
            <p><strong>Title:</strong> {selected_workout['title']}</p>
            <p><strong>Level:</strong> {selected_workout['primary_level']}</p>
            <p><strong>Goal:</strong> {selected_workout['primary_goal']}</p>
            <p><strong>Equipment:</strong> {selected_workout['equipment']}</p>
            <p><strong>Duration:</strong> {selected_workout['time_per_workout']:.0f} minutes</p>
            <p><strong>Program Length:</strong> {selected_workout['program_length']:.0f} weeks</p>
            <p><strong>Total Exercises:</strong> {selected_workout['total_exercises']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.header("🎯 Recommendations for You")
        
        # Generate recommendations
        recommendations, scores = recommend_workouts(
            selected_idx, df, content_matrix, numeric_scaled, top_k=5
        )
        
        # Display recommendations
        for i, (idx, score) in enumerate(zip(recommendations, scores), 1):
            rec_workout = df.iloc[idx]
            
            # Create expandable recommendation card
            with st.expander(f"#{i} {rec_workout['title'][:60]}... (Match: {score:.1%})"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write(f"**Level:** {rec_workout['primary_level']}")
                    st.write(f"**Goal:** {rec_workout['primary_goal']}")
                    st.write(f"**Equipment:** {rec_workout['equipment']}")
                
                with col_b:
                    st.write(f"**Duration:** {rec_workout['time_per_workout']:.0f} minutes")
                    st.write(f"**Program Length:** {rec_workout['program_length']:.0f} weeks")
                    st.write(f"**Total Exercises:** {rec_workout['total_exercises']}")
                
                # Progress bar for similarity score
                st.progress(score)
                
                # Description if available
                if pd.notna(rec_workout['description']) and rec_workout['description'].strip():
                    st.write(f"**Description:** {rec_workout['description'][:200]}...")
    
    # Analytics section
    st.header("📊 Recommendation Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate match statistics
    target = df.iloc[selected_idx]
    recommended = df.iloc[recommendations]
    
    level_matches = (recommended['primary_level'] == target['primary_level']).sum()
    goal_matches = (recommended['primary_goal'] == target['primary_goal']).sum()
    equipment_matches = (recommended['equipment'] == target['equipment']).sum()
    avg_similarity = scores.mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{level_matches}/5</h3>
            <p>Level Matches</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{goal_matches}/5</h3>
            <p>Goal Matches</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{equipment_matches}/5</h3>
            <p>Equipment Matches</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_similarity:.1%}</h3>
            <p>Avg Similarity</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization
    st.subheader("📈 Comparison Visualization")
    
    # Create comparison charts
    col1, col2, col3 = st.columns(3)
    
    numeric_cols = ['program_length', 'time_per_workout', 'total_exercises']
    col_names = ['Program Length (weeks)', 'Duration (minutes)', 'Total Exercises']
    
    for i, (col, col_name) in enumerate(zip(numeric_cols, col_names)):
        target_val = target[col]
        rec_vals = recommended[col].tolist()
        
        fig = go.Figure()
        
        # Add target line
        fig.add_vline(x=target_val, line_dash="dash", line_color="red", 
                     annotation_text="Target", annotation_position="top")
        
        # Add histogram
        fig.add_trace(go.Histogram(x=rec_vals, name="Recommendations", 
                                  marker_color="skyblue", opacity=0.7))
        
        fig.update_layout(
            title=col_name,
            xaxis_title="Value",
            yaxis_title="Count",
            height=300
        )
        
        if i == 0:
            col1.plotly_chart(fig, use_container_width=True)
        elif i == 1:
            col2.plotly_chart(fig, use_container_width=True)
        else:
            col3.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()