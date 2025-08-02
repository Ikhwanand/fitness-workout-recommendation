# Fitness Workout Recommendation System

## 🏋️ Overview
A content-based recommendation system for fitness workouts that suggests personalized workout programs based on user preferences and fitness goals. The system uses TF-IDF for text features and cosine similarity for recommendations.

## 📂 Project Structure
```
├── .gitignore
├── .python-version
├── app.py            # Streamlit web application
├── data/
│   ├── fitness_and_workout_dataset.csv  # Main dataset
│   └── about-dataset.md                 # Dataset documentation
├── notebook/
│   └── notebook-1.ipynb  # Jupyter notebook with EDA and model development
├── pyproject.toml
├── requirements.txt    # Python dependencies
└── uv.lock
```

## 🚀 Features
- Content-based recommendation using workout attributes
- Hybrid approach combining text and numeric features
- Interactive Streamlit web interface
- Detailed exploratory data analysis

## 🔧 Installation
1. Clone the repository
```bash
git clone https://github.com/ikhwanand/fitness-workout-recommendation.git
cd fitness-workout-recommendation
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run app.py
```

## 📊 Dataset
Contains 2598 workout programs with attributes:
- Program length (weeks)
- Time per workout (minutes)
- Fitness level (Beginner, Intermediate, Advanced)
- Fitness goals (Bodybuilding, Powerlifting, etc.)
- Equipment required
- Total exercises

See <mcfile name="about-dataset.md" path="data/about-dataset.md"></mcfile> for detailed dataset description.

## 🤖 Recommendation Algorithm
1. Text features (level, goal, equipment) processed with TF-IDF
2. Numeric features (duration, length, exercises) standardized
3. Hybrid similarity score combining both feature types
4. Top 5 most similar workouts recommended

## 📈 Results
Evaluation metrics from notebook:
- Average level match: 97.6%
- Average goal match: 99.4%
- Average equipment match: 98.6%
- Average similarity score: 0.903

## 📝 License
MIT