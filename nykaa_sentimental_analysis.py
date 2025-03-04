# Core dependencies
pip>=24.0
setuptools>=75.0.0
wheel

# NumPy and Pandas Fix
numpy>=1.26.0
pandas==2.2.3

# Streamlit
streamlit==1.25.0

# Machine Learning Dependencies
torch==2.2.0
transformers==4.28.1
tokenizers==0.14.1  # Updated tokenizers version to avoid Rust issue

# Data Visualization
seaborn==0.13.2
matplotlib==3.10.1
plotly==6.0.0
wordcloud==1.9.4

# Additional dependencies required
rust  # Ensure Rust is installed for tokenizers
