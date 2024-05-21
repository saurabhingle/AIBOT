import warnings
warnings.filterwarnings("ignore", message="If you use `@root_validator` with pre=False.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Importing FAISS from langchain.vectorstores is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Importing OpenAIEmbeddings from langchain.embeddings is deprecated.*", category=UserWarning)

from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings  # Updated import

import openai
import pandas as pd
from flask import Flask, request, jsonify, render_template
from langchain.chains import RetrievalQA
import time

# List of OpenAI API Keys
api_keys = [
    "sk-proj-ElESH4fNQ8xL0UKa1jouT3BlbkFJDvHy3QHd5g1uHmQhbtkM", 
    "sk-proj-JCqi4YQPRfE0By9ogt48T3BlbkFJTmcQTtT5b0XPpGyoDtem",
    "sk-proj-IVQJVTs78oHwN09dCwDfT3BlbkFJUauZGNWPvLF0Nvs5F3KG"
]
current_key_index = 0

# Function to switch API key
def switch_api_key():
    global current_key_index
    current_key_index = (current_key_index + 1) % len(api_keys)
    openai.api_key = api_keys[current_key_index]
    print(f"Switched to API key index {current_key_index}")

# Set initial API key
openai.api_key = api_keys[current_key_index]

# Load Movie Data from movies.csv
def load_movie_data():
    df = pd.read_csv("movies.csv")
    # Map the existing columns to the required columns
    df = df.rename(columns={
        "Series_Title": "title",
        "Poster_Link": "thumbnail",
        # Assuming we use "IMDB_Rating" as a placeholder for "url", since no url column exists
        # Alternatively, create a URL if there's a pattern or base URL to follow
        "IMDB_Rating": "url"
    })
    # Ensure the DataFrame has the necessary columns
    if not {'title', 'thumbnail', 'url'}.issubset(df.columns):
        raise ValueError("CSV file must contain 'Series_Title', 'Poster_Link', and 'IMDB_Rating' columns.")
    return df.to_dict(orient='records')

movie_data = load_movie_data()

# Create a FAISS vector store from movie data
embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)

# Retry logic for embedding creation with API key switching
def create_faiss_vector_store(movies, embedding, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return FAISS.from_texts([movie['title'] for movie in movies], embedding)
        except openai.RateLimitError as e:
            print(f"Rate limit exceeded. Switching API key and retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
            switch_api_key()
            embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
            time.sleep(delay)
    raise RuntimeError("Exceeded rate limit and retries")

vector_store = create_faiss_vector_store(movie_data, embedding)

# Function to create the movie search chain
def create_movie_search_chain():
    # Using FAISS directly as the retriever
    retriever = vector_store.as_retriever()
    llm = OpenAI(openai_api_key=openai.api_key)
    return RetrievalQA(llm=llm, retriever=retriever)

movie_search_chain = create_movie_search_chain()

# Create Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Search route
@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    result = run_query(query)
    return jsonify(result)

# Function to handle movie queries using LangChain and OpenAI
def run_query(query):
    response = movie_search_chain(query)
    movies = response['documents']
    results = []
    for movie in movies:
        results.append({
            "title": movie["title"],
            "thumbnail": movie["thumbnail"],
            "url": movie["url"]
        })
    return {"results": results}

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
