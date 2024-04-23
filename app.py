from flask import Flask, request
import pdfplumber
import re
from dotenv import load_dotenv
import os
import voyageai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pinecone import Pinecone
import json
import dspy
from openai import OpenAI
import redis
import time
import uuid
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load the environment variables
load_dotenv()

# Initialize the Pinecone client and index
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("rag-chat")

# Initialize the Voyage AI client for vector embeds
VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY')
vo = voyageai.Client()
vo_max_size = 128 #voyage only takes 128 pieces of text at a time
breakpoint_percentile_threshold = 90 #percentile at which to split the document into sections

# OpenAI and dspy setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
gpt4 = dspy.OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
gpt4_turbo = dspy.OpenAI(model="gpt-4-turbo", api_key=OPENAI_API_KEY)
dspy.configure(lm=gpt4)
client = OpenAI()

#Redis setup
r = redis.from_url(os.environ['REDIS_URL'])
num_tries = 5

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['pdf']

def pdf_to_text(file_stream) -> str:
    """
    Converts a PDF file to text from a file stream. This function handles exceptions
    during the PDF read process and logs errors for troubleshooting.

    Args:
        file_stream: A file-like object containing the PDF data.
    
    Returns:
        str: The text extracted from the PDF file, or an error message if an exception occurs.
    """
    text_parts = []

    try:
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        return f"An error occurred: {e}"

    return ''.join(text_parts)

def combine_sentences(sentences, buffer_size=1):
    """
    Combines sentences into a single sentence with a buffer of buffer_size sentences on either side.

    Args:
        sentences (list): A list of dictionaries, each containing a 'sentence' key with the sentence text.
        buffer_size (int): The number of sentences to include on either side of the current sentence.
    
    Returns:
        sentences (list): The input list of dictionaries, with an additional 'combined_sentence' key containing the combined sentence.
    """
    for i in range(len(sentences)):
        combined_sentence = ''
        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]['sentence'] + ' '
        combined_sentence += sentences[i]['sentence']
        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined_sentence += ' ' + sentences[j]['sentence']
        sentences[i]['combined_sentence'] = combined_sentence
    return sentences

def calculate_cosine_distances(sentences):
    """
    Calculates the cosine distances between the embeddings of consecutive sentences.

    Args:
        sentences (list): A list of dictionaries, each containing a 'combined_sentence_embedding' key with the sentence embedding.
    
    Returns:
        distances (list): A list of cosine distances between consecutive sentences.
        sentences (list): The input list of dictionaries, with an additional 'distance_to_next' key containing the cosine distance to the next sentence.
    """
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        
        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences

@app.route('/pdf_to_pinecone', methods=['POST'])
def pdf_to_pinecone():
    """
    Given a PDF file, extracts the text, semantically chunks the text into sentences, and uploads the sentences to Pinecone.

    Args:
        pdf_path (str): The path to the PDF file.
        id (str): The unique identifier for the document.
    
    Returns:    
        response: The Flask response object.
    """
    try:
        data = request.json
        id = data.get('id', '')
        if 'file' not in request.files:
            response = app.response_class(
                response=json.dumps({"message": f"No file found"}),
                status=500,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        file = request.files['file']
        if file.filename == '':
            response = app.response_class(
                response=json.dumps({"message": f"No file found"}),
                status=500,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        if file and allowed_file(file.filename):
            # Process the file directly without saving it
            text = pdf_to_text(file)
        else:
            response = app.response_class(
                response=json.dumps({"message": f"Wrong file type."}),
                status=500,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        if text == '':
            response = app.response_class(
                response=json.dumps({"message": "Unable to extract text from the PDF file."}),
                status=400,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        single_sentences_list = re.split(r'(?<=[.?!])\s+', text)
        sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(single_sentences_list)]
        #Semantic chunking
        sentences = combine_sentences(sentences)
        combined_embeddings = []
        #vo only takes 128 pieces of text at a time
        i = 0
        while i < len(sentences):
            combined_embeddings += vo.embed([x['combined_sentence'] for x in sentences[i:i+vo_max_size]], model="voyage-large-2", input_type="document").embeddings
            i += vo_max_size
        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = combined_embeddings[i]
        distances, sentences = calculate_cosine_distances(sentences)
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list

        # Initialize the start index
        start_index = 0

        # Create a list to hold the grouped sentences
        chunks = []

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            
            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        # grouped_sentences now contains the chunked sentences
        chunk_embeddings = vo.embed(chunks, model="voyage-large-2", input_type="document").embeddings
        id_name = id
        vectors = []
        # Upload the chunks to Pinecone
        for i, chunk in enumerate(chunks):
            thisid = id_name + str(i)
            vector = chunk_embeddings[i]
            metadata = {"text": chunk, "pdf_id": id_name}
            full_dct = {"id": thisid, "values": vector, "metadata": metadata}
            vectors.append(full_dct)
        dct = index.upsert(vectors)
        #Check if all the chunks were uploaded
        if dct["upserted_count"] == len(chunks):
            response = app.response_class(
                response=json.dumps({"message": "Successfully uploaded the document to Pinecone."}),
                status=200,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Methods', 'POST')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        else:
            response = app.response_class(
                response=json.dumps({"message": "Failed to upload the document to Pinecone."}),
                status=501,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({"message": f"An error occurred: {str(e)}"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

class QueryReformatter(dspy.Signature):
    """Given a query, make it more detailed by asking implied subquestions for a vector search.
    If there isn't a clear way to make the query more detailed, return the original query."""

    query = dspy.InputField()
    new_query = dspy.OutputField(desc="The more detailed version of the query, assuming it is known the information is contained in the writing. ONLY GIVE THE QUERY, no additional text.")

class PineconeRM(dspy.Retrieve):
    """
    Retrieval model used in DSPy, reformats the query and retrieves the top k passages from Pinecone.
    """
    def __init__(self, id:str = "", k:int = 3):
        super().__init__(k=k)
        self.id = id

    def forward(self, query:str) -> dspy.Prediction:
        dspy.configure(lm=gpt4)
        queryref = dspy.Predict(QueryReformatter)
        query_redone = queryref(query=query).new_query
        voyage_call = vo.embed(query_redone, model="voyage-large-2", input_type="query")
        query_vector = voyage_call.embeddings[0]
        if not self.id:
            result = index.query(
                vector=query_vector,
                top_k=self.k,
                include_metadata=True
            )
        else:
            result = index.query(
                vector=query_vector,
                filter={
                    "pdf_id": self.id
                },
                top_k=self.k,
                include_metadata=True
            )
        text_strings = [i["metadata"]["text"] for i in result["matches"]]
        return dspy.Prediction(
            passages=text_strings
        )

class GenerateAnswer(dspy.Signature):
    """Answer questions with as ground-truth information as possible. If a 
    question isn't asked, return a default answer."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="complete, detailed answer to the question in max 3 sentences.")

class RAG(dspy.Module):
    """Retrieve, Answer, Generate model for question answering."""
    def __init__(self, num_passages=2, id:str = ""):
        super().__init__()

        self.retrieve = PineconeRM(id=id, k=num_passages)
        self.generate_answer = dspy.Predict(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        print(prediction)
        return dspy.Prediction(context=context, answer=prediction.answer)

@app.route('/rag_qa', methods=['POST'])
def rag_qa():
    """
    Given a question and an ID, retrieves the top k passages from Pinecone and generates an answer using the RAG model.
    """
    try:
        data = request.json
        id = data.get('id', '')
        query = data.get('query', '')
        rag = RAG(id=id)
        call = rag(question=query)
        print(call)
        text = call.answer
        add_to_redis(id, text, False)
        response = app.response_class(
            response=json.dumps({"answer": text}),
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({"message": f"An error occurred: {str(e)}"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/update_redis', methods=['POST'])
def update_redis():
    try:
        data = request.json
        id = data.get('id', '')
        message = data.get('message', '')
        user = data.get('user', False)
        add_to_redis(id, message, user)
        response = app.response_class(
            response=json.dumps({"message": "Successfully added message to Redis."}),
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({"message": f"An error occurred: {str(e)}"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
def add_to_redis(id: str, message: str, user: bool):
    r.rpush(id + "_list", json.dumps({"text": message, "user": user}))

@app.route('/get_messages', methods=['GET'])
def get_messages():
    """
    Returns the messages stored in Redis for a given ID.
    """
    try:
        id = request.args.get('id', '')
        messages = [message.decode('utf-8') for message in r.lrange(id + "_list", 0, -1)]
        response = app.response_class(
            response=json.dumps({"messages": messages}),
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        app.logger.error(f"Failed to retrieve messages: {str(e)}")
        response = app.response_class(
            response=json.dumps({"message": f"An error occurred: {str(e)}"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/generate_id', methods=['POST'])
def generate_id():
    """
    Generates a unique ID for a new document and stores it in the redis db.
    """
    try:
        id = str(uuid.uuid4())
        r.set(id, "placehold")
        response = app.response_class(
            response=json.dumps({"id": id}),
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({"message": f"An error occurred: {str(e)}"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/confirm_id', methods=['GET'])
def confirm_id():
    """Confirms whether an ID exists in the redis database"""
    try:
        id = request.args.get('id', '')
        if r.exists(id):
            response = app.response_class(
                response=json.dumps({"exists": True}),
                status=200,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        else:
            response = app.response_class(
                response=json.dumps({"exists": False}),
                status=200,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({"message": f"An error occurred: {str(e)}"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
if __name__ == '__main__':
    app.run(debug=True)