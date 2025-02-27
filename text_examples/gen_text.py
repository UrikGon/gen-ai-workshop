import time
import boto3
from typing import List, Dict, Any, Optional

# Constants
DEFAULT_TEMPERATURE = 0.5
DEFAULT_REGION = "us-east-1"

# Available model IDs
model_ids = [
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-1-405b-instruct-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "mistral.mistral-large-2402-v1:0",
    "us.amazon.nova-pro-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.amazon.nova-micro-v1:0",
]

# Setup bedrock client at module level
try:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=DEFAULT_REGION,
    )
except Exception as e:
    print(f"Error initializing Bedrock client: {str(e)}")
    raise

def validate_model_id(model_id: str) -> None:
    """
    Validates if the provided model ID is in the list of available models.
    
    Args:
        model_id (str): The model ID to validate.
        
    Raises:
        ValueError: If the model ID is not valid.
    """
    if model_id not in model_ids:
        raise ValueError(f"Invalid model ID: {model_id}")

def generate_conversation(
    model_id: str,
    system_prompts: List[Dict[str, str]],
    messages: List[Dict[str, Any]]
) -> str:
    """
    Sends messages to a model and returns the generated response.
    
    Args:
        model_id (str): The model ID to use.
        system_prompts (List[Dict[str, str]]): The system prompts for the model.
        messages (List[Dict[str, Any]]): The messages to send to the model.
        
    Returns:
        str: The conversation response from the model.
        
    Raises:
        Exception: If there's an error during conversation generation.
    """
    try:
        print(f"Generating message with model {model_id}")
        
        inference_config = {"temperature": DEFAULT_TEMPERATURE}
        
        response = bedrock_runtime.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
        )
        
        # Log token usage
        token_usage = response.get("usage", {})
        print(f"Input tokens: {token_usage.get('inputTokens', 0)}")
        print(f"Output tokens: {token_usage.get('outputTokens', 0)}")
        print(f"Total tokens: {token_usage.get('totalTokens', 0)}")
        print(f"Stop reason: {response.get('stopReason', 'Unknown')}")
        
        return response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
        
    except Exception as e:
        print(f"Error generating conversation: {str(e)}")
        raise

def summarize_text(text: str) -> str:
    """
    Summarizes the provided text using a generative AI model.
    
    Args:
        text (str): The text to summarize.
        
    Returns:
        str: The generated summary.
    """
    if not text:
        return ""
        
    model_id = "us.amazon.nova-pro-v1:0"
    validate_model_id(model_id)
    
    system_prompts = [
        {"text": "You are an app that creates summaries of text in 50 words or less."}
    ]
    messages = [{
        "role": "user",
        "content": [{"text": f"Summarize the following text: {text}."}],
    }]

    return generate_conversation(model_id, system_prompts, messages)

def sentiment_analysis(text: str) -> str:
    """
    Performs sentiment analysis on the provided text.
    
    Args:
        text (str): The text to analyze.
        
    Returns:
        str: JSON string containing sentiment analysis results.
    """
    if not text:
        return "{}"
        
    model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    validate_model_id(model_id)
    
    system_prompts = [{
        "text": "You are a bot that takes text and returns a JSON object of sentiment analysis."
    }]
    messages = [{
        "role": "user",
        "content": [{"text": text}],
    }]

    return generate_conversation(model_id, system_prompts, messages)

def perform_qa(question: str, text: str) -> str:
    """
    Performs Q&A operation based on the provided text and question.
    
    Args:
        question (str): The question to answer.
        text (str): The context text to use for answering.
        
    Returns:
        str: The answer to the question.
    """
    if not question or not text:
        return "Invalid input: Question and text are required."
        
    model_id = "mistral.mistral-large-2402-v1:0"
    validate_model_id(model_id)
    
    system_prompts = [{
        "text": f"Given the following text, answer the question. If the answer is not in the text, 'say you do not know'. Here is the text: {text}"
    }]
    messages = [{
        "role": "user",
        "content": [{"text": question}],
    }]

    return generate_conversation(model_id, system_prompts, messages)

def main():
    """Main function to demonstrate the usage of the text processing functions."""
    try:
        # Sample text for testing
        text = """Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) 
        from leading AI companies like AI21 Labs, Anthropic, Cohere, Luma, Meta, Mistral AI, poolside (coming soon), 
        Stability AI, and Amazon through a single API, along with a broad set of capabilities you need to build generative 
        AI applications with security, privacy, and responsible AI. Using Amazon Bedrock, you can easily experiment with 
        and evaluate top FMs for your use case, privately customize them with your data using techniques such as fine-tuning 
        and Retrieval Augmented Generation (RAG), and build agents that execute tasks using your enterprise systems and 
        data sources. Since Amazon Bedrock is serverless, you don't have to manage any infrastructure, and you can securely 
        integrate and deploy generative AI capabilities into your applications using the AWS services you are already 
        familiar with"""

        print("\n=== Summarization Example ===")
        summary = summarize_text(text)
        print(f"Summary:\n{summary}")
        time.sleep(2)

        print("\n=== Sentiment Analysis Example ===")
        sentiment_analysis_json = sentiment_analysis(text)
        print(f"Sentiment Analysis JSON:\n{sentiment_analysis_json}")
        time.sleep(2)

        print("\n=== Q&A Example ===")
        questions = [
            "How many companies have models in Amazon Bedrock?",
            "Can Amazon Bedrock support RAG?",
            "When was Amazon Bedrock announced?"
        ]

        for question in questions:
            print(f"\nQuestion: {question}")
            answer = perform_qa(question, text)
            print(f"Answer: {answer}")
            time.sleep(2)

    except Exception as e:
        print(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()

