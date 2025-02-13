import json

import boto3
import streamlit as st

st.title("Building with Bedrock")  # Title of the application
st.subheader("Image Generation Demo")

REGION = "us-east-1"

# Define bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)


def generate_image_nova(text):
    """
    Generate an image using Amazon Nova Canvas model on demand.
    Args:
        model_id (str): The model ID to use.
        body (str) : The request body to use.
    Returns:
        image_bytes (bytes): The image generated by the model.
    """

    model_id = "amazon.nova-canvas-v1:0"

    accept = "application/json"
    content_type = "application/json"

    body = json.dumps(
        {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": text},
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 1024,
                "width": 1024,
                "cfgScale": 8.0,
                "seed": 0,
            },
        }
    )

    response = bedrock_runtime.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )
    response_body = json.loads(response.get("body").read())

    base64_image = response_body.get("images")[0]

    return base64_image


model = st.selectbox("Select model", ["Amazon Nova Canvas"])
