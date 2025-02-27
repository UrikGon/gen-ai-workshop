import base64
import io
import json
from typing import Optional

import boto3
import streamlit as st
from PIL import Image
from botocore.exceptions import ClientError

st.title("Building with Bedrock")
st.subheader("Image Understanding Demo")

REGION = "us-east-1"
SUPPORTED_FORMATS = ["PNG", "JPEG", "JPG"]
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB

# Cache the Bedrock client
@st.cache_resource
def get_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=REGION,
    )

def call_claude_sonnet(base64_string: str) -> str:
    try:
        prompt_config = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_string,
                            },
                        },
                        {"type": "text", "text": "Provide a caption for this image"},
                    ],
                }
            ],
        }

        body = json.dumps(prompt_config)
        bedrock_runtime = get_bedrock_client()
        
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        return response_body.get("content")[0].get("text", "No description available")
    
    except ClientError as e:
        st.error(f"Error calling Bedrock: {str(e)}")
        return "Error generating description"
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return "Error generating description"

def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    try:
        with io.BytesIO() as buffer:
            # Optimize image before converting to base64
            if format.upper() in ["JPEG", "JPG"]:
                image.save(buffer, format, optimize=True, quality=85)
            else:
                image.save(buffer, format, optimize=True)
            return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        st.error(f"Error converting image: {str(e)}")
        return ""

def validate_image(image_file) -> Optional[Image.Image]:
    try:
        # Check file size
        file_size = len(image_file.getvalue())
        if file_size > MAX_IMAGE_SIZE:
            st.error(f"Image size exceeds {MAX_IMAGE_SIZE/1024/1024}MB limit")
            return None
        
        # Open and validate image
        img = Image.open(image_file)
        if img.format.upper() not in SUPPORTED_FORMATS:
            st.error(f"Unsupported image format. Please use {', '.join(SUPPORTED_FORMATS)}")
            return None
        
        return img
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Streamlit UI
user_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
col1, col2 = st.columns(2)

if user_image is not None:
    # Validate and process image
    processed_image = validate_image(user_image)
    
    if processed_image:
        # Show image in column 1
        col1.image(processed_image, caption="Uploaded Image", use_container_width=True)
        
        # Add button to describe the image
        if col2.button("Describe Image"):
            # Convert image to base64
            base64_string = pil_to_base64(processed_image)
            
            if base64_string:
                # Call Claude Sonnet to describe the image
                with st.spinner("Generating description..."):  # Cambiado aquí
                    description = call_claude_sonnet(base64_string)
                    col2.write("### Image Description")
                    col2.write(description)
        else:
            col2.write("Click the button to generate a description")
else:
    col2.write("No image uploaded")