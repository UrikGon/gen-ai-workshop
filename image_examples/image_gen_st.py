import json
import base64
from io import BytesIO
import boto3
import streamlit as st
from typing import Optional
from botocore.exceptions import ClientError

# Cache the Bedrock client to avoid recreating it on every rerun
@st.cache_resource
def get_bedrock_client(region: str = "us-east-1") -> boto3.client:
    """
    Create and cache the Bedrock client.
    Args:
        region (str): AWS region name
    Returns:
        boto3.client: Bedrock runtime client
    """
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=region
    )

def base64_to_image(base64_string: str) -> BytesIO:
    """
    Convert a base64 string to an image that can be displayed in Streamlit
    Args:
        base64_string (str): The base64 encoded image
    Returns:
        BytesIO: The decoded image bytes
    Raises:
        ValueError: If the base64 string is invalid
    """
    try:
        return BytesIO(base64.b64decode(base64_string))
    except Exception as e:
        raise ValueError(f"Failed to decode base64 string: {str(e)}")

def generate_image_nova(
    text: str,
    height: int = 1024,
    width: int = 1024,
    cfg_scale: float = 8.0,
    seed: int = 0
) -> Optional[str]:
    """
    Generate an image using Amazon Nova Canvas model on demand.
    Args:
        text (str): The prompt text for image generation
        height (int): Height of the generated image
        width (int): Width of the generated image
        cfg_scale (float): Configuration scale parameter
        seed (int): Random seed for generation
    Returns:
        Optional[str]: Base64 encoded image or None if generation fails
    """
    model_id = "amazon.nova-canvas-v1:0"
    accept = "application/json"
    content_type = "application/json"

    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": text},
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": height,
            "width": width,
            "cfgScale": cfg_scale,
            "seed": seed,
        },
    })

    try:
        bedrock_runtime = get_bedrock_client()
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            accept=accept,
            contentType=content_type
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("images")[0]
    except ClientError as e:
        st.error(f"AWS API Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Building with Bedrock",
        page_icon="🎨",
        layout="wide"
    )

    st.title("Building with Bedrock")
    st.subheader("Image Generation Demo")

    # Create two columns for better layout
    col1, col2 = st.columns([2, 3])

    with col1:
        # Model selection
        model = st.selectbox(
            "Select model",
            ["Amazon Nova Canvas"],
            key="model_selection"
        )

        # Text input for the prompt
        user_prompt = st.text_area(
            "Enter your image generation prompt:",
            value="A beautiful sunset over mountains",
            height=100,
            key="prompt_input"
        )

        # Advanced options in an expander
        with st.expander("Advanced Options"):
            image_size = st.select_slider(
                "Image Size",
                options=[512, 768, 1024],
                value=1024,
                key="image_size"
            )
            cfg_scale = st.slider(
                "CFG Scale",
                min_value=1.0,
                max_value=15.0,
                value=8.0,
                step=0.5,
                key="cfg_scale"
            )

        # Button to generate image
        generate_button = st.button(
            "Generate Image",
            type="primary",
            key="generate_button"
        )

    with col2:
        if generate_button:
            if not user_prompt.strip():
                st.warning("Please enter a prompt first.")
            else:
                with st.spinner("🎨 Generating your masterpiece..."):
                    base64_image = generate_image_nova(
                        text=user_prompt,
                        height=image_size,
                        width=image_size,
                        cfg_scale=cfg_scale
                    )
                    
                    if base64_image:
                        try:
                            image = base64_to_image(base64_image)
                            st.image(
                                image,
                                caption=f"Generated Image: {user_prompt[:50]}...",
                                use_column_width=True
                            )
                            
                            # Add download button for the generated image
                            st.download_button(
                                label="Download Image",
                                data=image.getvalue(),
                                file_name="generated_image.png",
                                mime="image/png"
                            )
                        except ValueError as e:
                            st.error(f"Failed to process the generated image: {str(e)}")

if __name__ == "__main__":
    main()
