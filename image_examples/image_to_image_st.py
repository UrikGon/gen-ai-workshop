import base64
import io
import json
import os
import boto3
import streamlit as st
from PIL import Image

REGION = "us-east-1"

# Define bedrock
@st.cache_resource
def get_bedrock_client():
    """Cache the Bedrock client to avoid recreation on each rerun"""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=REGION,
    )

def image_to_base64(img) -> str:
    """Convert a PIL Image or local image file path to a base64 string for Amazon Bedrock"""
    if isinstance(img, str):
        if os.path.isfile(img):
            print(f"Reading image from file: {img}")
            with open(img, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise FileNotFoundError(f"File {img} does not exist")
    elif isinstance(img, Image.Image):
        print("Converting PIL Image to base64 string")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Expected str (filename) or PIL Image. Got {type(img)}")

def base64_to_pil(base64_string):
    """
    Convert a base64 string to a PIL Image
    Args:
        base64_string: base64 string of image
    Returns:
        PIL.Image: The decoded image
    """
    try:
        imgdata = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(imgdata))
        return image
    except Exception as e:
        st.error(f"Error converting base64 to image: {str(e)}")
        return None

def nova_update_image(change_prompt, init_image_b64, similarity_strength=0.7):
    """
    Use Bedrock API to generate an Image variation using Nova
    Args:
        change_prompt: Text prompt for image modification
        init_image_b64: Base64 string of initial image
        similarity_strength: How similar the output should be to input (0.2 to 1.0)
    Returns:
        str: Base64 string of generated image
    """
    try:
        body = {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "text": change_prompt,
                "images": [init_image_b64],
                "similarityStrength": similarity_strength,
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 512,
                "width": 512,
                "cfgScale": 8.0,
            },
        }

        bedrock_runtime = get_bedrock_client()
        response = bedrock_runtime.invoke_model(
            body=json.dumps(body),
            modelId="amazon.nova-canvas-v1:0",
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        return response_body.get("images")[0]
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def update_image_pipeline(user_image, change_prompt, model, similarity_strength=0.7):
    """
    Complete pipeline for updating an image
    Args:
        user_image: PIL Image object
        change_prompt: Text prompt for modification
        model: Model name to use
        similarity_strength: Similarity strength for Nova
    Returns:
        PIL.Image: Modified image
    """
    try:
        init_image_b64 = image_to_base64(user_image)
        
        if model == "Amazon Nova":
            updated_image_b64 = nova_update_image(
                change_prompt, 
                init_image_b64, 
                similarity_strength
            )
            if updated_image_b64:
                updated_image = base64_to_pil(updated_image_b64)
                return updated_image
        return None
    except Exception as e:
        st.error(f"Error in image pipeline: {str(e)}")
        return None

def main():
    st.title("Building with Bedrock")
    st.subheader("Image Generation Demo - Image to Image")

    # Create two columns for layout
    col1, col2 = st.columns(2)

    with col1:
        model = st.selectbox("Select model", ["Amazon Nova"])
        user_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        change_prompt = st.text_area(
            "Enter a prompt to change the image",
            placeholder="Describe how you want to modify the image..."
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            similarity_strength = st.slider(
                "Similarity Strength",
                min_value=0.2,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher values keep the image more similar to the original"
            )

    # Show user image and generate button in first column
    if user_image is not None:
        user_image = Image.open(user_image)
        col1.image(user_image, caption="Original Image")
        
        # Button to generate new image
        if col1.button("Update Image"):
            if not change_prompt:
                st.warning("Please enter a prompt to modify the image.")
            else:
                with st.spinner("Generating your image..."):
                    new_image = update_image_pipeline(
                        user_image,
                        change_prompt,
                        model,
                        similarity_strength
                    )
                    if new_image:
                        col2.image(new_image, caption="Generated Image")
                        
                        # Add download button
                        buf = io.BytesIO()
                        new_image.save(buf, format="PNG")
                        col2.download_button(
                            label="Download Image",
                            data=buf.getvalue(),
                            file_name="generated_image.png",
                            mime="image/png"
                        )
    else:
        col2.write("No image uploaded")

if __name__ == "__main__":
    main()
