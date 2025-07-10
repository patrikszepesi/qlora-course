import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="SageMaker Mixtral-8x7B",
    page_icon="🤖",
    layout="wide"
)

# --- API Configuration ---
API_GATEWAY_URL = "https://ad7xmnrx00.execute-api.us-west-2.amazonaws.com/invoke-llm-endpoint"

# --- App UI ---
st.title("🤖 Mixtral-8x7B Inference with SageMaker")
st.markdown("Enter a prompt below to get a response from the model.")

# --- User Input ---
with st.form("prompt_form"):
    user_prompt = st.text_area("Enter your prompt:", height=150, key="user_prompt")
    submitted = st.form_submit_button("Generate Response")

# --- Logic to Call API and Display Response ---
if submitted and user_prompt:
    headers = {
        "Content-Type": "application/json"
    }
    # The payload for the Lambda function expects a 'prompt' key.
    payload = json.dumps({"prompt": user_prompt})

    with st.spinner("Waiting for the model's response..."):
        try:
            # Make the POST request to the API Gateway
            response = requests.post(API_GATEWAY_URL, data=payload, headers=headers)
            
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status() 

            # Parse the response JSON
            result = response.json()
            
            # Handle different response structures
            if isinstance(result, dict):
                # If result has a 'body' key (API Gateway format)
                if 'body' in result:
                    if isinstance(result['body'], str):
                        # Body is a string, parse it
                        response_body = json.loads(result['body'])
                    else:
                        # Body is already a dict
                        response_body = result['body']
                else:
                    # Direct response format
                    response_body = result
            else:
                # If result is a string, parse it
                response_body = json.loads(result)

            # Check for an error key from the Lambda function
            if "error" in response_body:
                st.error(f"An error occurred in the backend: {response_body['error']}")
            else:
                # If no error, display the generated text
                generated_text = response_body.get("generated_text", "No text was generated.")
                st.subheader("Model Response:")
                st.info(generated_text)

        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err}")
            st.error(f"Response content: {response.text}")
        except requests.exceptions.RequestException as req_err:
            st.error(f"A request error occurred: {req_err}")
        except json.JSONDecodeError as json_err:
            st.error(f"Failed to decode the JSON response from the API: {json_err}")
            st.error(f"Response content: {response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

elif submitted:
    st.warning("Please enter a prompt before submitting.")