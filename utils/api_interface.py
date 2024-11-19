from openai import OpenAI
import google.generativeai as genai
from utils.config import OPENAI_API_KEY, GEMINI_API_KEY, MODEL_NAME

client = OpenAI(api_key=OPENAI_API_KEY)

def get_openai_response(prompt, model=MODEL_NAME, temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=temperature
    )
    return response.choices[0].message.content

def get_gemini_response(prompt, temperature=0.7):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        if response.text:
            return response.text
        else:
            return "Response blocked due to content safety filters."
    except Exception as e:
        return f"Error generating response: {str(e)}"

def get_dual_responses(prompt):
    """Get responses from both OpenAI and Gemini APIs."""
    openai_response = get_openai_response(prompt)
    gemini_response = get_gemini_response(prompt)
    return {
        'openai': openai_response,
        'gemini': gemini_response
    } 