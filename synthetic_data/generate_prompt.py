import os
import sys
import asyncio
from openai import AzureOpenAI
import numpy as np

# Hard-coded path to the project root
build_project_path = "C:/fine-tuning-build-project/Gemini2.5pro"
synthetic_data_path = os.path.join(build_project_path, 'fine-tuning-build-project', 'synthetic_data')

if synthetic_data_path not in sys.path:
    sys.path.append(synthetic_data_path)

from gpt_parsing import parse_gpt_response
from llm_requests import generate_prompt, get_client, async_make_api_call

async def test_prompt_parsing():
    """Test the prompt generation and response parsing with a simple example."""
    print("Setting up client...")
    client = get_client()
    model_name = "gpt-4.5-preview"

    test_titles = ["Software Engineer", "Data Scientist"]
    test_prompt = generate_prompt(test_titles)
    
    print("\n===== Generated Prompt =====")
    for message in test_prompt:
        print(f"\n{message['role'].upper()}:")
        print(message['content'][:300] + "..." if len(message['content']) > 300 else message['content'])
    
    print("\n===== Sending to API =====")
    try:
        response = await async_make_api_call(client, model_name, test_prompt, perturbation_std=0.0)
        print("\n===== Raw API Response =====")
        print(response.choices[0].message.content)
        
        print("\n===== Parsing Response =====")
        parsed_output = parse_gpt_response(response.choices[0].message.content, 2, 5)
        
        if parsed_output is not None:
            print("\n===== Parsed Output =====")
            for i, (title, variations) in enumerate(zip(test_titles, parsed_output)):
                print(f"\nVariations for: {title}")
                print('-' * 30)
                for j, variation in enumerate(variations):
                    print(f"{j+1}. {variation}")
        else:
            print("\n❌ PARSING FAILED ❌")
            print("The response couldn't be parsed according to the expected format.")
            print("Check the raw response above and adjust the parsing or prompting.")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    print("Testing prompt generation and parsing...")
    asyncio.run(test_prompt_parsing()) 