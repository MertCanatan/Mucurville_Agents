import requests

def prioritize_tasks(task_list):
    # NOTE: Replace "YOUR_API_KEY" with your actual GMI API key.
    url = "https://api.gmicloud.ai/v1/completions"
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    }

    prompt = build_prompt(task_list)

    data = {
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()

    return parse_response(result['choices'][0]['text'])
