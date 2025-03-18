def chat_arguments(provider_name: str):
    return {
        "model": "tenstorrent/Meta-Llama-3.1-70B-Instruct",
        "messages": [
            {"role": "system", "content": "Always reply like a pirate"},
            {
                "role": "user",
                "content": "Describe a scenic boardwalk by a lake in Wisconsin, as if you’re looking at an image. (No actual image attached.)"
            },
        ],
        "temperature": 1,
        "max_tokens": 1000,
        # "stream": True,
    }
