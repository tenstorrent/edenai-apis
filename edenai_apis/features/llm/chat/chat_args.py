# def chat_arguments(provider_name: str):
#     return {
#         "model": "tenstorrent/Meta-Llama-3.1-70B-Instruct",
#         "messages": [
#             {"role": "system", "content": "Always reply like a pirate"},
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": "Describe this image please!"},
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
#                         },
#                     },
#                 ],
#             },
#         ],
#         "temperature": 1,
#         "max_tokens": 1000,
#     }

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

