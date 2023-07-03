# Imports
import openai
import json
import argparse
from typing import Dict, List, Tuple
import os
import requests

# Notebook Imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path



def generate_image(base_images_number: int, labels:str, output_folder: str, variation_per_image: int):
    for option in labels:
        for i in range(base_images_number):
            response = openai.Image.create(
                prompt=option["prompt"],
                n=1,
                size="512x512",
            )
            try:
                img = response["data"][0]["url"]
                with open(f'{output_folder}/{option["label"]}.{img.split("/")[-1]}.png', 'wb+') as f:
                    f.write(requests.get(img).content)
                response2 = openai.Image.create_variation(
                    image=requests.get(img).content,
                    n=variation_per_image,
                    size="512x512"
                )
            except Exception as e:
                print(e)
            for img in response2['data']:
                try:
                    with open(f'{output_folder}/{option["label"]}.{img["url"].split("/")[-1]}.png', 'wb') as f:
                        f.write(requests.get(img["url"]).content)
                except Exception as e:
                    print(e)



def get_completion(prompt, template, model="gpt-3.5-turbo")-> List[Dict[int, str]]:
    """
    params:
        prompt (str):
        template (str):
        return (List[Dict[int, str]])
    """

    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{template}"}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    result = response.choices[0].message["content"].splitlines()
    result = [{'label': int(item.split('. ')[0]), 'prompt': item.split('. ')[1]} for item in result]

    return result


def check_affordance_func(objects: List, query_template: str, task: str):
    """
    params:
        objects (List):
        query_template (str):
        return (Dict[int, str])
    """
    
    # Format the objects as a comma-separated string
    objects_string = ", ".join(objects)

    # Replace the placeholder in the query template with the objects string
    query = query_template.format(objects=objects_string)

    # Generate responses using GPT-3
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=query + task,
        max_tokens=300,  # Adjust based on the desired response length
        temperature=0.2  # Adjust to control the randomness of the responses
    )

    # Extract the generated response from the API response
    generated_text = response.choices[0].text.strip()
    return generated_text



def check_affordance(objects: List, query_template: str, task: str):
    """
    params:
        objects (List):
        query_template (str):
        return (Dict[int, str])
    """
    
    # Format the objects as a comma-separated string
    objects_string = ", ".join(objects)

    # Replace the placeholder in the query template with the objects string
    query = query_template.format(objects=objects_string)

    # Generate responses using GPT-3
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=query + task,
        max_tokens=150,  # Adjust based on the desired response length
        temperature=0.3  # Adjust to control the randomness of the responses
    )

    # Extract the generated response from the API response
    generated_text = response.choices[0].text.strip()
    
    data_dict = {}

    lines = generated_text.split('\n')

    for line in lines:
        line = line.strip()
        if line:
            parts = line.split('. ')
            key = int(parts[0])
            value = parts[1]
            data_dict[key] = value
    return data_dict
    
    # prompt_json = {
    # "prompt": query,
    # "generated_prompt": generated_text}

    # # Convert the JSON object to a string
    # prompt_json_string = json.dumps(prompt_json)
    # return prompt_json_string

def process_dalle_images(response, filename, image_dir):
    # save the images
    urls = [datum["url"] for datum in response["data"]]  # extract URLs
    images = [requests.get(url).content for url in urls]  # download images
    image_names = [f"{filename}_{i + 1}.png" for i in range(len(images))]  # create names
    filepaths = [os.path.join(image_dir, name) for name in image_names]  # create filepaths
    for image, filepath in zip(images, filepaths):  # loop through the variations
        with open(filepath, "wb") as image_file:  # open the file
            image_file.write(image)  # write the image to the file

    return filepaths

def edit_image(image_path:str, mask_dir:str, variation_image:int, prompt:str):
    edit_response = openai.Image.create_edit(
        image=open(image_path, "rb"),  # from the generation section
        mask=open(mask_dir, "rb"), # from right above
        prompt=prompt,  # provide a prompt to fill the space
        n=variation_image,
        size="512x512",
        response_format="url",
    )
    return edit_response



def affordance_func(template, model="gpt-3.5-turbo-0613"):
    """
    params:
        prompt (str):
        template (str):
        return (List[Dict[int, str]])
    """

    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{template}"}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0 # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def extract_actions(text, paths_list):
    def map_paths_to_items(actions, paths_list):
        # Create a dictionary that associates each item name with its PosixPath
        path_dict = {}
        for path in paths_list:
            item_name = path.stem  # Extract the item name from the path (e.g., 'tomato' from 'tomato.png')
            path_dict[item_name] = str(path)

        # Update the list of action dictionaries
        for action in actions:
            item = action['Performed on'].lower()  # Convert to lowercase to match dictionary keys
            path = path_dict.get(item)  # Get the corresponding path, if it exists
            if path is not None:
                action['Performed on'] = path  # Replace 'Performed on' value with path

        return actions

    actions = text.split('###')[1:]  # Ignore the first split result because it's empty

    # Parse each action into a dictionary
    actions_dict = []
    for action in actions:
        lines = action.split('\n')
        action_dict = {}
        for line in lines:
            if line.startswith('Performed on:'):
                action_dict['Performed on'] = line.split(': ')[1]
            elif line.startswith('Effect:'):
                action_dict['Effect'] = line.split(': ')[1]
        actions_dict.append(action_dict)

    # Map the paths to the actions
    updated_actions = map_paths_to_items(actions_dict, paths_list)

    return updated_actions