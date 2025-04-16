from openai import OpenAI
import openai
import base64
import os

image_path = ''

client = OpenAI(
  base_url="",
  api_key="",
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image(image_path) 

response = client.chat.completions.create(
    model="gpt-4o", # "claude-3-5-sonnet-20241022" "gemini-2.0-flash"
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please assume the role of both a radiology and neurosurgery expert. I will input a brain CT slice with a window width of 200 HU and a window level of 20 HU. This slice may or may not contain intracranial hemorrhage, and if present, it may include one or more subtypes of hemorrhage. After reviewing the slice, please answer the following questions in English and Chinese: 1.Which part of the brain is depicted in this slice? 2.Is there any evidence of intracranial hemorrhage in this slice? Please answer with 'Yes' or 'No.' 3.If intracranial hemorrhage is present, please specify which hemorrhage subtypes are observed. The known subtypes of hemorrhage include: extradural hemorrhage (EDH), subdural hematoma (SDH), subarachnoid hemorrhage (SAH), intraparenchymal hemorrhage (IPH), and intraventricular hemorrhage (IVH). If no hemorrhage is present, please respond with 'None.' 4.If intracranial hemorrhage is present, please locate the hemorrhage and quantify the hemorrhage volume in each observed subtype. (Assume that one pixel in the image represents a hemorrhage volume of 1 unit.) If no hemorrhage is present, please respond with 'None.' 5.If intracranial hemorrhage is present, what is your treatment recommendation? Choose between conservative treatment or puncture therapy. If no hemorrhage is present, please respond with 'None.' 6.Are there any other abnormalities observed in this slice? If yes, please describe them. If no, please respond with 'None.'"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    max_tokens=1000,
    temperature=0.1,
)

print(response.choices[0].message.content)