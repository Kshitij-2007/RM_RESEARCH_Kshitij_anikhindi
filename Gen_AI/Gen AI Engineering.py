from google import genai
from pydantic import BaseModel as bm
from typing import Literal
from pydantic import ValidationError
import json
import time
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"

)
class schema(bm):
    topic:str
    answer:list[str]
    validity: Literal["high", "low", "medium"]

client = genai.Client()
input_user = input("You: ")
logging.info("user inout recieved")
# using prompt template
prompt=f""" you are a assistant and your job is to help user by providing him  proper step by step method  as a  numbered list for his request and end with a type of service 
question:{input_user}""" #f""" """" helps create a f string which is a strinfg that can be added easily
for attempt in range(3):
    logging.info("Attempt %d started", attempt)
    try:
        logging.info("Calling Gemini API")
        response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
        "response_mime_type":"application/json",
        "response_schema": schema

}    
)
        if not response.text:
            print("an empty response from the model")
            logging.debug(  )
        else:
            raw_json = json.loads(response.text)
            validated = schema(**raw_json)
            logging.info("Validating schema")
            print("Validated output:", validated)
            break
    except (json.JSONDecodeError,ValidationError,ValueError) as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        logging.error("JSON parsing failed on attempt %d: %s", attempt, e)
        time.sleep(2)
    
