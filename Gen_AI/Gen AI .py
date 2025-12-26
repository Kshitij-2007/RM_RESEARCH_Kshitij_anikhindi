from openai import OpenAI
client = OpenAI(api_key="API KEY")
user_input = input("You: ")
response = client.chat.completions.create(model="gpt-4o-mini",messages=[{"role": "user", "content": user_input}])
print("LLM:", response.choices[0].message.content)
