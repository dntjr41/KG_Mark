from transformers import AutoModelForCausalLM, AutoTokenizer
from config import HUGGINGFACE_TOKEN

model_name = "meta-llama/Meta-Llama-3-8B"
# token = HUGGINGFACE_TOKEN

# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:2", torch_dtype=torch.float8, use_auth_token=token)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

input_text = "The capital of France is"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))