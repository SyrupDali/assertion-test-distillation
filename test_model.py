import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_dir = "distilled_codet5_student"  # your fine-tuned student model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔍 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
model.eval()

# Sample input from your dataset
input_text = """FOCAL CODE:
/*
 * Licensed to the Apache Software Foundation (ASF)...
 */

TEST METHOD:
@Test
public void testNaturalNumber() throws Exception {
    Object ret = reader.read("123");
}
"""

inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
print("\n=== Generating...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=128)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== 📢 GENERATED ASSERTION(S) ===")
print(decoded if decoded.strip() else "[Empty output]")
