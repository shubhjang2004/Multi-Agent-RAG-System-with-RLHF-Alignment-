# Notebook 10: Constitutional AI Layer
# Cell 1
!pip install transformers torch google-generativeai -q

# Cell 2
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import google.generativeai as genai
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

PPO_MODEL_DIR = Path("models/ppo_model/final")
OUTPUT_DIR = Path("models/constitutional_ai")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cell 3
# Constitutional principles
CONSTITUTION = {
    "principles": [
        {
            "name": "Accuracy",
            "description": "Responses must be factually accurate and cite sources",
            "critique_request": "Identify any factual inaccuracies or unsupported claims"
        },
        {
            "name": "Citation",
            "description": "All claims must reference specific papers or sources",
            "critique_request": "Check if claims are properly cited with paper references"
        },
        {
            "name": "Uncertainty",
            "description": "Express uncertainty when information is ambiguous",
            "critique_request": "Identify claims that should express more uncertainty"
        },
        {
            "name": "Relevance",
            "description": "Responses must directly address the question",
            "critique_request": "Check if the response stays on topic and answers the question"
        },
        {
            "name": "Completeness",
            "description": "Cover all important aspects of the question",
            "critique_request": "Identify any important aspects that were not addressed"
        }
    ]
}

print("Constitutional principles loaded:")
for p in CONSTITUTION['principles']:
    print(f"  - {p['name']}: {p['description']}")

# Cell 4
# Load model
tokenizer = AutoTokenizer.from_pretrained(PPO_MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    PPO_MODEL_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model = model.to(DEVICE)
model.eval()

print("Model loaded")

# Cell 5
# Configure Gemini for critique (free tier)
# Note: You'll need to get a free API key from https://makersuite.google.com/app/apikey
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')

if GEMINI_API_KEY != 'YOUR_API_KEY_HERE':
    genai.configure(api_key=GEMINI_API_KEY)
    critique_model = genai.GenerativeModel('gemini-pro')
    USE_GEMINI = True
    print("Gemini configured for critique")
else:
    USE_GEMINI = False
    print("Warning: Gemini API key not set. Using rule-based critique.")

# Cell 6
# Critique function
def critique_response(response, principle, use_gemini=USE_GEMINI):
    """Critique a response based on a constitutional principle"""
    
    if use_gemini:
        prompt = f"""You are a research paper critic. Analyze this response:

Response: {response}

Constitutional Principle: {principle['description']}
Task: {principle['critique_request']}

Provide a brief critique (2-3 sentences) identifying issues:"""
        
        try:
            result = critique_model.generate_content(prompt)
            return result.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            use_gemini = False
    
    # Fallback: rule-based critique
    if not use_gemini:
        issues = []
        
        if principle['name'] == 'Citation':
            if '[' not in response and 'arxiv' not in response.lower():
                issues.append("No paper citations found")
        
        if principle['name'] == 'Uncertainty':
            uncertain_words = ['might', 'could', 'possibly', 'likely', 'may']
            if not any(word in response.lower() for word in uncertain_words):
                if '?' in response or 'uncertain' in response.lower():
                    pass
                else:
                    issues.append("Consider expressing more uncertainty")
        
        if principle['name'] == 'Completeness':
            if len(response.split()) < 30:
                issues.append("Response seems too brief")
        
        return " ".join(issues) if issues else "No major issues found"

# Cell 7
# Revision function
def revise_response(original_response, critiques, query):
    """Revise response based on critiques"""
    
    if not USE_GEMINI:
        # Simple rule-based revision
        revised = original_response
        
        # Add uncertainty markers
        if "express more uncertainty" in str(critiques).lower():
            revised = "Based on available research, " + revised
            revised = revised.replace("is ", "appears to be ")
            revised = revised.replace("are ", "seem to be ")
        
        # Add citation placeholder
        if "no paper citations" in str(critiques).lower():
            revised += " [Citation needed: See relevant papers in the knowledge base]"
        
        return revised
    
    # Gemini-based revision
    critique_text = "\n".join([f"- {c}" for c in critiques])
    
    prompt = f"""You are revising a research assistant's response.

Original Query: {query}
Original Response: {original_response}

Issues to fix:
{critique_text}

Provide an improved response that addresses these issues while maintaining accuracy:"""
    
    try:
        result = critique_model.generate_content(prompt)
        return result.text
    except Exception as e:
        print(f"Revision error: {e}")
        return original_response

# Cell 8
# Constitutional AI pipeline
def constitutional_ai_generate(query, model, tokenizer, constitution):
    """Generate response with constitutional AI safety"""
    
    # 1. Initial generation
    inputs = tokenizer(query, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    initial_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    initial_response = initial_response.replace(query, "").strip()
    
    print(f"\nInitial response: {initial_response}\n")
    
    # 2. Critique against each principle
    critiques = []
    for principle in constitution['principles']:
        critique = critique_response(initial_response, principle)
        if critique and "no major issues" not in critique.lower():
            critiques.append(f"{principle['name']}: {critique}")
            print(f"Critique ({principle['name']}): {critique}")
    
    if not critiques:
        print("No issues found - response approved")
        return initial_response
    
    # 3. Revise based on critiques
    print("\nRevising response...")
    revised_response = revise_response(initial_response, critiques, query)
    print(f"\nRevised response: {revised_response}")
    
    return revised_response

# Cell 9
# Test constitutional AI
test_queries = [
    "What is RLHF?",
    "Explain the transformer architecture",
    "How do diffusion models work?",
    "What are the latest advances in computer vision?"
]

results = []

print("\n" + "="*60)
print("CONSTITUTIONAL AI TESTING")
print("="*60)

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    safe_response = constitutional_ai_generate(
        query, 
        model, 
        tokenizer, 
        CONSTITUTION
    )
    
    results.append({
        'query': query,
        'response': safe_response
    })

# Cell 10
# Save results
with open(OUTPUT_DIR / 'constitutional_ai_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save constitution
with open(OUTPUT_DIR / 'constitution.json', 'w') as f:
    json.dump(CONSTITUTION, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}")
print("\nNotebook 10 Complete!")