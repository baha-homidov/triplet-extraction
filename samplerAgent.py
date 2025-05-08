from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Any

load_dotenv()

class TripletConsensusGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self.model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        print("\n=== Initializing Triplet Consensus Generator ===")
        print(f"Using LLM model: {self.model}")
        
    def generate_consensus(self, text: str, model_outputs: List[Dict[str, Any]]) -> List[List[str]]:
        """Generate consensus triplets from multiple model outputs"""
        print(f"\n[Processing Paragraph] Text length: {len(text)} characters")
        print(f"Received inputs from {len(model_outputs)} models")

        prompt = f"""作为农业病害专家，请根据以下文本和多个模型提取的三元组，生成最终准确的三元组列表：

【原始文本】
{text}

【模型提取结果】
{self._format_model_outputs(model_outputs)}

【任务要求】
1. 综合分析所有模型的提取结果
2. 保留准确反映文本信息的三元组
3. 合并相同含义但表述不同的三元组
4. 去除错误或冗余的三元组
5. 格式要求: [["主语","谓语","宾语"], ...]

请直接输出最终的三元组JSON数组，不要包含任何解释："""

        print("\nSending request to LLM for consensus generation...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一位专业的农业病害知识提取专家"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        try:
            print("Received LLM response, parsing results...")
            content = response.choices[0].message.content
            start = content.find('[')
            end = content.rfind(']') + 1
            result = json.loads(content[start:end])
            print(f"Consensus generation successful. Found {len(result)} triplets")
            return result
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return []

    def _format_model_outputs(self, model_outputs: List[Dict[str, Any]]) -> str:
        """Format model outputs for the prompt"""
        return "\n".join(
            f"{i+1}. {output['model_name']}模型: {output['triplets'] or '未提取到三元组'}"
            for i, output in enumerate(model_outputs))
        

def process_paragraphs(input_files: List[Dict[str, str]], output_file: str):
    """
    Process multiple input files containing model outputs and generate consensus
    """
    print("\n=== Starting Processing Pipeline ===")
    print(f"Processing {len(input_files)} model outputs")
    
    generator = TripletConsensusGenerator()
    results = []
    
    # Load all model data
    print("\n[Stage 1/3] Loading model outputs...")
    all_model_data = []
    for input_file in input_files:
        try:
            print(f"Loading {input_file['model_name']} data from {input_file['file_path']}")
            with open(input_file['file_path'], 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_model_data.append({
                    'model_name': input_file['model_name'],
                    'data': data
                })
                print(f"Successfully loaded {len(data)} paragraphs from {input_file['model_name']}")
        except Exception as e:
            print(f"Error loading {input_file['model_name']} data: {str(e)}")
            raise

    # Validate paragraph alignment
    print("\n[Stage 1.5/3] Validating paragraph alignment...")
    paragraph_counts = [len(m['data']) for m in all_model_data]
    if len(set(paragraph_counts)) != 1:
        print(f"Error: Models have different paragraph counts: {paragraph_counts}")
        raise ValueError("All model outputs must contain the same number of paragraphs")

    # Verify text consistency across models
    base_texts = [p['text'] for p in all_model_data[0]['data']]
    for model_data in all_model_data[1:]:
        model_texts = [p['text'] for p in model_data['data']]
        if model_texts != base_texts:
            print("Error: Paragraph texts are not aligned across models")
            raise ValueError("Paragraph content mismatch between models")

    num_paragraphs = len(all_model_data[0]['data'])
    print(f"\n[Stage 2/3] Processing {num_paragraphs} paragraphs...")
    
    for para_idx in range(num_paragraphs):
        print(f"\nProcessing paragraph {para_idx+1}/{num_paragraphs}")
        text = all_model_data[0]['data'][para_idx]['text']
        print(f"Text snippet: {text[:50]}...")

        # Collect model outputs with validation
        model_outputs = []
        for model_data in all_model_data:
            try:
                para_data = model_data['data'][para_idx]
                triplets = para_data.get('triplets', [])
                if para_data['text'] != text:
                    print(f"Text mismatch in {model_data['model_name']} at paragraph {para_idx+1}")
                    raise ValueError("Paragraph text mismatch")
            except IndexError:
                print(f"Critical error: {model_data['model_name']} missing paragraph {para_idx+1}")
                raise

            model_outputs.append({
                'model_name': model_data['model_name'],
                'triplets': triplets
            })
            print(f"{model_data['model_name']} extracted {len(triplets)} triplets")

        # Generate consensus
        consensus = generator.generate_consensus(text, model_outputs)
        
        results.append({
            "text": text,
            "consensus_triplets": consensus,
            "source_models": {m['model_name']: m['triplets'] for m in model_outputs}
        })

    # Save results
    print("\n[Stage 3/3] Saving final results...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n=== Processing Complete ===")
    print(f"Saved consensus results to: {output_file}")
    print(f"Total paragraphs processed: {len(results)}")
    total_triplets = sum(len(item['consensus_triplets']) for item in results)
    print(f"Total consensus triplets generated: {total_triplets}")

# Example usage
if __name__ == "__main__":
    process_paragraphs(
        input_files=[
            {'model_name': 'Qwen', 'file_path': 'qwen_output.json'},
            {'model_name': 'Llama', 'file_path': 'llama_output.json'},
            {'model_name': 'Gemma', 'file_path': 'gemma_output.json'}
        ],
        output_file='consensus_output.json'
    )
