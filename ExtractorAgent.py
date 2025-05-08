# main.py
from SegmentationAgent import segment_into_paragraphs
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize DeepInfra client
openai = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai",
)
def extract_triplets(text, model):

    """Agricultural disease-focused triplet extraction"""
    prompt = f"""作为农业病害专家，请从文本中专门提取与植物病害相关的三元组：

【核心关注】
1. 只提取与植物病害直接相关的以下关系：
   - 病原物与寄主植物的相互作用
   - 病害症状表现
   - 病害发展规律
   - 防治措施效果
   - 品种抗病性表现

【严格排除】
1. 非病害相关的农业技术
2. 一般栽培管理措施
3. 非病害相关的生理过程
4. 与病害无关的环境因素

【关系框架】
1. 病害基本关系：
   - [病原]→[侵染/危害]→[寄主部位]
   - [环境条件]→[影响]→[病害发展]
2. 症状表现：
   - [植物部位]→[表现]→[病征]
3. 防治措施：
   - [防治方法]→[控制]→[病害]
4. 品种特性：
   - [品种]→[抗性]→[病害]

【输出要求】
1. 必须严格限定在植物病害范畴
2. 格式：[["主语","谓语","宾语"], ...]
3. 只输出JSON数组
4. 谓语必须使用精准的病害动词：
   - 危害/侵染/导致/表现/防治/抗性等

【正确示例】
文本："稻瘟病侵染叶片导致褐斑，苯醚甲环唑可有效防治"
输出：[["稻瘟病", "侵染", "叶片"],
      ["叶片", "表现", "褐斑"],
      ["苯醚甲环唑", "防治", "稻瘟病"]]

【错误示例】
文本："合理灌溉促进水稻生长"
输出：[]  # 非病害相关

请从以下文本中严格提取病害相关三元组：
"{text}"
"""

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "您需要系统提取农业文本中的各类关系"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.35,
        max_tokens=2000,
        top_p=0.9
    )
    
    try:
        content = response.choices[0].message.content
        start = content.find('[')
        end = content.rfind(']') + 1
        return json.loads(content[start:end])
    except Exception as e:
        print(f"Error parsing: {str(e)}")
        return []
    


def batch_extract_triplets(input_file, output_files, models):
    """Process all models with consistent segmentation"""
    # Read and segment text once
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    
    print("\n=== Segmenting Text ===")
    paragraphs = segment_into_paragraphs(text)
    print(f"Segmented into {len(paragraphs)} paragraphs")
    
    # Process each model with the same paragraphs
    for model, output_file in zip(models, output_files):
        print(f"\n=== Processing {model} ===")
        
        results = []
        for i, para in enumerate(paragraphs):
            print(f"\nProcessing Paragraph {i+1}/{len(paragraphs)}")
            print(f"Text snippet: {para[:50]}...")
            
            # Extract triplets
            para_triplets = extract_triplets(para, model)
            
            results.append({
                "text": para,
                "triplets": para_triplets
            })
            
            print(f"Extracted {len(para_triplets)} triplets")
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(results)} paragraphs to {output_file}")

