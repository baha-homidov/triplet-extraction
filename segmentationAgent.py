import re
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
def segment_into_paragraphs(text):

    openai = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://api.deepinfra.com/v1/openai",
    )

    sentences = segment_by_sentence(text)
    if not sentences:
        return []

    paragraphs = []
    current_paragraph = [sentences[0]]


    prompt_template = """请分析以下农业病害文本的段落结构：
当前段落内容：{prev_paragraph}
需要判断的后续句子：{sentence}

请严格遵循以下规则：
1. 如果后续句子是新的章节标题（如“一、...”、“二、...”），回答“False”
2. 如果后续句子包含列表项（数字编号或符号开头），回答“False”
3. 如果后续句子与前文主题明显不同，回答“False”
4. 如果后续句子属于当前段落的技术细节延续，回答“True”

请只回答“True”或“False”，不要添加任何解释。"""

    def call_llm(prev_para, next_sentence):
        prompt = prompt_template.format(
            prev_paragraph=prev_para,
            sentence=next_sentence
        )
        response = openai.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "system", "content": "你是一个农业病害文本分析专家，需要准确判断段落结构。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1,
            stop=["\n"]
        )
        return response.choices[0].message.content.strip().lower()

    # Process sentences with context awareness
    for i, sentence in enumerate(sentences[1:]):
        prev_paragraph = ' '.join(current_paragraph)
        
        # First check for obvious structural patterns
        if is_section_header(sentence) or is_list_item(sentence):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = [sentence]
            continue
            
        result = call_llm(prev_paragraph, sentence)
        
        if result == 'true':
            current_paragraph.append(sentence)
        else:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = [sentence]

    paragraphs.append(' '.join(current_paragraph))
    return paragraphs


def is_section_header(sentence):
    return re.match(r'^[一二三四五六七八九十]、.+', sentence)

def is_list_item(sentence):
    return re.match(r'^(\d+\.|①|②|③|●|▪|-)', sentence.strip())


def segment_by_sentence(text):

    sentences = re.split(r'(?<=[!?。！？])|(?=\n[一二三四五六七八九十]、)', text)
    
    processed = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
            

        if re.match(r'^[一二三四五六七八九十]、', sent):
            processed.append(sent)
            continue
            

        sent = re.sub(r'(\d+)([a-zA-Z%])', r'\1\2', sent)  # Keep units with numbers
        processed.append(sent)
    
    return [s for s in processed if s]


with open('test.txt', 'r') as file:
    text = file.read()
paragraphs = segment_into_paragraphs(text)

with open('result.txt', 'w') as file:
    for i, para in enumerate(paragraphs):
        file.write(f"Paragraph {i+1}: {para}\n\n")
