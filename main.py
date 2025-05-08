from ExtractorAgent import batch_extract_triplets

from samplerAgent import process_paragraphs


batch_extract_triplets(
        input_file="test.txt",
        output_files=[
            "LlamaOutput.json",
            "QwenOutput.json",
            "GemmaOutput.json"
        ],
        models=[
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "google/gemma-3-12b-it"
        ]
    )



process_paragraphs(
        input_files=[
            {'model_name': 'Qwen', 'file_path': 'QwenOutput.json'},
            {'model_name': 'Llama', 'file_path': 'LlamaOutput.json'},
            {'model_name': 'Gemma', 'file_path': 'GemmaOutput.json'}
        ],
        output_file='consensus_output.json'
    )


