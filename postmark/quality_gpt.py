import openai
import re
import json
import torch
import statistics
from config import OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 1. JSONL 파일 읽기 함수
def load_jsonl(file_path):
    """JSONL 파일을 읽어 리스트로 반환"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def evaluate_text_with_gpt(original_text, watermarked_text, metric, model_name="gpt-3.5-turbo"):
    """
    주어진 원본 텍스트와 워터마크 텍스트의 metric(평가 기준: 'relevance', 'coherence', 'interestingness')을 비교 평가.
    모델은 model_name 인자를 통해 선택 가능 (예: "gpt-3.5-turbo" 또는 "gpt-4").
    """
    if metric == "relevance":
        prompt = (
            f"Compare the following two texts and evaluate their relevance to the intended meaning on a scale from 0 to 10.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide your answer in the format: [Original: X, Watermarked: Y]."
        )
    elif metric == "coherence":
        prompt = (
            f"Compare the coherence of the following two texts on a scale from 0 to 10.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide your answer in the format: [Original: X, Watermarked: Y]."
        )
    elif metric == "interestingness":
        prompt = (
            f"Compare how interesting the following two texts are on a scale from 0 to 10.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide your answer in the format: [Original: X, Watermarked: Y]."
        )
    else:
        raise ValueError("Invalid metric provided.")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert text evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=50
    )
        
    # 응답 텍스트 추출
    result_text = response.choices[0].message.content.strip()
    # 예시: "[Original: 7, Watermarked: 8]"
    try:
        match = re.search(r"\[Original:\s*(\d+),\s*Watermarked:\s*(\d+)\]", result_text)
        if match:
            orig_score = int(match.group(1))
            water_score = int(match.group(2))
            return orig_score, water_score  # 정규화 제거
    except Exception as e:
        print(f"Score extraction failed: {e}")
    return 0, 0

# 3. 메인 함수
def main():
    # 파일 경로
    file_path = "/home/wooseok/PostMark-main/outputs/test/test_only_change.jsonl"
    
    # 데이터 로드
    print("JSONL 파일을 로드 중...")
    data = load_jsonl(file_path)
    
    # 결과 저장용 딕셔너리
    insert_relevance_llm_result = {"original_better": 0, "llm_better": 0, "tie": 0}
    insert_relevance_ner_result = {"original_better": 0, "ner_better": 0, "tie": 0}
    insert_coherence_llm_result = {"original_better": 0, "llm_better": 0, "tie": 0}
    insert_coherence_ner_result = {"original_better": 0, "ner_better": 0, "tie": 0}
    insert_interestingness_llm_result = {"original_better": 0, "llm_better": 0, "tie": 0}
    insert_interestingness_ner_result = {"original_better": 0, "ner_better": 0, "tie": 0}
    
    watermark_relevance_llm_result = {"original_better": 0, "llm_better": 0, "tie": 0}
    watermark_relevance_ner_result = {"original_better": 0, "ner_better": 0, "tie": 0}
    watermark_coherence_llm_result = {"original_better": 0, "llm_better": 0, "tie": 0}
    watermark_coherence_ner_result = {"original_better": 0, "ner_better": 0, "tie": 0}
    watermark_interestingness_llm_result = {"original_better": 0, "llm_better": 0, "tie": 0}
    watermark_interestingness_ner_result = {"original_better": 0, "ner_better": 0, "tie": 0}
    
    total_items = len(data)
    
    for idx, item in enumerate(data):
        print(f"Processing item {idx + 1} of {total_items}...")
        original_text = item.get("original_text")
        insert_llm_text = item.get("llm_insertion_only")
        watermark_llm_text = item.get("llm_watermarked_only")
        insert_ner_text = item.get("ner_insertion_only")
        watermark_ner_text = item.get("ner_watermarked_only")
        
        # Insert
        insert_relevance_original_llm, insert_relevance_llm = evaluate_text_with_gpt(original_text, insert_llm_text, "relevance")
        insert_relevance_original_ner, insert_relevance_ner = evaluate_text_with_gpt(original_text, insert_ner_text, "relevance")

        insert_coherence_original_llm, insert_coherence_llm = evaluate_text_with_gpt(original_text, insert_llm_text, "coherence")
        insert_coherence_original_ner, insert_coherence_ner = evaluate_text_with_gpt(original_text, insert_ner_text, "coherence")
        
        insert_interestingness_original_llm, insert_interestingness_llm = evaluate_text_with_gpt(original_text, insert_llm_text, "interestingness")
        insert_interestingness_original_ner, insert_interestingness_ner = evaluate_text_with_gpt(original_text, insert_ner_text, "interestingness")
        
        # Watermark
        watermark_relevance_original_llm, watermark_relevance_llm = evaluate_text_with_gpt(original_text, watermark_llm_text, "relevance")
        watermark_relevance_original_ner, watermark_relevance_ner = evaluate_text_with_gpt(original_text, watermark_ner_text, "relevance")
        
        watermark_coherence_original_llm, watermark_coherence_llm = evaluate_text_with_gpt(original_text, watermark_llm_text, "coherence")
        watermark_coherence_original_ner, watermark_coherence_ner = evaluate_text_with_gpt(original_text, watermark_ner_text, "coherence")
        
        watermark_interestingness_original_llm, watermark_interestingness_llm = evaluate_text_with_gpt(original_text, watermark_llm_text, "interestingness")
        watermark_interestingness_original_ner, watermark_interestingness_ner = evaluate_text_with_gpt(original_text, watermark_ner_text, "interestingness")
        
        # Insert
        # Relevance
        if insert_relevance_original_llm > insert_relevance_llm:
            insert_relevance_llm_result["original_better"] += 1
        elif insert_relevance_original_llm < insert_relevance_llm:
            insert_relevance_llm_result["llm_better"] += 1
        else:
            insert_relevance_llm_result["tie"] += 1
        
        if insert_relevance_original_ner > insert_relevance_ner:
            insert_relevance_ner_result["original_better"] += 1
        elif insert_relevance_original_ner < insert_relevance_ner:
            insert_relevance_ner_result["ner_better"] += 1
        else:
            insert_relevance_ner_result["tie"] += 1
            
        # Coherence
        if insert_coherence_original_llm > insert_coherence_llm:
            insert_coherence_llm_result["original_better"] += 1
        elif insert_coherence_original_llm < insert_coherence_llm:
            insert_coherence_llm_result["llm_better"] += 1
        else:
            insert_coherence_llm_result["tie"] += 1
            
        if insert_coherence_original_ner > insert_coherence_ner:
            insert_coherence_ner_result["original_better"] += 1
        elif insert_coherence_original_ner < insert_coherence_ner:
            insert_coherence_ner_result["ner_better"] += 1
        else:
            insert_coherence_ner_result["tie"] += 1

        # Interestingness
        if insert_interestingness_original_llm > insert_interestingness_llm:
            insert_interestingness_llm_result["original_better"] += 1
        elif insert_interestingness_original_llm < insert_interestingness_llm:
            insert_interestingness_llm_result["llm_better"] += 1
        else:
            insert_interestingness_llm_result["tie"] += 1
            
        if insert_interestingness_original_ner > insert_interestingness_ner:
            insert_interestingness_ner_result["original_better"] += 1
        elif insert_interestingness_original_ner < insert_interestingness_ner:
            insert_interestingness_ner_result["ner_better"] += 1
        else:
            insert_interestingness_ner_result["tie"] += 1
            
        # Watermark
        # Relevance
        if watermark_relevance_original_llm > watermark_relevance_llm:
            watermark_relevance_llm_result["original_better"] += 1
        elif watermark_relevance_original_llm < watermark_relevance_llm:
            watermark_relevance_llm_result["llm_better"] += 1
        else:
            watermark_relevance_llm_result["tie"] += 1
        
        if watermark_relevance_original_ner > watermark_relevance_ner:
            watermark_relevance_ner_result["original_better"] += 1
        elif watermark_relevance_original_ner < watermark_relevance_ner:
            watermark_relevance_ner_result["ner_better"] += 1
        else:
            watermark_relevance_ner_result["tie"] += 1
            
        # Coherence
        if watermark_coherence_original_llm > watermark_coherence_llm:
            watermark_coherence_llm_result["original_better"] += 1
        elif watermark_coherence_original_llm < watermark_coherence_llm:
            watermark_coherence_llm_result["llm_better"] += 1
        else:
            watermark_coherence_llm_result["tie"] += 1
            
        if watermark_coherence_original_ner > watermark_coherence_ner:
            watermark_coherence_ner_result["original_better"] += 1
        elif watermark_coherence_original_ner < watermark_coherence_ner:
            watermark_coherence_ner_result["ner_better"] += 1
        else:
            watermark_coherence_ner_result["tie"] += 1
            
        # Interestingness
        if watermark_interestingness_original_llm > watermark_interestingness_llm:
            watermark_interestingness_llm_result["original_better"] += 1
        elif watermark_interestingness_original_llm < watermark_interestingness_llm:
            watermark_interestingness_llm_result["llm_better"] += 1
        else:
            watermark_interestingness_llm_result["tie"] += 1
            
        if watermark_interestingness_original_ner > watermark_interestingness_ner:
            watermark_interestingness_ner_result["original_better"] += 1
        elif watermark_interestingness_original_ner < watermark_interestingness_ner:
            watermark_interestingness_ner_result["ner_better"] += 1
        else:
            watermark_interestingness_ner_result["tie"] += 1
            
    # 비율 계산 및 출력
    print("\n=== 결과 비율 (100% 기준) ===")
    print("Insert Relevance:")
    print(f"LLM Original Better: {insert_relevance_llm_result['original_better'] / total_items * 100:.2f}%")
    print(f"LLM Better: {insert_relevance_llm_result['llm_better'] / total_items * 100:.2f}%")
    print(f"LLM Tie: {insert_relevance_llm_result['tie'] / total_items * 100:.2f}%")
    
    print(f"NER Original Better: {insert_relevance_ner_result['original_better'] / total_items * 100:.2f}%")
    print(f"NER Better: {insert_relevance_ner_result['ner_better'] / total_items * 100:.2f}%")
    print(f"NER Tie: {insert_relevance_ner_result['tie'] / total_items * 100:.2f}%")
    
    print("\nInsert Coherence:")
    print(f"LLM Original Better: {insert_coherence_llm_result['original_better'] / total_items * 100:.2f}%")
    print(f"LLM Better: {insert_coherence_llm_result['llm_better'] / total_items * 100:.2f}%")
    print(f"LLM Tie: {insert_coherence_llm_result['tie'] / total_items * 100:.2f}%")
    
    print(f"NER Original Better: {insert_coherence_ner_result['original_better'] / total_items * 100:.2f}%")
    print(f"NER Better: {insert_coherence_ner_result['ner_better'] / total_items * 100:.2f}%")
    print(f"NER Tie: {insert_coherence_ner_result['tie'] / total_items * 100:.2f}%")
    
    print("\nInsert Interestingness:")
    print(f"LLM Original Better: {insert_interestingness_llm_result['original_better'] / total_items * 100:.2f}%")
    print(f"LLM Better: {insert_interestingness_llm_result['llm_better'] / total_items * 100:.2f}%")
    print(f"LLM Tie: {insert_interestingness_llm_result['tie'] / total_items * 100:.2f}%")
    
    print(f"NER Original Better: {insert_interestingness_ner_result['original_better'] / total_items * 100:.2f}%")
    print(f"NER Better: {insert_interestingness_ner_result['ner_better'] / total_items * 100:.2f}%")
    print(f"NER Tie: {insert_interestingness_ner_result['tie'] / total_items * 100:.2f}%")
    
    print("\nWatermark Relevance:")
    print(f"LLM Original Better: {watermark_relevance_llm_result['original_better'] / total_items * 100:.2f}%")
    print(f"LLM Better: {watermark_relevance_llm_result['llm_better'] / total_items * 100:.2f}%")
    print(f"LLM Tie: {watermark_relevance_llm_result['tie'] / total_items * 100:.2f}%")
    
    print(f"NER Original Better: {watermark_relevance_ner_result['original_better'] / total_items * 100:.2f}%")
    print(f"NER Better: {watermark_relevance_ner_result['ner_better'] / total_items * 100:.2f}%")
    print(f"NER Tie: {watermark_relevance_ner_result['tie'] / total_items * 100:.2f}%")
    
    print("\nWatermark Coherence:")
    print(f"LLM Original Better: {watermark_coherence_llm_result['original_better'] / total_items * 100:.2f}%")
    print(f"LLM Better: {watermark_coherence_llm_result['llm_better'] / total_items * 100:.2f}%")
    print(f"LLM Tie: {watermark_coherence_llm_result['tie'] / total_items * 100:.2f}%")
    
    print(f"NER Original Better: {watermark_coherence_ner_result['original_better'] / total_items * 100:.2f}%")
    print(f"NER Better: {watermark_coherence_ner_result['ner_better'] / total_items * 100:.2f}%")
    print(f"NER Tie: {watermark_coherence_ner_result['tie'] / total_items * 100:.2f}%")
    
    print("\nWatermark Interestingness:")
    print(f"LLM Original Better: {watermark_interestingness_llm_result['original_better'] / total_items * 100:.2f}%")
    print(f"LLM Better: {watermark_interestingness_llm_result['llm_better'] / total_items * 100:.2f}%")
    print(f"LLM Tie: {watermark_interestingness_llm_result['tie'] / total_items * 100:.2f}%")
    
    print(f"NER Original Better: {watermark_interestingness_ner_result['original_better'] / total_items * 100:.2f}%")
    print(f"NER Better: {watermark_interestingness_ner_result['ner_better'] / total_items * 100:.2f}%")
    print(f"NER Tie: {watermark_interestingness_ner_result['tie'] / total_items * 100:.2f}%")
    
if __name__ == "__main__":
    main()
    
    # # 결과 저장용 딕셔너리
    # relevance_llm_result = {"original_better": 0, "llm_better": 0, "tie": 0}
    # relevance_ner_result = {"original_better": 0, "ner_better": 0, "tie": 0}
    # coherence_llm_result = {"original_better": 0, "llm_better": 0, "tie": 0}
    # coherence_ner_result = {"original_better": 0, "ner_better": 0, "tie": 0}
    # interestingness_llm_result = {"original_better": 0, "llm_better": 0, "tie": 0}
    # interestingness_ner_result = {"original_better": 0, "ner_better": 0, "tie": 0}
    
    # total_items = len(data)
    
#     for idx, item in enumerate(data):
#         original_text = item.get("original_text")
#         llm_text = item.get("llm_watermarked_text")
#         ner_text = item.get("ner_watermarked_text")
        
#         relevance_original_llm, relevance_llm = evaluate_text_with_gpt(original_text, llm_text, "relevance")
#         relevance_original_ner, relevance_ner = evaluate_text_with_gpt(original_text, ner_text, "relevance")
        
#         coherence_original_llm, coherence_llm = evaluate_text_with_gpt(original_text, llm_text, "coherence")
#         coherence_original_ner, coherence_ner = evaluate_text_with_gpt(original_text, ner_text, "coherence")
        
#         interestingness_original_llm, interestingness_llm = evaluate_text_with_gpt(original_text, llm_text, "interestingness")
#         interestingness_original_ner, interestingness_ner = evaluate_text_with_gpt(original_text, ner_text, "interestingness")
        
#         # Relevance
#         if relevance_original_llm > relevance_llm:
#             relevance_llm_result["original_better"] += 1
#         elif relevance_original_llm < relevance_llm:
#             relevance_llm_result["llm_better"] += 1
#         else:
#             relevance_llm_result["tie"] += 1
            
#         if relevance_original_ner > relevance_ner:
#             relevance_ner_result["original_better"] += 1
#         elif relevance_original_ner < relevance_ner:
#             relevance_ner_result["ner_better"] += 1
#         else:
#             relevance_ner_result["tie"] += 1
            
#         # Coherence
#         if coherence_original_llm > coherence_llm:
#             coherence_llm_result["original_better"] += 1
#         elif coherence_original_llm < coherence_llm:
#             coherence_llm_result["llm_better"] += 1
#         else:
#             coherence_llm_result["tie"] += 1
        
#         if coherence_original_ner > coherence_ner:
#             coherence_ner_result["original_better"] += 1
#         elif coherence_original_ner < coherence_ner:
#             coherence_ner_result["ner_better"] += 1
#         else:
#             coherence_ner_result["tie"] += 1
            
#         # Interestingness
#         if interestingness_original_llm > interestingness_llm:
#             interestingness_llm_result["original_better"] += 1
#         elif interestingness_original_llm < interestingness_llm:
#             interestingness_llm_result["llm_better"] += 1
#         else:
#             interestingness_llm_result["tie"] += 1
        
#         if interestingness_original_ner > interestingness_ner:
#             interestingness_ner_result["original_better"] += 1
#         elif interestingness_original_ner < interestingness_ner:
#             interestingness_ner_result["ner_better"] += 1
#         else:
#             interestingness_ner_result["tie"] += 1
        
#     # 비율 계산 및 출력
#     print("\n=== 결과 비율 (100% 기준) ===")
#     print("Relevance:")
#     print(f"LLM Original Better: {relevance_llm_result['original_better'] / total_items * 100:.2f}%")
#     print(f"LLM Better: {relevance_llm_result['llm_better'] / total_items * 100:.2f}%")
#     print(f"LLM Tie: {relevance_llm_result['tie'] / total_items * 100:.2f}%")
    
#     print(f"NER Original Better: {relevance_ner_result['original_better'] / total_items * 100:.2f}%")
#     print(f"NER Better: {relevance_ner_result['ner_better'] / total_items * 100:.2f}%")
#     print(f"NER Tie: {relevance_ner_result['tie'] / total_items * 100:.2f}%")
    
#     print("\nCoherence:")
#     print(f"LLM Original Better: {coherence_llm_result['original_better'] / total_items * 100:.2f}%")
#     print(f"LLM Better: {coherence_llm_result['llm_better'] / total_items * 100:.2f}%")
#     print(f"LLM Tie: {coherence_llm_result['tie'] / total_items * 100:.2f}%")
    
#     print(f"NER Original Better: {coherence_ner_result['original_better'] / total_items * 100:.2f}%")
#     print(f"NER Better: {coherence_ner_result['ner_better'] / total_items * 100:.2f}%")
#     print(f"NER Tie: {coherence_ner_result['tie'] / total_items * 100:.2f}%")
    
#     print("\nInterestingness:")
#     print(f"LLM Original Better: {interestingness_llm_result['original_better'] / total_items * 100:.2f}%")
#     print(f"LLM Better: {interestingness_llm_result['llm_better'] / total_items * 100:.2f}%")
#     print(f"LLM Tie: {interestingness_llm_result['tie'] / total_items * 100:.2f}%")
    
#     print(f"NER Original Better: {interestingness_ner_result['original_better'] / total_items * 100:.2f}%")
#     print(f"NER Better: {interestingness_ner_result['ner_better'] / total_items * 100:.2f}%")
#     print(f"NER Tie: {interestingness_ner_result['tie'] / total_items * 100:.2f}%")
    
# if __name__ == "__main__":
#     main()