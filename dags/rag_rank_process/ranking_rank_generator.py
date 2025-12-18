import os
import json
from text_processing_utils import run_ollama, save_jsonl_lines, read_jsonl_file
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
OLLAMA_MODEL_RANKER = "qwen3:8b" 
INPUT_FILE_UNRANKED_DATA = 'synthetic_ranking_unranked.jsonl'
OUTPUT_FILE_FINAL_RANKED = 'synthetic_ranking_final.jsonl'

# --- FIX APPLIED HERE ---
# The example JSON format keys must be escaped using double curly braces {{ and }}
# so that the .format() call ignores them and treats them as literal text.
PROMPT_RANKING_GEN = """
You are an expert Relevance Ranker. Your task is to assign a relevance score between 0 and 10 to each of the provided text excerpts based on how well they answer the question.

The scoring rules are strict:
- Score 10: The excerpt contains the full, direct answer to the question.
- Score 5-7: The excerpt contains partial information, background, or related context, but does not fully answer the question.
- Score 0: The excerpt is completely irrelevant or unrelated to the question.
- Scores 1-4 or 8-9: Use these for intermediate relevance levels.

Output a numbered list of JSON objects, one for each excerpt.
Example Output Format:
[
  {{ "excerpt_index": 1, "score": 10 }},
  {{ "excerpt_index": 2, "score": 2 }},
  {{ "excerpt_index": 3, "score": 6 }}
]

---
Question: {question}

Excerpts to Rank:
{excerpts_list_formatted}
---

Your Output (JSON Array Only):
"""
# --------------------------

def format_excerpts_for_prompt(excerpts: list) -> str:
    """Formats the list of excerpts into a numbered list for the prompt."""
    formatted_list = ""
    for i, excerpt in enumerate(excerpts):
        formatted_list += f"[{i+1}] Excerpt:\n---\n{excerpt['text']}\n---\n\n"
    return formatted_list

def parse_ranking_response(response_text: str, num_excerpts: int) -> list:
    """
    Attempts to parse the JSON array from the LLM response, with defensive key normalization
    to handle LLM errors like ' "excerpt_index" ' instead of 'excerpt_index'.
    """
    # 1. Try to isolate the JSON array structure
    start = response_text.find('[')
    end = response_text.rfind(']')
    
    if start == -1 or end == -1:
        print("⚠️ Warning: Could not find JSON array boundaries in LLM response. Response:", response_text[:50])
        return [{'excerpt_index': i + 1, 'score': 0} for i in range(num_excerpts)]

    try:
        json_str = response_text[start : end + 1]
        rankings = json.loads(json_str)
        
        # 2. Key Normalization and Validation
        final_rankings = []
        for rank in rankings:
            # Normalize keys to handle model outputting keys like ' "score" '
            normalized_rank = {}
            for key, value in rank.items():
                # Strip leading/trailing whitespace and quotes from the key
                normalized_key = key.strip().strip('"').strip("'")
                normalized_rank[normalized_key] = value

            # Safely extract score and index using .get()
            score = normalized_rank.get('score')
            index = normalized_rank.get('excerpt_index')

            if score is not None and index is not None:
                try:
                    # Convert to integer and clamp score to 0-10
                    score = max(0, min(10, int(score)))
                    final_rankings.append({'excerpt_index': int(index), 'score': score})
                except (ValueError, TypeError):
                    print(f"⚠️ Warning: Found non-integer score/index in rank: {normalized_rank}. Skipping.")
                    continue
        
        # 3. Handle cases where the LLM returned fewer ranks than expected
        if len(final_rankings) < num_excerpts:
            print(f"⚠️ Warning: LLM returned {len(final_rankings)} ranks, expected {num_excerpts}. Returning a list of 0-score placeholders.")
            # This returns a simple placeholder list; the loop below handles assignment
            return [{'excerpt_index': i + 1, 'score': 0} for i in range(num_excerpts)]
            
        return final_rankings

    except Exception as e:
        print(f"⚠️ Severe Error parsing LLM JSON response: {e}. Response: {response_text[:100]}...")
        # Fallback to zero scores if parsing fails
        return [{'excerpt_index': i + 1, 'score': 0} for i in range(num_excerpts)]


def generate_rankings(input_file: str):
    """Reads Q/Excerpt data, generates rankings, and saves the final data."""
    
    # Read the data from the first pipeline
    unranked_data = read_jsonl_file(input_file)
    print(f"Loaded {len(unranked_data)} unranked data points from {input_file}.")
    
    final_ranked_data = []

    for i, item in enumerate(unranked_data):
        question = item['question']
        excerpts = item['excerpts']
        
        print(f"Processing item {i+1}/{len(unranked_data)} for ranking...")
        
        # 1. Format the excerpts for the LLM prompt
        excerpts_list_formatted = format_excerpts_for_prompt(excerpts)
        
        # 2. Format the overall ranking prompt
        ranking_prompt = PROMPT_RANKING_GEN.format(
            question=question, 
            excerpts_list_formatted=excerpts_list_formatted
        )
        
        try:
            # 3. Generate the rankings using Ollama
            response_text = run_ollama(OLLAMA_MODEL_RANKER, "", ranking_prompt).strip()
            
            # 4. Parse the LLM response
            rankings = parse_ranking_response(response_text, len(excerpts))
            
            # 5. Combine original data with scores
            # Use a map to quickly look up scores by excerpt_index
            score_map = {rank['excerpt_index']: rank['score'] for rank in rankings}

            # Assign scores to the original excerpts array based on index matching
            for idx in range(len(excerpts)):
                # Since the LLM uses 1-based indexing, we look up using idx + 1
                score = score_map.get(idx + 1, 0) # Default to 0 if not found
                excerpts[idx]['score'] = score
            
            # Create the final data structure
            final_ranked_data.append({
                "question": question,
                "excerpts": excerpts
            })
            
            # Print status update
            scores = [e.get('score', 0) for e in excerpts]
            print(f"  Q: {question[:60]}...")
            print(f"  Ranks: {scores}")
            
        except Exception as e:
            print(f"Error generating ranking for item {i}: {e}. Skipping.")
            continue

    # Save the final results
    save_jsonl_lines(final_ranked_data, OUTPUT_FILE_FINAL_RANKED)
    print(f"\n✅ Finished. Saved {len(final_ranked_data)} final ranked records to {OUTPUT_FILE_FINAL_RANKED}")


if __name__ == "__main__":
    generate_rankings(INPUT_FILE_UNRANKED_DATA)