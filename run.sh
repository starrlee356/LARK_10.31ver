cd ~/liuxy/llmReason/baseline_LARK
python -m pdb gen_answers.py \
 --ground_truth_path ../data/NELL-betae/processed/sorted_test_answers \
 --question_path ../data/NELL-betae/processed/LARK_test_questions \
 --prediction_path ../data/NELL-betae/processed/LARK_test_predictions \
 --log_score_path ../data/NELL-betae/processed/LARK_scores \
 --random_list_path ../data/NELL-betae/processed/LARK_random_list \
 --qnum_dict_file ../data/NELL-betae/qnum_dict.pkl \
 --score_file 1p.txt \
 --qtype 1p \
 --qnum 1000 \
 --qsize 17021 \
 --model_name llama3:8b \
 --api vllm \
    
    

