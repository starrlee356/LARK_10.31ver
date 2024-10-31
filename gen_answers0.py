import os
import json
from tqdm import trange, tqdm
from LLMengine import BaseLLMAnswer 
from compute_scores import compute_score_main
from global_config import *
import logging
import argparse
import re
from llm import LLM_ollama, LLM_vllm, LLM_zhipu
import time

def clean_string(string):
    clean_str = re.sub(r"[^0-9,]","",string)
    return clean_str


def main(args):
    if args.api == "zhipu":
        llm = LLM_zhipu(args.model_name)
    if args.api == "vllm":
        llm = LLM_vllm(args.model_name)
    if args.api == "ollama":
        llm = LLM_ollama(args.model_name)
    engine = BaseLLMAnswer(llm)

    def gen_ans(qtype, qpattern, args):
        premise_questions = {}
        #for idx in range(test_q_num[qtype]):
        for idx in range(args.qnum):
            question_path = os.path.join(f"{args.output_path}","LARK_test_questions",f"{qtype}_{idx}_question.json")
            with open(question_path) as q_f:
                question = json.load(q_f)
                premise_questions[idx] = question
           
        li_premise_questions = list(premise_questions.items())
        predictions_path = os.path.join(f"{args.output_path}","LARK_test_preds")
        if not os.path.exists(predictions_path):
            os.makedirs(predictions_path)
        for i in tqdm(iterable=trange(len(premise_questions)//args.batch_size+1), desc=f"{qtype}:{qpattern}"):
            pq_subset = dict(li_premise_questions[i*args.batch_size:(i+1)*args.batch_size])
            engine.log_step_answer(qtype, pq_subset, output_path=predictions_path)

    if args.qtype == "all":
        for qtype, qpattern in QUERY_STRUCTS.items():
            gen_ans(qtype, qpattern, args)
    else:
        qpattern = QUERY_STRUCTS[args.qtype]
        gen_ans(args.qtype, qpattern, args)
    engine.get_token_len()
    print(f"avg prompt token len = {engine.prompt_token_len/engine.llm_cnt}")
    print(f"avg gen token len = {engine.gen_token_len/engine.llm_cnt}")
    print(f"llm time = {engine.llm_time} s")

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default="../data/NELL-betae/processed", help="Path to processed files.")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size of the model")
    parser.add_argument('--ckpt_dir', type=str, default="", help="Checkpoint dir for FAIR LLM")
    parser.add_argument('--tokenizer_path', type=str, default="", help="Tokenizer dir for FAIR LLM")
    parser.add_argument('--lora_weights',type=str, default="", help="Path to lora weights for Alpaca LLM")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')

    parser.add_argument('--ground_truth_path', type=str, default="../data/NELL-betae/processed/sorted_test_answers", help="Path to ground truth data.")
    parser.add_argument('--prediction_path', type=str, default="../data/NELL-betae/processed/LARK_test_preds", help="Path to the prediction files.")
    parser.add_argument('--log_score_path', type=str, default="../data/NELL-betae/processed/LARK_scores", help="Path to log scores")
    parser.add_argument('--score_file', type=str, default="1p.txt", help="file name to log scores")
    parser.add_argument('--random_size', type=int, default=100)
    parser.add_argument('--whole_size', type=int, default=0)
    parser.add_argument('--qtype', type=str, default="1p")
    parser.add_argument('--qnum', type=int, default=4000)

    parser.add_argument('--model_name', type=str, default="llama3:8b")
    parser.add_argument('--api', type=str, default="vllm")
    args = parser.parse_args()
    main(args)
    compute_score_main(args)
    end = time.time()
    print(f"total time = {end-start} s")



    
