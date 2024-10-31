from html import entities
import json
import os
import csv
from tqdm import tqdm
import logging
import pickle as pkl
import argparse
from premise_generator import PremiseGenerator
from prompt_step_generator import StepLogicalPromptGenerator
import multiprocessing as mp
from global_config import *

logging.basicConfig(level=logging.INFO)


def process_logical_queries(logical_query, qtype, idx):
    e1, r1, e2, r2, e3, r3 = logic_processor.parse_logical_query(logical_query, qtype)
    entity_set = list(filter(lambda x: x!=None, [e1, e2, e3]))
    relation_set = list(filter(lambda x: x!=None, [r1, r2, r3]))
    premise = premise_generator.generate_premise(entity_set, relation_set, qtype)
    question = logic_processor.generate_prompt(logical_query, qtype)
    premise_question = {"premise": premise,
                        "question": question
                        }
    question_file_path = os.path.join(f"{args.output_path}","LARK_all_questions",f"{qtype}_{idx}_question.json")

    with open(question_file_path,"w") as q_f:
        json.dump(premise_question, q_f)

#Premise Generation
def main():
    id2q = pkl.load(open(os.path.join(args.data_path, "processed", "idx2query.pkl"), "rb"))

    entity_triplets, relation_triplets = {}, {}
    triplet_files = [os.path.join(f"{args.data_path}","train.txt"), 
                     os.path.join(f"{args.data_path}","valid.txt"), 
                     os.path.join(f"{args.data_path}","test.txt")]
    for triplet_file in triplet_files:
        with open(triplet_file,"r") as kg_data_file:
            kg_tsv_file = csv.reader(kg_data_file, delimiter="\t")
            for line in kg_tsv_file:
                e1, r, e2 = map(int,line)
                triplet = (e1, r, e2)
                if e1 in entity_triplets: entity_triplets[e1].add(triplet)
                else: entity_triplets[e1] = set([triplet])
                if e2 in entity_triplets: entity_triplets[e2].add(triplet)
                else: entity_triplets[e2] = set([triplet])
                if r in relation_triplets: relation_triplets[r].add(triplet)
                else: relation_triplets[r] = set([triplet])
    if not os.path.exists(f"{args.output_path}"):
        os.makedirs(f"{args.output_path}")
    with open(os.path.join(f"{args.output_path}","relation_triplets.pkl"),"wb") as relation_triplets_file:
        pkl.dump(relation_triplets, relation_triplets_file)

    global logic_processor
    global premise_generator
    entities_path = os.path.join(f"{args.data_path}","id2ent.pkl")
    relations_path = os.path.join(f"{args.data_path}","id2rel.pkl")
    logic_processor = StepLogicalPromptGenerator(entities_path=entities_path,relations_path=relations_path)
    premise_generator = PremiseGenerator(entities_path=entities_path,
                                        relations_path=relations_path,
                                        entity_triplets_path=os.path.join(f"{args.output_path}","entity_triplets.pkl"),
                                        relation_triplets_path=os.path.join(f"{args.output_path}","relation_triplets.pkl"))
    
    step_question_path = os.path.join(f"{args.output_path}","LARK_all_questions")

    if not os.path.exists(step_question_path):
        os.makedirs(step_question_path)

    for qtype, qnum in whole_q_num.items():
        for idx in tqdm(range(qnum), desc=f"qtype"):
            q = id2q[qtype][idx]
            process_logical_queries(q, qtype, idx)
        
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/NELL-betae", help="Path to raw data.")
    parser.add_argument('--output_path', type=str, default="../data/NELL-betae/processed", help="Path to output the processed files.")
    args = parser.parse_args()
    main()
