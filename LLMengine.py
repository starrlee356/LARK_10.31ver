import torch
import os
import re
import ollama
from llm import LLM_ollama, LLM_vllm, LLM_zhipu
import time

class BaseLLMAnswer:
    def __init__(self, LLM):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_source_length = 512
        self.max_new_tokens = 64     
        self.LLM = LLM
        self.prompt_token_len, self.gen_token_len = self.LLM.get_token_length()
        self.llm_cnt = 0
        self.llm_time = 0

    def get_token_len(self):
        p, g = self.LLM.get_token_length()
        self.prompt_token_len = p - self.prompt_token_len
        self.gen_token_len = g - self.gen_token_len

    def clean_string(self, string):
        clean_str = re.sub(r"[^0-9,]","",string)
        return clean_str


    def process_step_question(self, qtype, premise_questions):
        if len(premise_questions)<1: return None
        explain_tag = premise_questions[0]["question"]["explain_tag"]
        question_tag = premise_questions[0]["question"]["question_tag"]
        step_questions = {}
        for premise_question in premise_questions:
            questions = premise_question["question"]["question"][qtype]
            premise = premise_question["premise"]
            phase = 1
            for question in questions:
                enhanced_premise_question = "".join([premise,question_tag,question,explain_tag])
                if phase in step_questions: step_questions[phase].append(enhanced_premise_question)
                else: step_questions[phase] = [enhanced_premise_question]
                phase += 1
        return step_questions
    
    def swap_question_placeholders(self, questions, step_answers):
        question_with_swaps = []
        for i in range(len(questions)):
            question = questions[i]
            if "[PP1]" in question:
                sai = self.clean_string(step_answers[1][i])
                question = question.replace("[PP1]", "{"+sai+"}")
            if "[PP2]" in question:
                sai = self.clean_string(step_answers[2][i])
                question = question.replace("[PP2]", "{"+sai+"}")    
            if "[PP3]" in question:
                sai = self.clean_string(step_answers[3][i])
                question = question.replace("[PP3]", "{"+sai+"}")
            question_with_swaps.append(question)
        return question_with_swaps
    
    def generate_answer(self, premise_questions):
        response = []
        for q in premise_questions:
            start = time.time()
            res = self.LLM.run(q)
            end = time.time()
            self.llm_time += end - start
            self.llm_cnt += 1 
            response.append(res)
        return response
    
    def generate_step_answer(self, qtype, premise_questions):
        step_questions = self.process_step_question(qtype, premise_questions)
        if step_questions == None: return []
        step_answers = {}
        final_phase = 1
        for phase in range(1,len(step_questions)+1):
            if phase == 1:
                step_answers[phase] = self.generate_answer(step_questions[phase])
            else:
                question_with_swaps = self.swap_question_placeholders(step_questions[phase], step_answers)
                step_answers[phase] = self.generate_answer(question_with_swaps)
            final_phase = phase
        return step_answers[final_phase]

    def log_step_answer(self, qtype, premise_questions={}, output_path=""):
        question_ids = list(premise_questions.keys())
        premise_questions = list(premise_questions.values())
        predicted_answers = self.generate_step_answer(qtype, premise_questions)
        for idx, prediction in enumerate(predicted_answers):
            with open(os.path.join(f"{output_path}",f"{qtype}_{question_ids[idx]}_predicted_answer.txt"),"w") as prediction_file:
                print(prediction, file=prediction_file)

        


