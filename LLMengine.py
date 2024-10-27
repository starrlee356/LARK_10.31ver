import torch
import os
import re
import ollama
from llm import LLM_ollama, LLM_vllm, LLM_zhipu

class BaseLLMAnswer:
    def __init__(self, model_name, api):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_source_length = 512
        self.max_new_tokens = 64
        self.model_name = model_name
        self.api = api        

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
            """
            content = ollama.generate(model=self.model_name, prompt=q)["response"]
            response.append(content)
            """
            if self.api == "zhipu":
                llm = LLM_zhipu(self.model_name)
            if self.api == "vllm":
                llm = LLM_vllm(self.model_name)
            if self.api == "ollama":
                llm = LLM_ollama(self.model_name)
            res = llm.run(q)
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

        


