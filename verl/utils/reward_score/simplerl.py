import torch
import re
from multiprocessing import Process, Queue
from .simplerl_utils.paser import extract_answer as qwen_extract_answer
from .simplerl_utils.grader import math_equal as qwen_math_equal

def qwen_math_equal_subprocess(prediction, reference,  timeout_seconds=10):
    def worker(q, prediction, reference):
        result = qwen_math_equal(prediction=prediction, reference=reference, timeout=False)
        q.put(result)

    q = Queue()
    p = Process(target=worker, args=(q, prediction, reference))
    p.start()
    
    # 添加超时处理
    p.join(timeout=timeout_seconds)  # 等待进程完成，最多等待 timeout_seconds 秒
    
    # 如果进程还在运行，则终止它并返回 False
    if p.is_alive():
        p.terminate()
        p.join()  # 确保进程被完全清理
        return False
        
    # 如果进程正常完成，获取结果
    try:
        return q.get_nowait()
    except:
        return False   

def preprocess_box_response_for_qwen_prompt(model_output, answer):
    # breakpoint()
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    extract_answer = qwen_extract_answer(model_output, data_name="math") #TODO: check the data_name, hard code here for now
    
    
    # temp_query = ""
    # temp_response = ""

    # temp_query = sequence.split("ASSISTANT:\n")[0]

    # temp_response = sequence.split("**Final Answer**")[-1]
    
    # pattern = r'\\boxed\{(.*?)\}\s*\\\]'

    # # 使用 re.DOTALL 确保能匹配跨行文本
    # match = re.search(pattern, temp_response)
    # #match = re.search(pattern, temp_response)
    # if match:
    #     temp_answer = match.group(1)
    # else:
    #     temp_answer = "none"

    # #temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    # #temp_response = sequence.split("<|reserved_special_token_0|>The final answer is:")[0]

    # #response_list = temp_response.split("<|reserved_special_token_0|>")

    # processed_solution = temp_response + "\n\n# Answer\n\n" + temp_answer + "<|reserved_special_token_0|>"

    # processed_solution = re.sub(r"<\|end_of_text\|>", "", processed_solution)
    
    if qwen_math_equal_subprocess(prediction=extract_answer, reference=answer):
        box_match = 1.0
    else:
        box_match = -0.5
        
    if "boxed" not in model_output:
        box_match = -1.0
        

    return "", box_match

def compute_score(solution_str, ground_truth) -> float:
    query, box_match = preprocess_box_response_for_qwen_prompt(solution_str, ground_truth)
    return box_match