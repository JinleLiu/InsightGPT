from torchmetrics import BLEUScore
from rouge import Rouge
import os
import json
from tqdm import  tqdm
from bert_score import score

def get_label_response(task_num):
    label_file_ic = r'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\image_captioning.json'
    label_file_oar = r'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\object_attributes_recognition.json'
    label_file_sr = r'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\spatial_reasoning.json'
    label_file_tr = r'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\template_recognition.json'
    label_file_tvr = r'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\traffic_violation_recognition.json'
    label_file_vr = r'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\visual_reasoning.json'

    task_list = ['image_captioning', 'object_attributes_recognition', 'spatial_reasoning', 'template_recognition', 'traffic_violation_recognition', 'visual_reasoning']

    compare_llm_list = ['GLM-4V', 'MiniCPM-V', 'MiniCPM-V25', 'QwenVL', 'Yi-vl', 'LLaVA-1.6', 'InsightGPT']
    # response_file = fr'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\{compare_llm_list[0]}_ans\{task_list[0]}_ans.json'

    # with open(label_file, 'r', encoding='utf-8') as f:
    #     raw_data = json.load(f)
    #     print(len(raw_data))

    for num in range(0, 6): # 这个是用来指定模型的
        label_file = fr'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\{task_list[num]}.json'
        response_file = fr'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\{compare_llm_list[task_num]}_ans\{task_list[num]}_ans.json'
        if os.path.exists(response_file) == True:
            with open(label_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            with open(response_file, 'r', encoding='utf-8') as f1:
                response_data = json.load(f1)
            for i in range(0,200):
                label_response = label_data[i]["response"]
                predict_response = response_data[i]["response"]
        else:
            print("No" + response_file)
    return label_response, predict_response

def get_metric_scores(predict, label):
    # for BLUE
    translate_corpus = [predict.split()]  # 预测的结果
    reference_corpus = [[label.split()]]  # 标签
    # for rough
    references_flat = ' '.join(label)
    candidates_flat = ' '.join(predict)

    bleu_1 = BLEUScore(n_gram=1)  # 默认n_gram=4
    bleu_2 = BLEUScore(n_gram=2)
    bleu_3 = BLEUScore(n_gram=3)
    bleu_4 = BLEUScore(n_gram=4)
    print("BLEU-1 Score:", bleu_1(reference_corpus, translate_corpus).item())
    print("BLEU-2 Score:", bleu_2(reference_corpus, translate_corpus).item())
    print("BLEU-3 Score:", bleu_3(reference_corpus, translate_corpus).item())
    print("BLEU-4 Score:", bleu_4(reference_corpus, translate_corpus).item())

    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidates_flat, references_flat, avg=True)
    rouge_l = rouge_scores['rouge-l']['f']
    print("ROUGE-L Score:", rouge_l)

def get_metric_scores(llm_num):
    label_file_ic = r'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\image_captioning.json'
    label_file_oar = r'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\object_attributes_recognition.json'
    label_file_sr = r'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\spatial_reasoning.json'
    label_file_tr = r'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\template_recognition.json'
    label_file_tvr = r'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\traffic_violation_recognition.json'
    label_file_vr = r'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\visual_reasoning.json'

    task_list = ['image_captioning', 'object_attributes_recognition', 'spatial_reasoning', 'template_recognition', 'traffic_violation_recognition', 'visual_reasoning','coco_image_captioning', 'coco_object_attributes_recognition']

    compare_llm_list = ['GLM-4V', 'MiniCPM-V', 'MiniCPM-V25', 'QwenVL', 'Yi-vl', 'LLaVA-1.6', 'InsightGPT']
    compare_list = ['cosine', '5', '15', '20']
    # response_file = fr'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\{compare_llm_list[0]}_ans\{task_list[0]}_ans.json'

    # with open(label_file, 'r', encoding='utf-8') as f:
    #     raw_data = json.load(f)
    #     print(len(raw_data))

    for num in tqdm([2,4]):  # 这个是用来指定任务的
        label_file = fr'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\{task_list[num]}.json'
        # response_file = fr'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\{compare_llm_list[llm_num]}_ans\{task_list[num]}_ans.json'
        response_file = fr'D:\LLM\InsightGPT\Ablation\trans\{task_list[num]}_{compare_list[llm_num]}.json'
        BLEU_1_score = 0
        precision = 0
        recall = 0
        F1score = 0
        Rouge_l_score = 0
        if os.path.exists(response_file) == True:
            with open(label_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            with open(response_file, 'r', encoding='utf-8') as f1:
                response_data = json.load(f1)
            rouge = Rouge()
            for i in range(0, len(response_data)):
                label_response = label_data[i]["response"]
                predict_response = response_data[i]["response"]
                # for BLUE
                translate_corpus = [predict_response.split()]  # 预测的结果
                reference_corpus = [[label_response.split()]]  # 标签
                # for rough
                references_flat = ' '.join(label_response)
                candidates_flat = ' '.join(predict_response)
                bleu_1 = BLEUScore(n_gram=1)  # 默认n_gram=4
                BLEU_1_score += float(bleu_1(reference_corpus, translate_corpus).item())
                rouge_scores = rouge.get_scores(candidates_flat, references_flat, avg=True)
                Rouge_l_score += float(rouge_scores['rouge-l']['f'])
                P, R, F1 = score([predict_response], [label_response], lang='en', model_type="roberta-large", verbose=True)
                precision += float(P)
                recall += float(R)
                F1score += float(F1)

        else:
            print("No" + response_file)

        # print(r'For {}'.format(task_list[num]))
        # print("BLEU-1 Score:", BLEU_1_score/200)
        # print("BERTScore-P:", precision/200)
        # print("BERTScore-R:", recall/200)
        # print("BERTScore-F:", F1score/200)
        # print("ROUGE-L Score:", Rouge_l_score/200)
        ans_dict = {"Task": task_list[num] + '_'+ compare_list[llm_num], "BLEU-1 Score:": BLEU_1_score/len(response_data), "BERTScore-P:": precision/len(response_data), "BERTScore-R:": recall/len(response_data), "BERTScore-F:": F1score/len(response_data), "ROUGE-L Score:": Rouge_l_score/len(response_data)}
        # result_solve_flie = fr'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\{compare_llm_list[llm_num]}_ans\coco_results.json'
        result_solve_flie = fr'D:\LLM\InsightGPT\Ablation\trans\results.json'

        with open(result_solve_flie, 'a+', encoding='utf-8') as f2:
            json.dump(ans_dict, f2, indent=4)


if __name__ == '__main__':
    for i in range(0, 4):
        get_metric_scores(i)