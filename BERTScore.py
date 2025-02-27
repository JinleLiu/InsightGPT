from bert_score import score

# 定义参考句子和生成句子
refs = ["The image depicts an urban street scene at an intersection. The road is marked with multiple lane markings and arrows indicating the direction of traffic flow. There is a blue car in the center of the intersection, which appears to be stopped or moving very slowly, as suggested by the position of the traffic lights. The traffic lights are red, indicating that vehicles should stop.On the left side of the intersection, there are pedestrian crossings with white stripes, and there are no pedestrians visible in the image. The right side of the intersection has a dedicated bus lane, as indicated by the blue sign with a white arrow pointing to the right.The surrounding area includes a mix of buildings and greenery, suggesting a blend of urban and green spaces. The sky is overcast, and the lighting in the image is soft, which might indicate either early morning or late afternoon.The overall scene is orderly and typical of a well-regulated urban traffic system. The presence of traffic lights and clear lane markings suggests that this is a location where traffic management is important to maintain safety and order. The blue car's position in the center of the intersection could be due to a number of reasons, such as waiting for a pedestrian to cross, obeying a traffic signal, or perhaps being in the process of making a turn. The absence of other vehicles or pedestrians in the image could be due to the timing of the photograph, or it could indicate a moment of low traffic activity."]
cands = ["The image depicts a wide, multi-lane road with a clear view of a blue car in the middle lane. The road is marked with white lines indicating lanes and a crosswalk, and there are traffic lights visible at the intersection. The surroundings suggest a city or urban environment, with trees and buildings in the background. The sky is overcast, and the lighting suggests it might be early morning or late afternoon. The overall scene is calm and there are no other vehicles or pedestrians visible in the immediate vicinity."]

# 使用bert_score计算分数
P, R, F1 = score(cands, refs, lang='en', model_type="roberta-large", verbose=True)

# 打印结果
print("Precision:", P)
print("Recall:", R)
print("F1 score:", F1)
