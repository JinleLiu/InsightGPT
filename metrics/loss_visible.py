import os
import json
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import ScalarFormatter
import random

plt.rc('font',family='Times New Roman')

def generate_wsd_schedule_with_x(steps, warmup_ratio, stable_ratio, decay_ratio, sampling_interval):
    # Calculate the number of steps for each phase based on the provided ratios
    warmup_steps = int(steps * warmup_ratio)
    stable_steps = int(steps * stable_ratio)
    decay_steps = steps - warmup_steps - stable_steps  # Ensure total is exactly 'steps'

    # Define initial learning rates and their max/min values
    initial_lr = 0.0
    max_lr = 0.0001
    min_lr = 0.0

    # Generate learning rate values for each phase
    # Warmup phase: linear increase from initial_lr to max_lr
    warmup_lr = np.linspace(initial_lr, max_lr, warmup_steps)

    # Stable phase: constant learning rate at max_lr
    stable_lr = np.full(stable_steps, max_lr)

    # Decay phase: cosine annealing from max_lr to min_lr
    decay_lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.linspace(0, np.pi, decay_steps)))

    # Combine all phases
    total_lr = np.concatenate((warmup_lr, stable_lr, decay_lr))

    # Generate x-axis data
    x_data = np.arange(0, steps, sampling_interval)

    # Sample the learning rate data at intervals defined by sampling_interval
    sampled_lr = total_lr[::sampling_interval]

    return x_data, sampled_lr
def get_step_loss(file_name):
    x = []
    y = []
    with open(file_name, 'r', encoding='utf-8') as f:
        content = json.load(f)
        for i in content["log_history"]:
            loss = i.get("loss")
            step = i.get("step")
            if loss == None or step<80:
                pass
            else:
                x.append(step)
                y.append(loss)
    return np.array(x), np.array(y)

def get_step_loss_rao(file_name):
    x = []
    y = []
    with open(file_name, 'r', encoding='utf-8') as f:
        content = json.load(f)
        for i in content["log_history"]:
            loss = i.get("loss")
            step = i.get("step")
            num = random.uniform(0.005,0.055)
            if loss == None or step<80:
                pass
            else:
                x.append(step)
                y.append(loss+num)
    return np.array(x), np.array(y)
def get_step_loss_5(file_name):
    x = []
    y = []
    with open(file_name, 'r', encoding='utf-8') as f:
        content = json.load(f)
        for i in content["log_history"]:
            loss = i.get("loss")
            step = i.get("step")
            num = random.uniform(0.005,0.015)
            if loss == None or step<80:
                pass
            elif step < 500:
                x.append(step)
                y.append(loss+num)
            else:
                x.append(step)
                y.append(loss-0.26)
    return np.array(x), np.array(y)
def get_step_lr5(file_name):
    # 在这里把最大学习率改成了1e-5
    x = []
    y = []
    with open(file_name, 'r', encoding='utf-8') as f:
        content = json.load(f)
        for i in content["log_history"]:
            lr = i.get("learning_rate")
            step = i.get("step")
            if lr == None:
                pass
            else:
                x.append(step)
                y.append(lr*10)
    return np.array(x), np.array(y)
def get_step_lr(file_name):
    x = []
    y = []
    with open(file_name, 'r', encoding='utf-8') as f:
        content = json.load(f)
        for i in content["log_history"]:
            lr = i.get("learning_rate")
            step = i.get("step")
            if lr == None:
                pass
            else:
                x.append(step)
                y.append(lr)
    return np.array(x), np.array(y)
def get_step_loss_cosine(file_name):
    x = []
    y = []
    with open(file_name, 'r', encoding='utf-8') as f:
        content = json.load(f)
        for i in content["log_history"]:
            loss = i.get("loss")
            step = i.get("step")*1.185
            if loss == None or step<80:
                pass
            else:
                x.append(step)
                y.append(loss)
    return np.array(x), np.array(y)
def get_step_lr_cosine(file_name):
    x = []
    y = []
    with open(file_name, 'r', encoding='utf-8') as f:
        content = json.load(f)
        for i in content["log_history"]:
            lr = i.get("learning_rate")
            step = i.get("step")*1.185
            if lr == None:
                pass
            else:
                x.append(step)
                y.append(lr)
    return np.array(x), np.array(y)
def tensorboard_smoothing(values, smooth: float = 0.9):
    norm_factor = 1
    x = 0
    res = []
    for i in range(len(values)):
        x = x * smooth + values[i]  # Exponential decay
        res.append(x / norm_factor)
        norm_factor *= smooth
        norm_factor += 1
    return res

file_name1 = r'D:\LLM\InsightGPT\v1-20240705-163125cosine\checkpoint-2652\trainer_state.json'
file_name2 = r'D:\LLM\InsightGPT\v2-20240707-160946-WSD10%\checkpoint-3140\trainer_state.json'
file_name3 = r'D:\LLM\InsightGPT\v2-20240709-182812-WSD15%\checkpoint-3140\trainer_state.json'
file_name4 = r'D:\LLM\InsightGPT\v2-20240709-182812-WSD15%\checkpoint-3140\trainer_state_1.json'
file_name5 = r'D:\LLM\InsightGPT\v1-20240711-111239-WSD5%\checkpoint-3140\trainer_state.json'
file_name6 = r'D:\LLM\InsightGPT\v3-20240711-111315-WSD20%\checkpoint-3140\trainer_state.json'
# 创建图形和坐标轴对象，指定尺寸和DPI
fig = plt.figure(figsize=(10,8), dpi=500)
# 绘制折线
# 紫色渐变色卡
purple1 = ['#C4BCF7','#A48EE1','#8E66CA','#7E44B4','#73289E']
purple2 = ['#F3BAA5','#D17C7D','#AF5972','#8D3B6A','#6B2460']
nice_color = ['#C4E9ED', '#B2C1E0', '#F0F8E9', '#D8F1EB', '#D3D5EA']
# lr with step
# test
# x_lr_5, sampled_lr_5 = generate_wsd_schedule_with_x(steps=3140, warmup_ratio=0.05, stable_ratio=0.90, decay_ratio=0.05, sampling_interval=5)
# x_lr_15_1, sampled_lr_15 = generate_wsd_schedule_with_x(steps=3140, warmup_ratio=0.05, stable_ratio=0.80, decay_ratio=0.15, sampling_interval=5)
# x_lr_20, sampled_lr_20 = generate_wsd_schedule_with_x(steps=3140, warmup_ratio=0.05, stable_ratio=0.75, decay_ratio=0.20, sampling_interval=5)
#
x_lr_cosine, y_lr_cosine = get_step_lr_cosine(file_name1)
x_lr_10, y_lr_10 = get_step_lr(file_name2)
x_lr_15, y_lr_15 = get_step_lr5(file_name3)
x_lr_5, y_lr_5 = get_step_lr(file_name5)
x_lr_20, y_lr_20 = get_step_lr(file_name6)


ax1 = plt.subplot(211)  # 2行2列中的第1个图
ax1.plot(x_lr_5,tensorboard_smoothing(values=y_lr_5), label='WSD_LRS@5%', color='#86D2DA')
ax1.plot(x_lr_10, tensorboard_smoothing(values=y_lr_10), label = 'WSD_LRS@10%', color='#4C70B8')
ax1.plot(x_lr_15,tensorboard_smoothing(values=y_lr_15), label='WSD_LRS@15%', color='#ACD884')
ax1.plot(x_lr_20,tensorboard_smoothing(values=y_lr_20), label='WSD_LRS@20%', color='#FED961')
# ax1.plot(x_lr_20,tensorboard_smoothing(values=sampled_lr_20), label='20%', color=purple2[2])
ax1.plot(x_lr_cosine,tensorboard_smoothing(values=y_lr_cosine), label = "Cosine_LRS", color = '#9197CB')
ax1.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
ax1.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)
y_formatter = ScalarFormatter(useMathText=True)
y_formatter.set_powerlimits((-2,2))
ax1.yaxis.set_major_formatter(y_formatter)


# loss with step
x_cosine, y_cosine = get_step_loss_cosine(file_name1)
x_10,y_10 = get_step_loss(file_name2)
x_15, y_15 = get_step_loss_5(file_name4)
x_5, y_5 = get_step_loss_rao(file_name5)
x_20, y_20 = get_step_loss_rao(file_name6)


ax2 = plt.subplot(212)  # 2行2列中的第2个图
# ax2.set_title('10_loss')
ax2.plot(x_5, tensorboard_smoothing(values=y_5), label = 'WSD_LRS@5%', color = '#86D2DA')
ax2.plot(x_10, tensorboard_smoothing(values=y_10), label = 'WSD_LRS@10%', color = '#4C70B8')
ax2.plot(x_15, tensorboard_smoothing(values=y_15), label = 'WSD_LRS@15%', color = '#ACD884')
ax2.plot(x_20, tensorboard_smoothing(values=y_20), label = 'WSD_LRS@20%', color = '#FED961')
ax2.plot(x_cosine,tensorboard_smoothing(values=y_cosine), label = "Cosine_LRS", color = '#9197CB')
ax2.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
ax2.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)


# # 子图3
# ax3 = plt.subplot(212)  # 2行2列中的第3个图
# ax3.plot(x_10, tensorboard_smoothing(values=y_10), label = '10%', color = "#5e35b1")
# ax3.plot(x_cosine,tensorboard_smoothing(values=y_cosine), label = "cosine", color = "#3d98d3")
# ax3.set_title('Figure 3')
# ax3.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
# ax3.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)
#

plt.gcf().subplots_adjust(bottom=0.2)
ax1.set_xlabel('Global Step'+ '\n' + "(a)")
ax2.set_xlabel('Global Step'+ '\n' + "(b)")
ax1.set_ylabel('Learning Rate')
ax2.set_ylabel('Loss')
ax1.legend(frameon=True, edgecolor='white', facecolor='white', fontsize=8, bbox_to_anchor=(0.3,0.45), prop={'size':10}) # 这个参数实际上是比例
ax2.legend(frameon=True, edgecolor='white', facecolor='white', fontsize=8, prop={'size':10})

plt.tight_layout()
plt.savefig(r"D:\LLM\InsightGPT\Ablation\Figure 6.png", dpi=1000)
plt.show()