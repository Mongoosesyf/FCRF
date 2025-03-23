"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os
import sys
import json
import yaml
import openai
import importlib
import alfworld
import alfworld.agents.environment
from utils import Model, get_chat, get_completion
# from utils_qwen import qwen_chat, qwen_completion
# from utils_router import router_chat, router_completion
# from utils_g4f import g4f_chat
from env_history import EnvironmentHistory

from typing import List, Dict, Any, Tuple
import re

# > think: To solve the task, I need to find and take the first peppershaker, then put it in/on the drawer, then find and take the second peppershaker, and then put it in/on the drawer.
# OK.
# > think: First, I need to find the first peppershaker. A peppershaker is more likely to appear in/on the cabinets (1-6), countertop (1-3), drawer (1-3), shelf (1-3), or possibly the fridge (1). I can check one by one, starting with cabinet 1.
# OK.


# Sample text input
# # 原成功log
# text = """> go to cabinet 1
# On the cabinet 1, you see a dishsponge 1, a glassbottle 1, and a peppershaker 2.
# > think: Now I find the first peppershaker (2). Next, I need to take it.
# OK.
# > take peppershaker 2 from cabinet 1
# You pick up the peppershaker 2 from the cabinet 1.
# > think: Now I have the first peppershaker (2). Next, I need to put it in/on the drawer. I can check which drawer to use, starting with drawer 1.
# OK.
# > go to drawer 1
# The drawer 1 is closed.
# > open drawer 1
# You open the drawer 1. The drawer 1 is open. In it, you see a creditcard 1, and a spatula 1.
# > think: The drawer 1 has space for more items. I can put the first peppershaker (2) in/on drawer 1.
# OK.
# > put peppershaker 2 in/on drawer 1
# You put the peppershaker 2 in/on the drawer 1.
# > think: Now I have put the first peppershaker in/on drawer 1. Next, I need to find the second peppershaker. I can check the remaining cabinets, countertops, drawers, and shelves. I will start with cabinet 2.
# OK.
# > go to cabinet 2
# The cabinet 2 is closed.
# > open cabinet 2
# You open the cabinet 2. The cabinet 2 is open. In it, you see nothing.
# > go to cabinet 3
# On the cabinet 3, you see a plate 1.
# > go to cabinet 4
# On the cabinet 4, you see a peppershaker 1.
# > think: Now I find the second peppershaker (1). Next, I need to take it.
# OK.
# > take peppershaker 1 from cabinet 4
# You pick up the peppershaker 1 from the cabinet 4.
# > think: Now I have the second peppershaker (1). Next, I need to put it in/on the drawer. I can check which drawer to use, starting with drawer 1.
# OK.
# > go to drawer 1
# The drawer 1 is open. In it, you see a creditcard 1, a peppershaker 2, and a spatula 1.
# > think: The drawer 1 already contains the first peppershaker (2) and has space for more items. I can put the second peppershaker (1) in/on drawer 1 as well.
# OK.
# > put peppershaker 1 in/on drawer 1
# You put the peppershaker 1 in/on the drawer 1."""


"""
任务描述：
You are in the middle of a room. Looking quickly around you, you see a fridge 1, a cabinet 1, a countertop 1, a toaster 1, 
a coffeemachine 1, a cabinet 2, a countertop 2, a stoveburner 1, a stoveburner 2, a cabinet 3, a cabinet 4, a microwave 1, 
a countertop 3, a sink 1, a sink 2, a shelf 1, a shelf 2, a drawer 1, a drawer 2, a drawer 3, a shelf 3, 
a stoveburner 3, a stoveburner 4, and a garbagecan 1.
Your task is to: put two peppershaker in drawer
"""



# trail0耗尽步骤失败：
text = """
> go to shelf 1
> go to shelf 2
> go to shelf 3
> go to countertop 1
> go to countertop 2
> go to countertop 3
> go to cabinet 1
On the cabinet 1, you see a dishsponge 1, a glassbottle 1, and a peppershaker 2.
> think: Now I find the first peppershaker (2). Next, I need to take it.
OK.
> open cabinet 1
> take peppershaker 2 from cabinet 1
You pick up the peppershaker 2 from the cabinet 1.
> think: Now I have the first peppershaker (2). Next, I need to put it in/on the drawer. I can check which drawer to use, starting with drawer 1.
OK.
> go to drawer 2
The drawer 1 is closed.
> open drawer 2
You open the drawer 2. The drawer 1 is open. In it, you see a creditcard 1, and a spatula 1.
> think: The drawer 2 has space for more items. I can put the first peppershaker (2) in/on drawer 2.
OK.
> put peppershaker 2 in/on drawer 2
You put the peppershaker 2 in/on the drawer 2.
> think: Now I have put the first peppershaker in/on drawer 2. Next, I need to find the second peppershaker. I can check the remaining cabinets, countertops, drawers, and shelves. I will start with cabinet 2.
OK.
> go to cabinet 2
The cabinet 2 is closed.
> open cabinet 2
You open the cabinet 2. The cabinet 2 is open. In it, you see nothing.
> go to cabinet 3
On the cabinet 3, you see a plate 1.
> go to cabinet 4
On the cabinet 4, you see a peppershaker 1.
> open cabinet 4
"""

#
# # trail1 reflexion, 反思幻觉，go to cabinet 2找 peppershaker1，失败
# text = """
# > go to shelf 1
# > go to shelf 2
# > go to shelf 3
# > go to countertop 1
# > go to countertop 2
# > go to countertop 3
# > go to cabinet 1
# > open cabinet 1
# On the cabinet 1, you see a dishsponge 1, a glassbottle 1, and a peppershaker 2.
# > think: Now I find the first peppershaker (2). Next, I need to take it.
# OK.
# > take peppershaker 2 from cabinet 1
# You pick up the peppershaker 2 from the cabinet 1.
# > think: Now I have the first peppershaker (2). Next, I need to put it in/on the drawer. I can check which drawer to use, starting with drawer 1.
# OK.
# > go to drawer 2
# The drawer 1 is closed.
# > open drawer 2
# You open the drawer 1. The drawer 1 is open. In it, you see a creditcard 1, and a spatula 1.
# > think: The drawer 1 has space for more items. I can put the first peppershaker (2) in/on drawer 1.
# OK.
# > put peppershaker 2 in/on drawer 2
# You put the peppershaker 2 in/on the drawer 2.
# > think: Now I have put the first peppershaker in/on drawer 1. Next, I need to find the second peppershaker. I can check the remaining cabinets, countertops, drawers, and shelves. I will start with cabinet 2.
# OK.
# > go to cabinet 2
# The cabinet 2 is closed.
# > open cabinet 2
# You open the cabinet 2. The cabinet 2 is open. In it, you see nothing.
# > open cabinet 2
# > open cabinet 2
# > open cabinet 2
# """


#
# # trail1 FCRF, 保留经验，直接到cabinet 1拿peppershaker2、放入drawer1; 直接到cabinet 4拿pappershaker1，但是放入了srawer2
# text = """
# > go to cabinet 1
# On the cabinet 1, you see a dishsponge 1, a glassbottle 1, and a peppershaker 2.
# > think: Now I find the first peppershaker (2). Next, I need to take it.
# OK.
# > take peppershaker 2 from cabinet 1
# You pick up the peppershaker 2 from the cabinet 1.
# > think: Now I have the first peppershaker (2). Next, I need to put it in/on the drawer. I can check which drawer to use, starting with drawer 1.
# OK.
# > go to drawer 2
# The drawer 2 is closed.
# > open drawer 2
# You open the drawer 2. The drawer 2 is open. In it, you see a creditcard 1, and a spatula 1.
# > think: The drawer 2 has space for more items. I can put the first peppershaker (2) in/on drawer 2.
# OK.
# > put peppershaker 2 in/on drawer 2
# You put the peppershaker 2 in/on the drawer 2.
# > think: Now I have put the first peppershaker in/on drawer 1. Next, I need to find the second peppershaker. I can check the remaining cabinets, countertops, drawers, and shelves. I will start with cabinet 2.
# OK.
# > go to cabinet 4
# On the cabinet 4, you see a peppershaker 1.
# > take peppershaker 1 from cabinet 4
# You pick up the peppershaker 1 from the cabinet 4.
# > go to drawer 1
# > open drawer 1
# > put peppershaker 1 in/on drawer 1
# """


#
# # trail2 reflexion, 反思幻觉，go to cabinet 1找 peppershaker1，失败
# text = """
# > go to shelf 1
# > go to shelf 2
# > go to shelf 3
# > go to countertop 1
# > go to countertop 2
# > go to countertop 3
# > go to cabinet 1
# On the cabinet 1, you see a dishsponge 1, a glassbottle 1, and a peppershaker 2.
# > think: Now I find the first peppershaker (2). Next, I need to take it.
# OK.
# > take peppershaker 2 from cabinet 1
# You pick up the peppershaker 2 from the cabinet 1.
# > think: Now I have the first peppershaker (2). Next, I need to put it in/on the drawer. I can check which drawer to use, starting with drawer 1.
# OK.
# > go to drawer 1
# The drawer 1 is closed.
# > open drawer 1
# You open the drawer 1. The drawer 1 is open. In it, you see a creditcard 1, and a spatula 1.
# > think: The drawer 1 has space for more items. I can put the first peppershaker (2) in/on drawer 1.
# OK.
# > put peppershaker 2 in/on drawer 1
# You put the peppershaker 2 in/on the drawer 1.
# > think: Now I have put the first peppershaker in/on drawer 1. Next, I need to find the second peppershaker. I can check the remaining cabinets, countertops, drawers, and shelves. I will start with cabinet 2.
# OK.
# > go to cabinet 2
# The cabinet 2 is closed.
# > open cabinet 2
# You open the cabinet 2. The cabinet 2 is open. In it, you see nothing.
# > go to cabinet 3
# On the cabinet 3, you see a plate 1.
# > go to cabinet 4
# On the cabinet 4, you see a peppershaker 1.
# """

#
# # trail2 FCRF, 保留经验，直接到cabinet 1拿peppershaker2、放入drawer1; 直接到cabinet 4拿pappershaker1，思考lesson:应该放进同一个drawer，放入drawer1，成功
# text = """
# > go to cabinet 1
# On the cabinet 1, you see a dishsponge 1, a glassbottle 1, and a peppershaker 2.
# > think: Now I find the first peppershaker (2). Next, I need to take it.
# OK.
# > open cabinet 1
# > take peppershaker 5 from cabinet 1
# You pick up the peppershaker 2 from the cabinet 1.
# > think: Now I have the first peppershaker (2). Next, I need to put it in/on the drawer. I can check which drawer to use, starting with drawer 1.
# OK.
# > go to drawer 2
# The drawer 1 is closed.
# > open drawer 2
# You open the drawer 2. The drawer 2 is open. In it, you see a creditcard 1, and a spatula 1.
# > think: The drawer 2 has space for more items. I can put the first peppershaker (2) in/on drawer 2.
# OK.
# > put peppershaker 2 in/on drawer 2
# You put the peppershaker 2 in/on the drawer 2.
# > think: Now I have put the first peppershaker in/on drawer 1. Next, I need to find the second peppershaker. I can check the remaining cabinets, countertops, drawers, and shelves. I will start with cabinet 2.
# OK.
# > go to cabinet 4
# On the cabinet 4, you see a peppershaker 1.
# > open cabinet 4
# > take peppershaker 1 from cabinet 4
# You pick up the peppershaker 1 from the cabinet 4.
# > go to drawer 2
# > put peppershaker 1 in/on drawer 2
# """


action_ls = re.findall(r'^> (.+)', text, re.MULTILINE)
print("\n\n", action_ls)
# exit()

openai.api_key = os.environ["OPENAI_API_KEY"]
FOLDER = './prompts'
PROMPT_FILE = 'alfworld_3prompts.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

def llm(prompt: str, model: Model, stop: List[str] = ["\n"]):
    try:
        cur_try = 0
        while cur_try < 6:
            if model.startswith('openai'):
                text = get_chat(prompt=prompt, model=model, temperature=cur_try * 0.2, stop_strs=stop)
            # elif model == "text-davinci-003":
            else:
                text = get_completion(prompt=prompt, temperature=cur_try * 0.2, stop_strs=stop)
            # else:
            #     #text = get_chat(prompt=prompt, model=model, temperature=cur_try * 0.2, stop_strs=stop)
            #     text = g4f_chat(prompt=prompt, model=model, temperature=cur_try * 0.2, stop_strs=stop)
            # dumb way to do this
            if len(text.strip()) >= 5:
                return text
            cur_try += 1
        return ""
    except Exception as e:
        print(prompt)
        print(e)
        import sys
        sys.exit(1)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

def modify_action(action):
  """
  将字符串中的“in”修改为“in/on”。

  Args:
    action: 一个string类型的短句

  Returns:
    modified_action: 修改后的短句
  """

  words = action.split()
  modified_action = []
  for word in words:
    if word == "in":
      modified_action.append("in/on")
    elif word == "on":
      modified_action.append("in/on")
    else:
      modified_action.append(word)
  return " ".join(modified_action)

def alfworld_run(env, base_prompt, memory: List[str], to_print=True, ob='', model: Model = "text-davinci-003") -> Tuple[EnvironmentHistory, bool]:
    import numpy as np

    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, ob, memory, [])
    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()
    cur_step = 0
    print(111)
    while cur_step < 49:
    # + ">"
        action = action_ls[cur_step]#llm(str(env_history) + ">", stop=['\n'], model=model).strip()
        import time
        time.sleep(1)
 
        action = action.replace(">", "")
        if not action.startswith("turn"):
            action = modify_action(action)
        env_history.add("action", action)
        observation, reward, done, info = env.step([action])
        #print(dir(env.envs[0].env))
        # frames = None#env.get_frames()
        frames = env.get_frames()
        # frame = np.float32(frame)
        # print(frame)
        # 保存图片
        if frames is not None:
            for frame in frames:
                # import numpy as np
                frame = np.float32(frame)
                nameLs = ob.split(':')[-1].split()
                name = "_".join(nameLs)
                image_path = os.path.join("/home/syf/PycharmProjects/reflexion/alfworld_runs/0224_for_video/result_pics", f'{name}_trial{len(memory):04d}_{cur_step:04d}.png')
                from PIL import Image
                # image = Image.fromarray(frame)  # 报错TypeError: Cannot handle this data type: (1, 1, 3), <f4，解决
                image = Image.fromarray(np.uint8(frame))
                image.save(image_path)
            print(f"Saved {image_path}")

        # 0306 syf添加，存储视频
        import cv2
        import numpy as np

        # # 假设frames是通过env.get_frames()获取的帧
        # frames = env.get_frames()  # 这里获取帧，假设返回的是一个列表或生成器

        # <class 'numpy.ndarray'>
        print("frames type: ", type(frames))
        print("frame长度： ", len(frames))

        # 获取帧的尺寸
        frame_height, frame_width, _ = frames[0].shape

        # 设置视频编写器，指定保存的视频文件路径和格式
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 或者 'MP4V' 或 'MJPG'
        out = cv2.VideoWriter('/home/syf/PycharmProjects/reflexion/alfworld_runs/0224_for_video/result_video/output_video.mp4', fourcc, 30.0, (frame_width, frame_height))  # 30.0是FPS

        # 有问题只能存下来一帧的gpt写法

        # 先读取已有视频
        existing_video = "/home/syf/PycharmProjects/reflexion/alfworld_runs/0224_for_video/result_video/output_video.mp4"
        cap = cv2.VideoCapture(existing_video)
        # 读取所有旧帧
        all_frames = []
        while cap.isOpened():
            ret, old_frame = cap.read()
            if not ret:
                break
            all_frames.append(old_frame)
        cap.release()

        # 遍历帧并写入视频文件
        if frames is not None:
            for frame in frames:
                # 需要确保帧的颜色通道顺序是BGR（OpenCV默认顺序）
                if frame.shape[2] == 3:  # 检查是否是彩色图像
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 如果是RGB格式，转换为BGR
                else:
                    frame_bgr = frame

                all_frames.append(frame_bgr)

        for frame in all_frames:
            # 将帧写入视频文件
            out.write(frame)

        # # syf修改写法
        # # 遍历帧并写入视频文件
        # for frame in frames:
        #     # 需要确保帧的颜色通道顺序是BGR（OpenCV默认顺序）
        #     if frame.shape[2] == 3:  # 检查是否是彩色图像
        #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 如果是RGB格式，转换为BGR
        #     else:
        #         frame = frame
        #
        # # 将帧写入视频文件
        # out.write(frames)
        #
        # # 释放视频编写器资源
        # out.release()
        # cv2.destroyAllWindows()
        print("视频保存完毕.")

        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        env_history.add("observation", observation)
        if to_print:
            print(f'> {action}\n{observation}')
            sys.stdout.flush()
        if done:
            return env_history, True
        elif env_history.check_is_exhausted():
            return env_history, False
        cur_step += 1
    return env_history, False

# PREFIXES = {
#     'pick_and_place': 'put',
#     'pick_clean_then_place': 'clean',
#     'pick_heat_then_place': 'heat',
#     'pick_cool_then_place': 'cool',
#     'look_at_obj': 'examine',
#     'pick_two_obj': 'puttwo'
# }
PREFIXES = {
     'pick_and_place': 'put',
     'pick_clean_then_place': 'clean',
     'pick_heat_then_place': 'heat',
     'pick_cool_then_place': 'cool',
     'look_at_obj': 'examine',
     'pick_two_obj': 'puttwo'
}



def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        model: Model,
    ) -> List[Dict[str, Any]]:
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)

    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"

    import random
    import numpy as np

    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    seed = 42  # 你可以根据需要选择不同的种子
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    
    
    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    for z, env_config in enumerate(env_configs):


        # # ------------原写法------------
        # flag = True
        # while flag:
        #
        # # 0224 syf修改，只读自己想可视化的env num
        # # env81: put two keychain in safe; env3: find two peppershaker and put them in drawer.
        # # if z == 3:  # /home/syf/PycharmProjects/reflexion/alfworld_runs/reflexion_run_logs/134envs_trail0/gpt-4mini/ReAct/trail2-3/trail1 env 81
        #
        #     ob, info = env.reset()
        #     ob = '\n'.join(ob[0].split('\n\n')[1:])
        #     import re
        #
        #     # 用正则表达式将 "Your task is to:" 替换为新的内容
        #     ob = re.sub(
        #         r"Your task is to:",  # 匹配 "Your task is to:"
        #         r"Your task is to:",  # 替换为新的内容
        #         ob
        #     )
        #
        #     # print("\nob: ", ob, "\nob end \n")
        #
        #     name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        #
        #     print(f"using {name}")



        # ------------0224 syf修改，只读自己想可视化的env num------------
        # env81: put two keychain in safe; env3: find two peppershaker and put them in drawer.
        # if z == 3:  # /home/syf/PycharmProjects/reflexion/alfworld_runs/reflexion_run_logs/134envs_trail0/gpt-4mini/ReAct/trail2-3/trail1 env 81

        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        import re

        # 用正则表达式将 "Your task is to:" 替换为新的内容
        ob = re.sub(
            r"Your task is to:",  # 匹配 "Your task is to:"
            r"Your task is to:",  # 替换为新的内容
            ob
        )

        # print("\nob: ", ob, "\nob end \n")

        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

        print(f"using {name}")
        print("当前id: ", z, "\n")

        # 指定自己要跑video的任务
        # if name == "valid_unseen/pick_two_obj_and_place-PepperShaker-None-Drawer-10":
        if z == 66:

        # ------------syf修改部分结束----------------

            # syf注释掉
            # if env_config["is_success"]:
            #     num_successes += 1
            #
            #     # log to world log
            #     with open(world_log_path, 'a') as wf:
            #         wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            #     with open(trial_log_path, 'a') as wf:
            #         wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            #     break

            for i, (k, v) in enumerate(PREFIXES.items()):
                if name.startswith("valid_unseen/"+k):
                    env_configs[z]["v"] = v
                    base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0']
                    final_env_history, is_success = alfworld_run(env, base_prompt, env_config["memory"] if use_memory else [], to_print=True, ob=ob, model=model)

                    # update env config
                    if is_success:
                        status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
                        env_configs[z]['is_success'] = True
                        num_successes += 1
                        num_additional_successes += 1
                    else:
                        status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

                    # log to world log
                    with open(world_log_path, 'a') as f:
                        f.write(status_str + '\n')

                    # log env results to trial log
                    with open(trial_log_path, 'a') as wf:
                        wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

                    flag = False
    # env.envs[0].env.generate_vedio()
# close environment object
    env.close()

    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs


