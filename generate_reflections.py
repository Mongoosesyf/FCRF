# syf搭建的Constructivism Self- Reflection方法框架
# 核心：分为mentor和reasoner模型，reasoner负责推理;
# mentor提醒reasoner保留正确部分轨迹经验(task specific)，并收集错误经验维护pool，每次调用检查错误
from numbers import Complex

from utils import get_completion

from typing import List, Dict, Any
import json

import os

with open("./reflexion_few_shot_examples.txt", 'r') as f:
    FEW_SHOT_EXAMPLES = f.read()


def _get_scenario(s: str) -> str:
    """Parses the relevant scenario from the experience log."""
    return s.split("Here is the task:")[-1].strip()

# # syf 0107添加，读取倒数第二个trail，用于对比抽取改正方法pool
# 这么写不对，这样读取的应该是上一个env而不是上一个trail的
# def _get_last_scenario(s: str) -> str:
#     """Parses the relevant scenario from the experience log."""
#     return s.split("Here is the task:")[-2].strip()


def _generate_reflection_query(log_str: str, memory: List[str]) -> str:
    # syf add注释: 此函数是文本构建拼接生成reflexion需要的prompt
    """Allows the Agent to reflect upon a past experience."""
    scenario: str = _get_scenario(log_str)
    query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. 
    You were unsuccessful in completing the task. Do not summarize your environment, 
    but rather think about the strategy and path you took to attempt to complete the task. 
    Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. 
    For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. 
    You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{FEW_SHOT_EXAMPLES}

{scenario}"""

    if len(memory) > 0:
        query += '\n\nPlans from past attempts:\n'
        for i, m in enumerate(memory):
            query += f'Trial #{i}: {m}\n'

    query += '\n\nNew plan:'
    return query


# syf添加，mentor引导reasoner总结保留正确部分轨迹经验 (task specific, 针对性)
# def _generate_experience_query(log_str: str, memory: List[str]) -> str:
def _generate_experience_query(log_str: str, memory: List[str], v: str) -> str:
    # syf add: 此函数是文本构建拼接生成reflexion需要的prompt

    # 读json写法
    with open("constructivism_prompts/my_experience_prompt.json", 'r') as f:
        experience = json.load(f)

    # # 读txt写法
    # with open("constructivism_prompts/my_experience_prompt.txt", 'r') as f:
    #     experience = f.read()

    scenario: str = _get_scenario(log_str)
    # print("scenario111: ", scenario)
    # scenario = get_association(scenario)
    query: str = f"""  
    
    You will be given the history of a past experience in which you were placed in an environment and given a task to complete. 
    You were unsuccessful in completing the task.
    Now by aligning the task goal, summary the valuable actions you have gained in your past experience.

    Here are two examples:
    =================The first example=====================
    {experience[f'{v}_0']}
    
    =================The second example=====================
    {experience[f'{v}_1']}
    
    Here is the history of the past experience and your task:
    {scenario}
    
    Strictly follow the format of example, summarize the valuable actions directly in concise language, without adding your own divergent thoughts.

    """

    query += '\n\nValuable experience summarization:'
    # query = get_association(query)
    return query


# syf添加，mentor持续维护外部pool，引导reasoner针对性解决错误 (整个scence compatible, 可泛化)
def _generate_lesson_query(log_str: str, memory: List[str], v: str) -> str:
    # syf add: 此函数是文本构建拼接生成reflexion需要的prompt，生成的prompt再输入get_completion(this prompt)调用LLM生成reflection
    # with open("constructivism_prompts/mentor_lesson_pool.txt", 'r') as f1:  # 人工编写的pool
    with open("constructivism_prompts/LLM_pool.txt", 'r') as f1:  # LLM抽取的pool
        mentor_lesson_pool = f1.read()
    with open("constructivism_prompts/my_lesson_prompt.txt", 'r') as f2:
        lesson = f2.read()

    scenario: str = _get_scenario(log_str)
    # scenario = get_association(scenario)
    query: str = f"""
    
    You will be given the history of a past experience in which you were placed in an environment and given a task to complete. 
    You were unsuccessful in completing the task.
    Now to solve your mistake, I will give you a mentor lesson pool, in which listed typical constraints of your task environment.
    Analyze which tip in the mentor lesson pool might be the reason of your mistake, and then make failure lesson summarization refer to the form of examples given later. 
    The existed tips in the mentor lesson pool can be your inspirations, but don't be limited by them. Summary failure lessons according to your specific task experience.
    Pay attention, only choose ONE tip in the pool fits your task type, and only choose it when you think it might be reason of your mistake. 

    Here is the mentor lesson pool:
    {mentor_lesson_pool}
    
    Here is your task type, remember to extract one through tips in the pool which precisely contains your task type at the beginning:
    {v}

    Here are two examples:
    {lesson}

    Here is the history of the past experience and your task:
    {scenario}

    Strictly follow the format of example, directly extract one lesson from the pool in one sentence, without adding your own divergent thoughts.

    """

    query += '\n\nFailure lesson summarization:'
    return query


# syf添加，通过成功experience和失败lesson组合引导新的plan
def _generate_plan_query(log_str: str, memory: List[str], experience, lesson) -> str:
    """Allows the Agent to reflect upon a past experience."""
    with open("constructivism_prompts/plan_examples.txt", 'r') as f:
        plan_example = f.read()

    scenario: str = _get_scenario(log_str)
    # scenario = del_think(scenario)
    query: str = f"""
    
    You will be given the history of a past experience in which you were placed in an environment and given a task to complete. 
    You were unsuccessful in completing the task.
    Now I will give you valuable experience summarization and failure lesson summarization based on your previous experience. 
    Please create a plan for the next attempt with reference to specific actions that you should have taken, incorporating insights from these two analyses. 
    You need to output in the format of the example without any additional redundant content.
    I will give you some examples to help you better understand how to generate plan.
    
    Here are two examples:
    {plan_example}
    
    Here is the history you need for generating plan:
    {scenario}
    
    Here is the valuable experience you need for generating plan:
    {experience}
    
    Here is the failed lesson you need for generating plan:
    {lesson}
    """

    query += '\n\nPlan:'
    return query

# syf1224修改，只利用experience，生成reflexion架构的plan，不注入pool信息，鼓励自由探索
def _generate_exp_rfl_plan_query(log_str: str, memory: List[str], experience) -> str:
    """Allows the Agent to reflect upon a past experience."""
    with open("reflexion_with_exp_few_shot_examples.txt", 'r') as f:
        exp_rfl_example = f.read()
    scenario: str = _get_scenario(log_str)

    query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. 
    You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. 
    I'll give you your past task trail and valuable experience through the trail. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. 
    For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. 
    You will need this later when you are solving the same task. Give your plan after "Plan". 

    Here are two examples:
    {exp_rfl_example}

    Here is the history of the past experience and your task:
    {scenario}
    
    Here is the valuable experience:
    {experience}

    """

    if len(memory) > 0:
        query += '\n\nPlans from past attempts:\n'
        for i, m in enumerate(memory):
            query += f'Trial #{i}: {m}\n'

    query += '\n\nNew plan:'
    return query


# syf 1229添加, LLM在只使用experience的前 k trails抽取经验约束，填充构造pool，用于后续调用
def Update_Lesson_Pool(log_str: str, last_trail_log_str: str, v: str, pool_file_path = "./constructivism_prompts/LLM_pool.txt"):

    scenario: str = _get_scenario(log_str)
    # last_trail_scenario: str = _get_last_scenario(log_str)
    last_trail_scenario: str = _get_scenario(last_trail_log_str)

    with open("constructivism_prompts/lesson_examples.txt", 'r') as f:
        lesson_example = f.read()

    # 旧写法，直接应用success的抽取
    # prompt: str = f"""
    # You will be given the history of a past experience in which you were placed in an environment and given a task to complete.
    # You have successfully completed the task, and now you have to summarize lessons about the task environment through your past task trail.
    # Please focus on the inherent constrains of the environment and the interact logic of task objects, which you can reuse in other similar tasks, for example, you have to go to the desklamp before using it, and you should heat using the microwave.
    # Please summarize the most useful lesson using only one sentence.
    # I will give you some examples to help you better understand how to summarize lessons.
    #
    # Here are two examples:
    # {lesson_example}
    #
    # Here is the history of the past experience and your task:
    # {scenario}
    #
    # """

    # 新写法，用fail->success，即改对的抽取
    prompt: str = f"""
    You will be given the history of a past experience in which you were placed in an environment and given a task to complete. 
    You kept failing in the past attempts, but exactly succeed in the latest attempt.
    Now you have to summarize lessons especially base on how did you correct the trail from failure to success, through your two past task trails.
    Please focus on the inherent constrains of the environment and the interact logic of task objects, which you can reuse in other similar tasks, for example, you have to go to the desklamp before using it, and you should heat using the microwave.
    Please summarize the most useful lesson using only one sentence.
    I will give you some examples to help you better understand how to summarize lessons.

    Here are two examples:
    {lesson_example}

    Here is your past failed trail and your task:
    {last_trail_scenario}

    Here is your latest successful trail for the same task:
    {scenario}

    """

    current_env_lesson = get_completion(prompt)

    # 去除tips开头的\n和"Lesson summarize:"字段, 只保留lesson文本
    current_env_lesson = current_env_lesson.strip()
    if current_env_lesson.startswith('Lesson summarize:'):
        current_env_lesson = current_env_lesson.lstrip("Lesson summarize:")

    # 根据任务类型拼接索引格式
    current_env_tip = '[' + v + '] ： ' + str(current_env_lesson) + '\n'
    with open (pool_file_path, 'a') as f:
        f.write(current_env_tip)


# 法一. reflexion原生更新方法，原名update_memory
def reflexion_orig_update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()

    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        start_env_num = -1
        if start_env_num <= i:
            if not env['is_success'] and not env['skip']:
                if len(env['memory']) > 3:
                    memory: List[str] = env['memory'][-3:]
                else:
                    memory: List[str] = env['memory']
                reflection_query: str = _generate_reflection_query(env_logs[i], memory)
                reflection: str = get_completion(
                    reflection_query)  # type: ignore  # syf add: 这一步是用上一步拼成的reflx prompt调用gpt生成反思内容（？
                env_configs[i]['memory'] += [reflection]

    return env_configs


# 法二. 自己的成功experience + 失败lesson + 综合plan更新方法
# syf 修改，自己的_update_memory
def update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()

    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']

            # 法一. reflexion原生更新方法
            # reflection_query: str = _generate_reflection_query(env_logs[i], memory)
            # reflection: str = get_completion(
            #     reflection_query)  # type: ignore  # syf add: 这一步是用上一步拼成的reflx prompt调用gpt生成反思内容（？
            # env_configs[i]['memory'] += [reflection]

            # 法二. 自己的成功experience + 失败lesson + 综合plan更新方法
            # experience_query = _generate_experience_query(env_logs[i], memory)  # 不判别任务类型，直接调用2个example的版本
            experience_query = _generate_experience_query(env_logs[i], memory, env['v'])  # 加类型参数，再针对调用2个同类example的版本
            experience = get_completion(experience_query)
            if not experience.startswith('Valuable experience summarization'):
                experience = "Valuable experience summarization:\n" + experience

            lesson_query = _generate_lesson_query(env_logs[i], memory, env['v'])
            lesson = get_completion(lesson_query)
            if not lesson.startswith('Failure lesson summarization'):
                lesson = "Failure lesson summarization:\n" + lesson

            total_plan_query = _generate_plan_query(env_logs[i], memory, experience, lesson)
            total_plan = get_completion(total_plan_query)
            if not total_plan.startswith('Plan'):
                total_plan = "Plan:" + total_plan

            total_inference: str = experience + '\n\n' + lesson + '\n\n' + total_plan

            # causalinference_query = _generate_counterfactual_query(env_logs[i], memory, env['v'])
            # causalinference = router_completion(causalinference_query)
            print("##########\n")
            print(total_inference)
            env_configs[i]['memory'] += [total_inference]

    return env_configs


# 法三. 只使用experience, plan由reflexion架构量级的prompt引导生成，鼓励先自由探索，一定trails后再调update_memory注入pool
# 只使用exp进行反思，且积累pool。积累的素材为上一轮fail这一轮success的，即改对的envs
def update_exp_rfl_memory(trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()

    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):

        # syf 1229添加，利用成功任务在线更新维护pool
        # if env['is_success'] and not env['skip']:  # 旧判别条件：这一轮success。
        if env['additional_success'] and not env['skip']:  # 新判别条件：这一轮success且上一轮fail（是addi succ，改对的）
            Update_Lesson_Pool(env_logs[i], env['v'])

        # syf添加部分结束，以下为原有部分
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']

            # 经验
            experience_query = _generate_experience_query(env_logs[i], memory, env['v'])  # 加类型参数，再针对调用2个同类example的版本
            experience = get_completion(experience_query)
            if not experience.startswith('Valuable experience summarization'):
                experience = "Valuable experience summarization:\n" + experience

            # 结合经验进行自由简单reflexion
            plan_query = _generate_exp_rfl_plan_query(env_logs[i], memory, experience)
            plan = get_completion(plan_query)
            if not plan.startswith('New plan'):
                plan = "New plan:" + plan

            total_inference: str = experience + '\n\n' + plan
            print("##########\n")
            print(total_inference)
            env_configs[i]['memory'] += [total_inference]


    return env_configs

'''
PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}
'''
# syf0122添加，灵活反思的前置函数，评估反思难度，返回结果为只使用exp简单探索的trail数量
def eval_refl_difficulty(num_trials, task_str, i, env_configs: List[Dict[str, Any]]):
    # class Complex():
    #     put = [2, 1]
    #     clean = [3, 2]
    #     heat = [3, 2]
    #     cool = [3, 2]
    #     examine = [2, 1]
    #     puttwo = [3, 1]

    # '类型': [交互物品数量，交互operation数]
    complex_dict = {'put': [1, 1],  # 3 simple
                    'clean': [2, 2],  # 1
                    'heat' : [2, 2],  # 1
                    'cool' : [2, 2],  # 1
                    'examine': [2, 1],  # 2
                    'puttwo': [3, 1]  # 1
                    }

    max_complex = 5
    # task_type = env_configs['v']
    # current_task_complex = int(complex_dict[task_type][0]) + int(complex_dict[task_type][1])

    # value_list = [v for k, v in complex_dict.items() if k == env_configs[i]['v']]

    for k, v in complex_dict.items():
        if k == env_configs[i]['v']:
            value_list = v

    # print(value_list)
    current_task_complex = int(value_list[0]) + int(value_list[1])

    simple_trail_num = int((1 - (current_task_complex / max_complex)) * 5)  # 任务越复杂，做exp+lesson比例越多；否则做简单exp探索反思
    if simple_trail_num < 1:
        simple_trail_num = 1  # 至少使用一轮simple反思

    return simple_trail_num


# syf0122添加，灵活评估+选择任务反思难度。
# 目前评估方法：任务类型+交互物品数套公式评级，简单/普通/困难。
# 目前写法是每个trail评估一次，所以后面也可以加入上个trail的反思，做软（反思内容）-硬（解析任务难度）加权评估
def flexible_update_memory(num_trials, trial_idx, trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()
    print("trial_log_path: ", trial_log_path)

    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')

    # syf添加，记录下last trail log用于pool对比生成
    if trial_idx > 1:
        last_trial_log_path = 'reflexion_run_logs/trail_' + str(trial_idx-2) + '.log'
        # with open(last_trial_log_path, 'r') as f:
        with open("./reflexion_run_logs/trial_0.log", 'r') as f:
            last_trial_full_log: str = f.read()
        last_trial_env_logs: List[str] = last_trial_full_log.split('#####\n\n#####')

    start_env_num = 1000

    # syf 0208添加一轮循环，先把反思强度评估和这一轮的增量success lesson pool做好
    if start_env_num == -1:  # 之在最开始初始化时跑一次，中断继续时不需要再跑
    # if 1>0:  # 之在最开始初始化时跑一次，中断继续时不需要再跑
        for i, env in enumerate(env_configs):

            # 评估反思强度
            if env_configs[i]['simple_rfl_num'] == 0:  # 还没评估过任务难度，则评估，赋值simple_rfl_num
                # scenario: str = _get_scenario(env_logs[i])  # 得到Here is your task后的整个上一trail序列
                # print("i=", i)
                # print("env_logs: ", env_logs[1])
                # print("scenario: ", scenario)
                # task_str = scenario.split("Your task is to:")[1]  # 保留：task描述+\n+后面一堆actions
                # task_str = task_str.split('\n')[0].strip()  # 保留：task描述，如put a clean knife in countertop.
                task_str = " "
                simple_trail_num = eval_refl_difficulty(num_trials, task_str, i, env_configs)
                env_configs[i]['simple_rfl_num'] = simple_trail_num  # 评估完毕，写入key

            # # 做这一轮的lesson pool更新, 只有在trail>1时候做，因为trail0没有增量改正的错误轨迹
            # if trial_idx > 1 and env['additional_success'] and not env['skip']:  # 新判别条件：这一轮success且上一轮fail（是addi succ，改对的）
            #     Update_Lesson_Pool(env_logs[i], last_trial_env_logs[i], env['v'])

        # 更新自己的env_config_json
        uptodate_env_config_name = "reflexion_run_logs/trail" + str(trial_idx) + "_uptodate_env_config.json"
        with open(uptodate_env_config_name, 'w') as wf:
            json.dump(env_configs, wf, indent=4)

    for i, env in enumerate(env_configs):
        if start_env_num <= i:
        # if 0 < 1:
        # # 赋值反思强度和做POOL，移到前面了
        #     if env_configs[i]['simple_rfl_num'] == 0:  # 还没评估过任务难度，则评估，赋值simple_rfl_num
        #         scenario: str = _get_scenario(env_logs[i])  # 得到Here is your task后的整个上一trail序列
        #         task_str = scenario.split("Your task is to:")[1]  # 保留：task描述+\n+后面一堆actions
        #         task_str = task_str.split('\n')[0].strip()  # 保留：task描述，如put a clean knife in countertop.
        #         simple_trail_num = eval_refl_difficulty(num_trials, task_str, i, env_configs)
        #         env_configs[i]['simple_rfl_num'] = simple_trail_num  # 评估完毕，写入key

            # if trial_idx <= env_configs[i]['simple_rfl_num']:
            if 1<0:
                # exp简单反思
                # return update_exp_rfl_memory(trial_log_path, env_configs)

                # # syf 1229添加，利用成功任务在线更新维护pool
                # # if env['is_success'] and not env['skip']:  # 旧判别条件：这一轮success。
                # if env['additional_success'] and not env['skip']:  # 新判别条件：这一轮success且上一轮fail（是addi succ，改对的）
                #     Update_Lesson_Pool(env_logs[i], env['v'])

                # syf添加部分结束，以下为原有部分
                # if unsolved, get reflection and update env config
                if not env['is_success'] and not env['skip']:
                    if len(env['memory']) > 3:
                        memory: List[str] = env['memory'][-3:]
                    else:
                        memory: List[str] = env['memory']

                    # 经验
                    experience_query = _generate_experience_query(env_logs[i], memory,
                                                                  env['v'])  # 加类型参数，再针对调用2个同类example的版本
                    experience = get_completion(experience_query)
                    if not experience.startswith('Valuable experience summarization'):
                        experience = "Valuable experience summarization:\n" + experience

                    # 结合经验进行自由简单reflexion
                    plan_query = _generate_exp_rfl_plan_query(env_logs[i], memory, experience)
                    plan = get_completion(plan_query)
                    if not plan.startswith('New plan'):
                        plan = "New plan:" + plan

                    total_inference: str = experience + '\n\n' + plan
                    print("##########\n")
                    print(total_inference)
                    env_configs[i]['memory'] += [total_inference]

            else:
                # CRFL 综合exp+lesson反思
                # return update_memory(trial_log_path, env_configs)

                # if unsolved, get reflection and update env config
                if not env['is_success'] and not env['skip']:
                    if len(env['memory']) > 3:
                        memory: List[str] = env['memory'][-3:]
                    else:
                        memory: List[str] = env['memory']

                    # 法一. reflexion原生更新方法
                    # reflection_query: str = _generate_reflection_query(env_logs[i], memory)
                    # reflection: str = get_completion(
                    #     reflection_query)  # type: ignore  # syf add: 这一步是用上一步拼成的reflx prompt调用gpt生成反思内容（？
                    # env_configs[i]['memory'] += [reflection]

                    # # 法二. 自己的成功experience + 失败lesson + 综合plan更新方法
                    # # experience_query = _generate_experience_query(env_logs[i], memory)  # 不判别任务类型，直接调用2个example的版本
                    # experience_query = _generate_experience_query(env_logs[i], memory, env['v'])  # 加类型参数，再针对调用2个同类example的版本
                    # experience = get_completion(experience_query)
                    # if not experience.startswith('Valuable experience summarization'):
                    #     experience = "Valuable experience summarization:\n" + experience
                    # lesson pool ablation
                    experience = "Valuable experience summarization:\n There is no experience yet."

                    lesson_query = _generate_lesson_query(env_logs[i], memory, env['v'])
                    lesson = get_completion(lesson_query)
                    if not lesson.startswith('Failure lesson summarization'):
                        lesson = "Failure lesson summarization:\n" + lesson

                    total_plan_query = _generate_plan_query(env_logs[i], memory, experience, lesson)
                    total_plan = get_completion(total_plan_query)
                    if not total_plan.startswith('Plan'):
                        total_plan = "Plan:" + total_plan

                    total_inference: str = experience + '\n\n' + lesson + '\n\n' + total_plan

                    # causalinference_query = _generate_counterfactual_query(env_logs[i], memory, env['v'])
                    # causalinference = router_completion(causalinference_query)
                    print("##########\n")
                    print(total_inference)
                    env_configs[i]['memory'] += [total_inference]

        # syf0207添加，每跑完一个env的reflection，都更新自己的env_config_json
        uptodate_env_config_name = "reflexion_run_logs/trail" + str(trial_idx) + "_uptodate_env_config.json"
        with open(uptodate_env_config_name, 'w') as wf:
            json.dump(env_configs, wf, indent=4)

    return env_configs