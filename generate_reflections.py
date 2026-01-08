# Constructivism Self- Reflection
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



def _generate_reflection_query(log_str: str, memory: List[str]) -> str:
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


def _generate_experience_query(log_str: str, memory: List[str], v: str) -> str:
    # syf add: 此函数是文本构建拼接生成reflexion需要的prompt

    with open("constructivism_prompts/my_experience_prompt.json", 'r') as f:
        experience = json.load(f)

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


def _generate_lesson_query(log_str: str, memory: List[str], v: str) -> str:
    with open("constructivism_prompts/LLM_pool.txt", 'r') as f1:  # LLM extracted pool
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


def Update_Lesson_Pool(log_str: str, last_trail_log_str: str, v: str, pool_file_path = "./constructivism_prompts/LLM_pool.txt"):

    scenario: str = _get_scenario(log_str)
    # last_trail_scenario: str = _get_last_scenario(log_str)
    last_trail_scenario: str = _get_scenario(last_trail_log_str)

    with open("constructivism_prompts/lesson_examples.txt", 'r') as f:
        lesson_example = f.read()

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

    current_env_lesson = current_env_lesson.strip()
    if current_env_lesson.startswith('Lesson summarize:'):
        current_env_lesson = current_env_lesson.lstrip("Lesson summarize:")

    current_env_tip = '[' + v + '] ： ' + str(current_env_lesson) + '\n'
    with open (pool_file_path, 'a') as f:
        f.write(current_env_tip)


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


            experience_query = _generate_experience_query(env_logs[i], memory, env['v'])
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


def update_exp_rfl_memory(trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()

    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):

        if env['additional_success'] and not env['skip']:
            Update_Lesson_Pool(env_logs[i], env['v'])

        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']

            experience_query = _generate_experience_query(env_logs[i], memory, env['v'])
            experience = get_completion(experience_query)
            if not experience.startswith('Valuable experience summarization'):
                experience = "Valuable experience summarization:\n" + experience

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
def eval_refl_difficulty(num_trials, task_str, i, env_configs: List[Dict[str, Any]]):

    #  'type': [interaction obj num，interaction operation num]
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

    simple_trail_num = int((1 - (current_task_complex / max_complex)) * 5)
    if simple_trail_num < 1:
        simple_trail_num = 1

    return simple_trail_num


def flexible_update_memory(num_trials, trial_idx, trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()
    print("trial_log_path: ", trial_log_path)

    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')

    if trial_idx > 1:
        last_trial_log_path = 'reflexion_run_logs/trail_' + str(trial_idx-2) + '.log'
        # with open(last_trial_log_path, 'r') as f:
        with open("./reflexion_run_logs/trial_0.log", 'r') as f:
            last_trial_full_log: str = f.read()
        last_trial_env_logs: List[str] = last_trial_full_log.split('#####\n\n#####')

    start_env_num = -1
    if start_env_num == -1:
        for i, env in enumerate(env_configs):

            if env_configs[i]['simple_rfl_num'] == 0:
                task_str = " "
                simple_trail_num = eval_refl_difficulty(num_trials, task_str, i, env_configs)
                env_configs[i]['simple_rfl_num'] = simple_trail_num  # 评估完毕，写入key

            # # update lesson pool更新, only when trail>1
            # if trial_idx > 1 and env['additional_success'] and not env['skip']:
            #     Update_Lesson_Pool(env_logs[i], last_trial_env_logs[i], env['v'])

        # 更新自己的env_config_json
        uptodate_env_config_name = "reflexion_run_logs/trail" + str(trial_idx) + "_uptodate_env_config.json"
        with open(uptodate_env_config_name, 'w') as wf:
            json.dump(env_configs, wf, indent=4)

    for i, env in enumerate(env_configs):
        if start_env_num <= i:

            # CRFL 综合exp+lesson反思
            # return update_memory(trial_log_path, env_configs)

            # if trial_idx <= env_configs[i]['simple_rfl_num']:

            # if unsolved, get reflection and update env config
            if not env['is_success'] and not env['skip']:
                if len(env['memory']) > 3:
                    memory: List[str] = env['memory'][-3:]
                else:
                    memory: List[str] = env['memory']

                # orig reflexion
                # reflection_query: str = _generate_reflection_query(env_logs[i], memory)
                # reflection: str = get_completion(
                #     reflection_query)  # type: ignore  # syf add: 这一步是用上一步拼成的reflx prompt调用gpt生成反思内容（？
                # env_configs[i]['memory'] += [reflection]

                # # FCRF
                # # experience_query = _generate_experience_query(env_logs[i], memory)
                # experience_query = _generate_experience_query(env_logs[i], memory, env['v'])
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


    return env_configs


