import os
import json
import argparse

from alfworld_trial import run_trial
from generate_reflections import update_memory
from generate_reflections import reflexion_orig_update_memory
from generate_reflections import update_exp_rfl_memory
from generate_reflections import Update_Lesson_Pool
from generate_reflections import flexible_update_memory
# from generate_reflections_orig import update_memory

from generate_reflections import eval_refl_difficulty

from typing import Any, List, Dict

import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, help="The number of trials to run")
    parser.add_argument("--num_envs", type=int, help="The number of environments per trial")
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--use_memory", action='store_true', help="Allow the Agent to use memory")
    parser.add_argument("--is_resume", action='store_true', help="To resume run")
    parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    parser.add_argument("--start_trial_num", type=int, help="If resume, the start trial num", default=0)
    parser.add_argument("--model", type=str, help="The model to use. One of `gpt-4`, `gpt-3.5-turbo`, or `text-davinci-003")

    args = parser.parse_args()

    assert args.num_trials > 0, "Number of trials should be positive"
    assert args.num_envs > 0, "Number of environments should be positive"

    return args

def main(args) -> None:
    if args.is_resume:
        if not os.path.exists(args.resume_dir):
            raise ValueError(f"Resume directory `{args.resume_dir}` does not exist")
        logging_dir = args.resume_dir

        # load environment configs
        # env_config_path: str = os.path.join(args.resume_dir, f'env_results_trial_{args.start_trial_num - 1}.json')  # 原写法
        env_config_path: str = "reflexion_run_logs/trail" + str(args.start_trial_num) + "_uptodate_env_config.json"  # syf0207修改，为中断继续，读自己存下来的
        if not os.path.exists(env_config_path):
            raise ValueError(f"Environment config file `{env_config_path}` does not exist")
        with open(env_config_path, 'r') as rf:
            env_configs: List[Dict[str, Any]] = json.load(rf)
    else:
        # Create the run directory
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)
        logging_dir = args.run_name

        # initialize environment configs
        # 'additional_success'键为syf添加，用来跟踪改对的envs，用于lesson pool构建;
        # 'simple_rfl_num'键为syf添加，用来进行灵活反思，评估此任务应该进行简单探索的trail数，默认为1
        env_configs: List[Dict[str, Any]] = []
        for i in range(args.num_envs):
            env_configs += [{
                'name': f'env_{i}',
                'memory': [],
                'is_success': False,
                'skip': False,
                'v': '',
                'additional_success': False,
                'simple_rfl_num': 0
            }]
    
    world_log_path: str = os.path.join(logging_dir, 'world.log')

    # print start status to user
    if args.is_resume:
        print(f"""
    -----
    Resuming run with the following parameters:
    Run name: {logging_dir}
    Number of trials: {args.num_trials}
    Number of environments: {args.num_envs}
    Use memory: {args.use_memory}
    Resume trial number: {args.start_trial_num}

    Sending all logs to `{args.run_name}`
    -----
    """)
    else:
        print(f"""
    -----
    Starting run with the following parameters:
    Run name: {logging_dir}
    Number of trials: {args.num_trials}
    Number of environments: {args.num_envs}
    Use memory: {args.use_memory}

    Sending all logs to `{args.run_name}`
    -----
    """)

    # run trials
    trial_idx = args.start_trial_num
    while trial_idx < args.num_trials:
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** Start Trial #{trial_idx} *****\n\n')

        # set paths to log files
        trial_log_path: str = os.path.join(args.run_name, f'trial_{trial_idx-1}.log')
        trial_env_configs_log_path: str = os.path.join(args.run_name, f'env_results_trial_{trial_idx-1}.json')
        # if os.path.exists(trial_log_path):
        #     open(trial_log_path, 'w').close()
        # if os.path.exists(trial_env_configs_log_path):
        #     open(trial_env_configs_log_path, 'w').close()

        # update memory if needed
        if args.use_memory:
            # # 原写法
            # env_configs: List[Dict[str, Any]] = update_memory(trial_log_path, env_configs)  # 自己的和orig reflexion都叫update memory

            # # 1223syf修改，k trails前先鼓励自由探索，之后再注入pool
            # if trial_idx <= 2 :
            #     env_configs: List[Dict[str, Any]] = reflexion_orig_update_memory(trial_log_path, env_configs)
            # else:
            #     env_configs: List[Dict[str, Any]] = update_memory(trial_log_path, env_configs)

            # # 1224修改，只通过reflexion_orig_update_memory跑reflexion
            # env_configs: List[Dict[str, Any]] = reflexion_orig_update_memory(trial_log_path, env_configs)

            # # 1224syf修改尝试2，k trails前先使用exp+rfl鼓励自由探索+积累lesson，之后再注入pool信息
            # if trial_idx <= 2 :
            #     env_configs: List[Dict[str, Any]] = update_exp_rfl_memory(trial_log_path, env_configs)
            # else:
            #     env_configs: List[Dict[str, Any]] = update_memory(trial_log_path, env_configs)

            # 0122syf修改，灵活评估+选择任务反思难度。
            # 目前评估方法：任务类型+交互物品数套公式评级，简单/普通/困难。
            # 目前写法是每个trail评估一次，所以后面也可以加入上个trail的反思，做软（反思内容）-硬（解析任务难度）加权评估
            # 逻辑是trail套env num
                env_configs: List[Dict[str, Any]] = flexible_update_memory(args.num_trials, trial_idx, trial_log_path, env_configs)


        # log env configs for trial（自行理解：把生成的反思写入上一轮的trail。env_json_0有1个反思，1有2个...类推）
        with open(trial_env_configs_log_path, 'w') as wf:
            json.dump(env_configs, wf, indent=4)

        trial_log_path: str = os.path.join(args.run_name, f'trial_{trial_idx}.log')
        trial_env_configs_log_path: str = os.path.join(args.run_name, f'env_results_trial_{trial_idx}.json')
        # syf添加注释：应该是如果已经存在log和json文件，就清空掉。syf0204注释掉，因为需要中断后接着跑
        # trial_env_configs_log_path已经写入完了，run_trail不调用了，需要保留更新用于下一个trail；
        # trial_log_path和world_log_path在run_trail中还要持续调用写入，不能覆盖。
        # if os.path.exists(trial_log_path):
        #     open(trial_log_path, 'w').close()
        if os.path.exists(trial_env_configs_log_path):
            open(trial_env_configs_log_path, 'w').close()

        # syf0207添加，更新自己的env_config_json
        uptodate_env_config_name = "reflexion_run_logs/trail" + str(trial_idx) + "_uptodate_env_config.json"
        with open(uptodate_env_config_name, 'w') as wf:
            json.dump(env_configs, wf, indent=4)

        # run trial
        run_trial(trial_log_path, world_log_path, trial_idx, env_configs, args.use_memory, args.model)

        # log world for trial
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')

        trial_idx += 1


if __name__ == '__main__':
    args = get_args()
    main(args)
