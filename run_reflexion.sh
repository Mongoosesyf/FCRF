#python main.py \
#        --num_trials 10 \
#        --num_envs 134 \
#        --run_name "reflexion_run_logs" \
#        --use_memory \
#        --model "gpt-3.5-turbo"

export OPENAI_API_KEY="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"
export ALFWORLD_DATA="/home/syf/PycharmProjects/reflexion/Alfworld/my_ALFWORLD_DATA"

## 跑trail0:(只跑一次)。跑完后把trail0生成的反思删除，跑main_rfl_first.py
## model为0205syf修改，原来模型为"openai/gpt-3.5-turbo", 因为alfworld推理总是中断，尝试用instruct/gpt4模型推理
## 原本--num_trials 2，有--use_memory \再手动删除反思内容，syf02认为不用，删除此参数只推理不反思，--num_trials改为1
#python main.py \
#        --num_trials 2 \
#        --num_envs 134 \
#        --run_name "reflexion_run_logs" \
#        --use_memory \
#        --model "openai/gpt-4o-mini" \


## 每一轮先生成反思再跑trail，从trail1开始跑
## 记得在main_rfl_first.py里修改import，跑自己的import自己的，跑orig reflexion import原生generate_reflexion
## 模型原来为--model "openai/gpt-3.5-turbo"
## --num_trials 就是实际有几个trails，比如num trails = 3，就跑trail 0,1,2
#python main_rfl_first.py \
#        --num_trials 3 \
#        --num_envs 134 \
#        --run_name "reflexion_run_logs" \
#        --use_memory \
#        --model "openai/gpt-4o-mini" \
#        --start_trial_num 2 \
#        --is_resume \
#        --resume_dir "reflexion_run_logs"

## --use_memory false, ReAct
## --model "openai/gpt-4o-mini" \
#python main_rfl_first.py \
#        --num_trials 5 \
#        --num_envs 134 \
#        --run_name "reflexion_run_logs" \
#        --model "openai/gpt-3.5-turbo" \
#        --start_trial_num 4 \
#        --is_resume \
#        --resume_dir "reflexion_run_logs"

# 0224添加，跑录视频的脚本
# py文件为改过for 脚本的alwfowrd_trail.py，原始版本存档在/存档 文件夹里
python main_rfl_first.py \
        --num_trials 1 \
        --num_envs 134 \
        --run_name "reflexion_run_logs" \
        --model "openai/gpt-4o-mini" \
        --start_trial_num 0 \
        --is_resume \
        --resume_dir "reflexion_run_logs"