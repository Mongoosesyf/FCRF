export OPENAI_API_KEY="your API key"
export ALFWORLD_DATA="your alfworld position"

python main.py \
        --num_trials 10 \
        --num_envs 134 \
        --run_name "reflexion_run_logs" \
        --use_memory \
        --model "gpt-3.5-turbo"

