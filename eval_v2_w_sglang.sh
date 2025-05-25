#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

declare -A MODE_MODEL_DICT

MODE_MODEL_DICT["llama3.2-base"]="meta-llama/Llama-3.2-3B-Instruct"

prompt_modes=("full_template" "simple_steer")
COMMON_SENSE=False
STORY_COMPLETION=False
OPEN_ENDED_TEXT_GENERATION=True
temperature=1.0
NUM_WORKERS=8
COMMON_SENSE_NUM_EXAMPLES=512
STORY_COMPLETION_NUM_EXAMPLES=256
OPEN_ENDED_TEXT_GENERATION_NUM_EXAMPLES=1024

if [ ! -d "data" ]; then
    mkdir data
fi

set -xe
# Common Sense
if [ "$COMMON_SENSE" == "True" ]; then
    if [ ! -d "eval/common_sense/results" ]; then
        mkdir eval/common_sense/results
    fi
    ## Common Gen
    for mode in "${!MODE_MODEL_DICT[@]}"; do
        for prompt_mode in "${prompt_modes[@]}"; do
            echo "Evaluating $mode, model ${MODE_MODEL_DICT[$mode]}, prompt_mode $prompt_mode"
            python eval/common_sense/common_gen_sglang.py \
                --mode $mode \
                --model ${MODE_MODEL_DICT[$mode]} \
                --num_samples $COMMON_SENSE_NUM_EXAMPLES \
                --num_workers $NUM_WORKERS \
                --batch_size 64 \
                --max_new_tokens 64 \
                --temperature $temperature \
                --prompt_mode $prompt_mode
        done
    done

    # ELI5
    for mode in "${!MODE_MODEL_DICT[@]}"; do
        for prompt_mode in "${prompt_modes[@]}"; do
            echo "Evaluating $mode, model ${MODE_MODEL_DICT[$mode]}, prompt_mode $prompt_mode"
            python eval/common_sense/eli5_sglang.py \
                --mode $mode \
                --model ${MODE_MODEL_DICT[$mode]} \
                --num_samples $COMMON_SENSE_NUM_EXAMPLES \
                --num_workers $NUM_WORKERS \
                --batch_size 64 \
                --max_new_tokens 128 \
                --temperature $temperature \
                --prompt_mode $prompt_mode
        done
    done

    # Natural Questions
    for mode in "${!MODE_MODEL_DICT[@]}"; do
        for prompt_mode in "${prompt_modes[@]}"; do
            echo "Evaluating $mode, model ${MODE_MODEL_DICT[$mode]}, prompt_mode $prompt_mode"
            python eval/common_sense/natural_questions_sglang.py \
                --mode $mode \
                --model ${MODE_MODEL_DICT[$mode]} \
                --num_samples $COMMON_SENSE_NUM_EXAMPLES \
                --num_workers $NUM_WORKERS \
                --batch_size 64 \
                --max_new_tokens 128 \
                --temperature $temperature \
                --prompt_mode $prompt_mode
        done
    done
fi

# Story Completion
if [ "$STORY_COMPLETION" == "True" ]; then
    if [ ! -d "eval/story_completion/results" ]; then
        mkdir eval/story_completion/results
    fi
    for mode in "${!MODE_MODEL_DICT[@]}"; do
        for prompt_mode in "${prompt_modes[@]}"; do
            echo "Evaluating $mode, model ${MODE_MODEL_DICT[$mode]}, prompt_mode $prompt_mode"
            python eval/story_completion/writingprompts_sglang.py \
                --mode $mode \
                --model ${MODE_MODEL_DICT[$mode]} \
                --num_samples $STORY_COMPLETION_NUM_EXAMPLES \
                --num_workers $NUM_WORKERS \
                --batch_size 8 \
                --max_new_tokens 512 \
                --temperature $temperature \
                --prompt_mode $prompt_mode
        done
    done

    for mode in "${!MODE_MODEL_DICT[@]}"; do
        for prompt_mode in "${prompt_modes[@]}"; do
            echo "Evaluating $mode, model ${MODE_MODEL_DICT[$mode]}, prompt_mode $prompt_mode"
            python eval/story_completion/rocstory_sglang.py \
                --mode $mode \
                --model ${MODE_MODEL_DICT[$mode]} \
                --num_samples $STORY_COMPLETION_NUM_EXAMPLES \
                --num_workers $NUM_WORKERS \
                --batch_size 8 \
                --max_new_tokens 512 \
                --temperature $temperature \
                --prompt_mode $prompt_mode
        done
    done

    for mode in "${!MODE_MODEL_DICT[@]}"; do
        for prompt_mode in "${prompt_modes[@]}"; do
            echo "Evaluating $mode, model ${MODE_MODEL_DICT[$mode]}, prompt_mode $prompt_mode"
            python eval/story_completion/story_cloze_sglang.py \
                --mode $mode \
                --model ${MODE_MODEL_DICT[$mode]} \
                --num_samples $STORY_COMPLETION_NUM_EXAMPLES \
                --num_workers $NUM_WORKERS \
                --batch_size 8 \
                --max_new_tokens 512 \
                --temperature $temperature \
                --prompt_mode $prompt_mode
        done
    done
fi

# Open Ended Text Generation
if [ "$OPEN_ENDED_TEXT_GENERATION" == "True" ]; then
    if [ ! -d "eval/open_ended_text_generation/results" ]; then
        mkdir eval/open_ended_text_generation/results
    fi
    # for mode in "${!MODE_MODEL_DICT[@]}"; do
    #     for prompt_mode in "${prompt_modes[@]}"; do
    #         echo "Evaluating $mode, model ${MODE_MODEL_DICT[$mode]}, prompt_mode $prompt_mode"
    #         python eval/open_ended_text_generation/news_generation_sglang.py \
    #             --mode $mode \
    #             --model ${MODE_MODEL_DICT[$mode]} \
    #             --model_engine gpt-4o \
    #             --num_samples $OPEN_ENDED_TEXT_GENERATION_NUM_EXAMPLES \
    #             --num_workers $NUM_WORKERS \
    #             --batch_size 128 \
    #             --max_new_tokens 512 \
    #             --temperature $temperature \
    #             --prompt_mode $prompt_mode
    #     done
    # done

    # for mode in "${!MODE_MODEL_DICT[@]}"; do
    #     for prompt_mode in "${prompt_modes[@]}"; do
    #         echo "Evaluating $mode, model ${MODE_MODEL_DICT[$mode]}, prompt_mode $prompt_mode"
    #         python eval/open_ended_text_generation/travel_generation_sglang.py \
    #             --mode $mode \
    #             --model ${MODE_MODEL_DICT[$mode]} \
    #             --model_engine gpt-4o \
    #             --num_samples $OPEN_ENDED_TEXT_GENERATION_NUM_EXAMPLES \
    #             --num_workers $NUM_WORKERS \
    #             --batch_size 32 \
    #             --max_new_tokens 64 \
    #             --temperature $temperature \
    #             --prompt_mode $prompt_mode 
    #     done
    # done

    for mode in "${!MODE_MODEL_DICT[@]}"; do
        for prompt_mode in "${prompt_modes[@]}"; do
            echo "Evaluating $mode, model ${MODE_MODEL_DICT[$mode]}, prompt_mode $prompt_mode"
            python eval/open_ended_text_generation/book_generation_sglang.py \
                --mode $mode \
                --model ${MODE_MODEL_DICT[$mode]} \
                --model_engine gpt-4o \
                --num_samples $OPEN_ENDED_TEXT_GENERATION_NUM_EXAMPLES \
                --num_workers $NUM_WORKERS \
                --batch_size 32 \
                --max_new_tokens 64 \
                --temperature $temperature \
                --prompt_mode $prompt_mode 
        done
    done
fi

