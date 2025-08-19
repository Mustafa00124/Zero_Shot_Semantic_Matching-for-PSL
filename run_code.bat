@echo off
setlocal

REM Define lists for parameters
set seed_list=1
set method_list=finetuned_gemini zero_shot semantic_zero_shot
set num_words_list=1

REM Loop over parameters
for %%s in (%seed_list%) do (
    for %%m in (%method_list%) do (
        for %%n in (%num_words_list%) do (
            echo Running: python main.py --method %%m --num_words %%n --seed %%s
            python main.py --method %%m --num_words %%n --seed %%s
        )
    )
)

endlocal
