@echo off
setlocal

REM Define lists for parameters
set seed_list=1
set method_list=attentionlite_mhi c3d cnn_lstm zero_shot mediapipe_transformer mediapipe_lstm
set num_words_list=1
set split_list=train test

REM Loop over parameters
for %%s in (%seed_list%) do (
    for %%m in (%method_list%) do (
        for %%n in (%num_words_list%) do (
            for %%p in (%split_list%) do (
                echo Running: python main.py --method=%%m --split=%%p --num_words=%%n --seed=%%s
                python main.py --method=%%m --split=%%p --num_words=%%n --seed=%%s
            )
        )
    )
)

endlocal
