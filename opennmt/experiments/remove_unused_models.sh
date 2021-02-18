utils_path=../../utils

best_model=$(python $utils_path/get_best_model_path.py --run_dir run) && \
mv $best_model run/model && \
rm run/*.pt && \
mv run/model run/model.pt