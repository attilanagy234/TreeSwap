log_file=$(grep 'log_file' config.yaml | awk '{ print $2 }')

echo "--Removing unused models--"
best_model=run/$(grep --text -A1 'Model is improving' $log_file | tail -n1 | sed 's/.*\/\(.*_step_[0-9]\+\.pt\).*/\1/') && \
best_model_path=${best_model##* } && \
echo "Best model found is: $best_model_path"
mv $best_model_path run/model && \
rm run/*.pt && \
mv run/model run/model.pt
