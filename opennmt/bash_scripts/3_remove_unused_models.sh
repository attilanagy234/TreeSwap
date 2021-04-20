log_file=$(grep 'log_file' config.yaml | awk '{ print $2 }')

best_model=run/$(grep -A1 'Model is improving' $log_file | tail -n1 | sed 's/.*\/\(.*_step_[0-9]\+\.pt\).*/\1/') && \
mv $best_model run/model && \
rm run/*.pt && \
mv run/model run/model.pt