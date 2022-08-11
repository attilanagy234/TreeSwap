#!/bin/bash

log_file=$(yq -r .log_file config.yaml)

echo "--Removing unused models--"
best_model=run/$(grep --text -A1 'Model is improving' $log_file | tail -n1 | sed 's/.*\/\(.*_step_[0-9]\+\.pt\).*/\1/') && \
echo "Best model found is: $best_model"
mv $best_model run/model && \
rm -f run/*.pt
mv run/model run/model.pt
