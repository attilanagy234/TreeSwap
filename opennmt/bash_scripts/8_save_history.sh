HISTORY_DIR=../../history
SAVE_DIR=$HISTORY_DIR/$(date '+%Y-%m-%d_%H:%M:%S')
pair_output_path=$(grep 'pair_output_path:' config.yaml | awk '{ print $2 }')
translate_model=$(grep 'translate_model:' config.yaml | awk '{ print $2 }')

TO_SAVE="config.yaml run/final_result.txt run/pred.txt.sp run/pred.txt run/tensorboard $pair_output_path $translate_model"
mkdir -p $HISTORY_DIR
mkdir $SAVE_DIR
for FILE in $TO_SAVE
do
    cp -r $FILE $SAVE_DIR/
done