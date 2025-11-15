#!/bin/bash -l

#SBATCH --job-name=train_with_mot17_a40
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --output=/home/atuin/v100dd/v100dd19/sbatch_ld-detr_sf_clip/result-%x-%j.txt

dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=/home/atuin/v100dd/v100dd19/ld-detr/outputs/results_mot17
exp_id=exp

######## data paths
train_path=/home/atuin/v100dd/v100dd19/FlashVTG/internvideo2/mot17_train_release.jsonl
eval_path=/home/atuin/v100dd/v100dd19/FlashVTG/internvideo2/mot17_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=/home/atuin/v100dd/v100dd19/FlashVTG_sf_clip/final_version_slowfast_clip


# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features_mot17)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features_mot17)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features_mot17/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
max_v_l=750
seed=$((RANDOM << 15 | RANDOM))

PYTHONPATH=$PYTHONPATH:. python ld_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--max_v_l ${max_v_l} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--seed ${seed} \
${@:1}
