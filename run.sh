#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using xvectors.
# It is closely based on "X-vectors: Robust DNN Embeddings for Speaker
# Recognition" by Snyder et al.  In the future, we will add score-normalization
# and a more effective form of PLDA domain adaptation.
#
# Pretrained models are available for this recipe.  See
# http://kaldi-asr.org/models.html and
# https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html
# for details.
train_cmd=
iter=
. ./cmd.sh
. ./path.sh
set -e
. ./utils/parse_options.sh

model_name=ModelL2LossWithoutDropoutTdnnRelu

stage=0
train_stage=-1

mfccdir=./mfcc
vaddir=./mfcc

data_dir=./data
exp_dir=./exp
egs_dir=$exp_dir/xvector_nnet_1a/egs
nnet_dir=$exp_dir/xvector_nnet_1a
# SRE16 trials
sre16_trials=$data_dir/sre16_eval_test/trials
sre16_trials_tgl=$data_dir/sre16_eval_test/trials_tgl
sre16_trials_yue=$data_dir/sre16_eval_test/trials_yue

if [ $stage -le 0 ]; then
  # Path to some, but not all of the training corpora
  sre18_sre16_fisher_mixer6=/home4T_0/data/sre18/sre18_sre16_fisher_mixer6
  swb_sre04_12=/home4T_0/data/sre18/swb_sre04-12
  
  # Prepare telephone and microphone speech from Mixer6.
  local/make_mx6.sh $sre18_sre16_fisher_mixer6 $data_dir/

  # Prepare SRE10 test and enroll. Includes microphone interview speech.
  # NOTE: This corpus is now available through the LDC as LDC2017S06.
  local/make_sre10.pl $swb_sre04_12/sre10 $data_dir/

  # Prepare SRE08 test and enroll. Includes some microphone speech.
  local/make_sre08.pl $swb_sre04_12/sre08/test $swb_sre04_12/sre08/train $data_dir/

  # This prepares the older NIST SREs from 2004-2006.
  local/my_make_sre.sh $swb_sre04_12 $data_dir/

  # Combine all SREs prior to 2016 and Mixer6 into one dataset
  utils/combine_data.sh $data_dir/sre \
    $data_dir/sre2004 $data_dir/sre2005_train \
    $data_dir/sre2005_test $data_dir/sre2006_train \
    $data_dir/sre2006_test \
    $data_dir/sre08 $data_dir/mx6 $data_dir/sre10
  utils/validate_data_dir.sh --no-text --no-feats $data_dir/sre
  utils/fix_data_dir.sh $data_dir/sre

  # Prepare SWBD corpora.
  local/make_swbd_cellular1.pl $swb_sre04_12/LDC2001S13 \
    $data_dir/swbd_cellular1_train
  local/make_swbd_cellular2.pl $swb_sre04_12/LDC2004S07 \
    $data_dir/swbd_cellular2_train
  local/make_swbd2_phase1.pl $swb_sre04_12/LDC98S75 \
    $data_dir/swbd2_phase1_train
  local/make_swbd2_phase2.pl $swb_sre04_12/LDC99S79 \
    $data_dir/swbd2_phase2_train
  local/make_swbd2_phase3.pl $swb_sre04_12/LDC2002S06 \
    $data_dir/swbd2_phase3_train

  # Combine all SWB corpora into one dataset.
  utils/combine_data.sh $data_dir/swbd \
    $data_dir/swbd_cellular1_train $data_dir/swbd_cellular2_train \
    $data_dir/swbd2_phase1_train $data_dir/swbd2_phase2_train $data_dir/swbd2_phase3_train
  
  sre16_path=$sre18_sre16_fisher_mixer6/LDC2018E30_2016_NIST_Speaker_Recognition_Evaluation_Test_Set
  sre16_eval_path=$sre16_path/data/eval/R149_0_1
  sre16_dev_path=$sre16_path/data/dev/R148_0_0
  # Prepare NIST SRE 2016 evaluation data.
  local/make_sre16_eval.pl $sre16_eval_path $data_dir

  # Prepare NIST SRE 2016 development data.
  local/make_sre16_dev.pl $sre16_dev_path $data_dir

  # Prepare unlabeled Cantonese and Tagalog development data. This dataset
  # was distributed to SRE participants.
  local/make_sre16_unlabeled.pl $sre16_dev_path $data_dir
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  # if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
  #   utils/create_split_dir.pl \
  #     /export/b{14,15,16,17}/$USER/kaldi-data/egs/sre16/v2/xvector-$(date +'%m_%d_%H_%M')/mfccs/storage $mfccdir/storage
  # fi
  for name in sre swbd sre16_eval_enroll sre16_eval_test sre16_major; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 48 --cmd "$train_cmd" \
      $data_dir/${name}  $exp_dir/make_mfcc $mfccdir
    utils/fix_data_dir.sh $data_dir/${name}
    sid/compute_vad_decision.sh --nj 48 --cmd "$train_cmd" \
      $data_dir/${name}  $exp_dir/make_vad $vaddir
    utils/fix_data_dir.sh $data_dir/${name}
  done
  utils/combine_data.sh --extra-files "utt2num_frames" $data_dir/swbd_sre $data_dir/swbd $data_dir/sre
  utils/fix_data_dir.sh $data_dir/swbd_sre
fi

# In this section, we augment the SWBD and SRE data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' $data_dir/swbd_sre/utt2num_frames > $data_dir/swbd_sre/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    $data_dir/swbd_sre $data_dir/swbd_sre_reverb
  cp $data_dir/swbd_sre/vad.scp $data_dir/swbd_sre_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" $data_dir/swbd_sre_reverb $data_dir/swbd_sre_reverb.new
  rm -rf $data_dir/swbd_sre_reverb
  mv $data_dir/swbd_sre_reverb.new $data_dir/swbd_sre_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 8000 /home/jyh/D/jyh/data/MUSAN/musan $data_dir

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh $data_dir/musan_${name}
    mv $data_dir/musan_${name}/utt2dur $data_dir/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "$data_dir/musan_noise" $data_dir/swbd_sre $data_dir/swbd_sre_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "$data_dir/musan_music" $data_dir/swbd_sre $data_dir/swbd_sre_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "$data_dir/musan_speech" $data_dir/swbd_sre $data_dir/swbd_sre_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh $data_dir/swbd_sre_aug $data_dir/swbd_sre_reverb $data_dir/swbd_sre_noise $data_dir/swbd_sre_music $data_dir/swbd_sre_babble

  # Take a random subset of the augmentations (128k is somewhat larger than twice
  # the size of the SWBD+SRE list)
  utils/subset_data_dir.sh $data_dir/swbd_sre_aug 128000 $data_dir/swbd_sre_aug_128k
  utils/fix_data_dir.sh $data_dir/swbd_sre_aug_128k

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 48 --cmd "$train_cmd" \
    $data_dir/swbd_sre_aug_128k  $exp_dir/make_mfcc $mfccdir

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh $data_dir/swbd_sre_combined $data_dir/swbd_sre_aug_128k $data_dir/swbd_sre

  # Filter out the clean + augmented portion of the SRE list.  This will be used to
  # train the PLDA model later in the script.
  utils/copy_data_dir.sh $data_dir/swbd_sre_combined $data_dir/sre_combined
  utils/filter_scp.pl $data_dir/sre/spk2utt $data_dir/swbd_sre_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > $data_dir/sre_combined/utt2spk
  utils/fix_data_dir.sh $data_dir/sre_combined

fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    $data_dir/swbd_sre_combined $data_dir/swbd_sre_combined_no_sil  $exp_dir/swbd_sre_combined_no_sil
  utils/fix_data_dir.sh $data_dir/swbd_sre_combined_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv $data_dir/swbd_sre_combined_no_sil/utt2num_frames $data_dir/swbd_sre_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data_dir/swbd_sre_combined_no_sil/utt2num_frames.bak > $data_dir/swbd_sre_combined_no_sil/utt2num_frames
  utils/filter_scp.pl $data_dir/swbd_sre_combined_no_sil/utt2num_frames $data_dir/swbd_sre_combined_no_sil/utt2spk > $data_dir/swbd_sre_combined_no_sil/utt2spk.new
  mv $data_dir/swbd_sre_combined_no_sil/utt2spk.new $data_dir/swbd_sre_combined_no_sil/utt2spk
  utils/fix_data_dir.sh $data_dir/swbd_sre_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' $data_dir/swbd_sre_combined_no_sil/spk2utt > $data_dir/swbd_sre_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data_dir/swbd_sre_combined_no_sil/spk2num | utils/filter_scp.pl - $data_dir/swbd_sre_combined_no_sil/spk2utt > $data_dir/swbd_sre_combined_no_sil/spk2utt.new
  mv $data_dir/swbd_sre_combined_no_sil/spk2utt.new $data_dir/swbd_sre_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl $data_dir/swbd_sre_combined_no_sil/spk2utt > $data_dir/swbd_sre_combined_no_sil/utt2spk

  utils/filter_scp.pl $data_dir/swbd_sre_combined_no_sil/utt2spk $data_dir/swbd_sre_combined_no_sil/utt2num_frames > $data_dir/swbd_sre_combined_no_sil/utt2num_frames.new
  mv $data_dir/swbd_sre_combined_no_sil/utt2num_frames.new $data_dir/swbd_sre_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh $data_dir/swbd_sre_combined_no_sil
fi

if [ $stage -le 6 ]; then

  local/tf/run_xvector.sh --stage $stage --train-stage -1 \
    --model_name $model_name \
    --data $data_dir/swbd_sre_combined_no_sil --nnet-dir $nnet_dir \
    --egs-dir $egs_dir
fi

if [ ${stage} -le 7 ]; then
  # The SRE16 major is an unlabeled dataset consisting of Cantonese and
  # and Tagalog.  This is useful for things like centering, whitening and
  # score normalization.
  local/tf/extract_xvectors.sh --cmd "${train_cmd}" --nj 1 \
    ${nnet_dir} $data_dir/sre16_major \
    ${nnet_dir}/xvectors_sre16_major

  # Extract xvectors for SRE data (includes Mixer 6). We'll use this for
  # things like LDA or PLDA.
  local/tf/extract_xvectors.sh --cmd "${train_cmd}" --nj 1 \
    ${nnet_dir} $data_dir/sre_combined \
    ${nnet_dir}/xvectors_sre_combined

  # The SRE16 test data
  local/tf/extract_xvectors.sh --cmd "${train_cmd}" --nj 1 \
    ${nnet_dir} $data_dir/sre16_eval_test \
    ${nnet_dir}/xvectors_sre16_eval_test

  # The SRE16 enroll data
  local/tf/extract_xvectors.sh --cmd "${train_cmd}" --nj 1 \
    ${nnet_dir} $data_dir/sre16_eval_enroll \
    ${nnet_dir}/xvectors_sre16_eval_enroll
fi


if [ ${stage} -le 8 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  ${train_cmd} ${nnet_dir}/xvectors_sre16_major/log/compute_mean.log \
    ivector-mean scp:${nnet_dir}/xvectors_sre16_major/xvector.scp \
    ${nnet_dir}/xvectors_sre16_major/mean.vec || exit 1;

  lda_dim=100
  # This script uses LDA to decrease the dimensionality prior to PLDA.
  ${train_cmd} ${nnet_dir}/xvectors_sre_combined/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_sre_combined/xvector.scp ark:- |" \
    ark:$data_dir/sre_combined/utt2spk ${nnet_dir}/xvectors_sre_combined/transform.mat || exit 1;

  # Train an out-of-domain PLDA model.
  ${train_cmd} ${nnet_dir}/xvectors_sre_combined/log/plda.log \
    ivector-compute-plda ark:$data_dir/sre_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_sre_combined/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    ${nnet_dir}/xvectors_sre_combined/plda || exit 1;

  # Here we adapt the out-of-domain PLDA model to SRE16 major, a pile
  # of unlabeled in-domain data.  In the future, we will include a clustering
  # based approach for domain adaptation, which tends to work better.
  ${train_cmd} ${nnet_dir}/xvectors_sre16_major/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    ${nnet_dir}/xvectors_sre_combined/plda \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_sre16_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${nnet_dir}/xvectors_sre16_major/plda_adapt || exit 1;
fi

if [ ${stage} -le 9 ]; then
  echo "sre16_$model_name score:" >>score.txt
  # Get results using the out-of-domain PLDA model.
  ${train_cmd} ${nnet_dir}/scores/log/sre16_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre16_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre_combined/plda - |" \
    "ark:ivector-mean ark:$data_dir/sre16_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre16_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre16_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre16_major/mean.vec scp:${nnet_dir}/xvectors_sre16_eval_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/sre16_eval_scores_$lda_dim || exit 1;

  utils/filter_scp.pl $sre16_trials_tgl ${nnet_dir}/scores/sre16_eval_scores_$lda_dim > ${nnet_dir}/scores/sre16_eval_tgl_scores
  utils/filter_scp.pl $sre16_trials_yue ${nnet_dir}/scores/sre16_eval_scores_$lda_dim > ${nnet_dir}/scores/sre16_eval_yue_scores
  pooled_eer=$(paste $sre16_trials ${nnet_dir}/scores/sre16_eval_scores_$lda_dim | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl ${nnet_dir}/scores/sre16_eval_tgl_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue ${nnet_dir}/scores/sre16_eval_yue_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA, EER: Pooled ${pooled_eer} %, Tagalog ${tgl_eer} %, Cantonese ${yue_eer} %" >>score.txt
  
fi

if [ ${stage} -le 10 ]; then
  # Get results using the adapted PLDA model.
  ${train_cmd} ${nnet_dir}/scores/log/sre16_eval_scoring_adapt.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre16_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre16_major/plda_adapt - |" \
    "ark:ivector-mean ark:$data_dir/sre16_eval_enroll/spk2utt scp:${nnet_dir}/xvectors_sre16_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre16_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre16_major/mean.vec scp:${nnet_dir}/xvectors_sre16_eval_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/sre16_eval_scores_adapt_$lda_dim || exit 1;

  utils/filter_scp.pl $sre16_trials_tgl ${nnet_dir}/scores/sre16_eval_scores_adapt_$lda_dim > ${nnet_dir}/scores/sre16_eval_tgl_scores_adapt
  utils/filter_scp.pl $sre16_trials_yue ${nnet_dir}/scores/sre16_eval_scores_adapt_$lda_dim > ${nnet_dir}/scores/sre16_eval_yue_scores_adapt
  pooled_eer=$(paste $sre16_trials ${nnet_dir}/scores/sre16_eval_scores_adapt_$lda_dim | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  tgl_eer=$(paste $sre16_trials_tgl ${nnet_dir}/scores/sre16_eval_tgl_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  yue_eer=$(paste $sre16_trials_yue ${nnet_dir}/scores/sre16_eval_yue_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Adapted PLDA, EER: Pooled ${pooled_eer} %, Tagalog ${tgl_eer} %, Cantonese ${yue_eer} %" >>score.txt

fi

