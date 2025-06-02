#!/usr/bin/env python
"""
Script to compute pooled EER and min tDCF for ASVspoof2021 LA. 
Usage:
$: python PATH_TO_SCORE_FILE PATH_TO_GROUDTRUTH_DIR phase
 
 -PATH_TO_SCORE_FILE: path to the score file 
 -PATH_TO_GROUNDTRUTH_DIR: path to the directory that has tje CM protocol and ASV score.
    Please follow README, download the key files, and use ./keys
 -phase: either progress, eval, or hidden_track
Example:
$: python evaluate.py score.txt ./keys eval
"""

import os.path

import numpy as np
import pandas

import evaluation.eval_metric_LA as em

Pspoof = 0.05
cost_model = {
    'Pspoof': Pspoof,  # Prior probability of a spoofing attack
    'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    'Cmiss': 1,  # Cost of tandem system falsely rejecting target speaker
    'Cfa': 10,  # Cost of tandem system falsely accepting nontarget speaker
    'Cfa_spoof': 10,  # Cost of tandem system falsely accepting spoof
}


def load_asv_metrics(asv_key_file, asv_scr_file, phase):
    # Load organizers' ASV scores
    asv_key_data = pandas.read_csv(asv_key_file, sep=' ', header=None)
    asv_scr_data = pandas.read_csv(asv_scr_file, sep=' ', header=None)[
        asv_key_data[7] == phase]
    idx_tar = asv_key_data[asv_key_data[7] == phase][5] == 'target'
    idx_non = asv_key_data[asv_key_data[7] == phase][5] == 'nontarget'
    idx_spoof = asv_key_data[asv_key_data[7] == phase][5] == 'spoof'

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scr_data[2][idx_tar]
    non_asv = asv_scr_data[2][idx_non]
    spoof_asv = asv_scr_data[2][idx_spoof]
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,
     Pfa_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv,
                                                asv_threshold)

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=False):
    bona_cm = cm_scores[cm_scores[5] == 'bonafide']['2_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['2_x'].values

    if invert == False:
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    else:
        eer_cm = em.compute_eer(-bona_cm, -spoof_cm)[0]

    if invert == False:
        tDCF_curve, _ = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv,
                                        Pfa_spoof_asv, cost_model, False)
    else:
        tDCF_curve, _ = em.compute_tDCF(-bona_cm, -spoof_cm, Pfa_asv, Pmiss_asv,
                                        Pfa_spoof_asv, cost_model, False)

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    return min_tDCF, eer_cm


def eval_score_file(score_file, truth_dir, phase):
    asv_key_file = os.path.join(truth_dir, 'ASV/trial_metadata.txt')
    asv_scr_file = os.path.join(truth_dir, 'ASV/ASVTorch_Kaldi/score.txt')
    cm_key_file = os.path.join(truth_dir, 'CM/trial_metadata.txt')

    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics(
        asv_key_file, asv_scr_file, phase)
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None,
                                        skipinitialspace=True)

    # check here for progress vs eval set
    cm_scores = submission_scores.merge(cm_data[cm_data[7] == phase], left_on=0,
                                        right_on=1, how='inner')
    min_tDCF, eer_cm = performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv)

    # just in case that the submitted file reverses the sign of positive and negative scores
    min_tDCF2, eer_cm2 = performance(cm_scores, Pfa_asv, Pmiss_asv,
                                     Pfa_spoof_asv, invert=True)

    eer_cm = min(eer_cm, eer_cm2)
    min_tDCF = min(min_tDCF, min_tDCF2)

    out_data = "min_tDCF: %.4f\n" % min_tDCF
    out_data += "eer: %.2f\n" % (100 * eer_cm)
    print(out_data, end="")

    return eer_cm, min_tDCF
