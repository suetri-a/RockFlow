import numpy as np
import tifffile
from utilities import two_point_correlation
import pandas as pd
from tqdm import trange
import argparse
from scipy.optimize import curve_fit
import glob
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--original', help='Path to original image file ')
parser.add_argument('--synthetic', help='Path to folder with synthetic images + common name e.g. ../imgs/img3d_')
parser.add_argument('--ending', type=str, default='.tif', help='Image format for synthetic samples')
parser.add_argument('--output', help='Path to save output files')
parser.add_argument('--seed_min', type=int, default=0, help='Starting image #')
parser.add_argument('--seed_max', type=int, help='Number of samples to process')

def main(args):
    # Covariance analysis functions
    def radial_average(cov):
        avg = np.mean(cov, axis=0)
        return avg

    def straight_line_at_origin(porosity):
        def func(x, a):
            return a * x + porosity
        return func

    orig_img = tifffile.imread(args.original)

    # Confirm pore and grain phase values
    pore_phase = orig_img.max()
    grain_phase = orig_img.min()
    print("Pore Phase Value: ", pore_phase)
    print("Grain Phase Value: ", grain_phase)
    print("Image size is: ", orig_img.shape)

    # Calculate covariance for pore phase of original sample
    two_point_covariance_pore_phase_orig = {}
    for i, direc in enumerate(["x", "y", "z"]):
        two_point_direc = two_point_correlation(orig_img, i, var=pore_phase)
        two_point_covariance_pore_phase_orig[direc] = two_point_direc

    direc_covariances_pore_phase_orig = {}
    for direc in ["x", "y", "z"]:
        direc_covariances_pore_phase_orig[direc] = np.mean(np.mean(two_point_covariance_pore_phase_orig[direc], axis=0), axis=0)
    #print('Shape of x dir is: ', direc_covariances_pore_phase_orig["x"].shape)

    orig_cov_pph = pd.DataFrame(direc_covariances_pore_phase_orig)
    orig_cov_pph.to_csv(os.path.join(args.output, "orig_pph.csv"), sep=",", index=False)

    # Check first few cov results
    #orig_cov_pph_backload = pd.read_csv(os.path.join(args.output, "orig_pph.csv"))
    #orig_cov_pph_backload.head()

    del two_point_covariance_pore_phase_orig
    del two_point_direc

    # Calculate covariance for grain phase of original sample
    two_point_covariance_grain_phase_orig = {}
    for i, direc in enumerate(["x", "y", "z"]):
        two_point_direc = two_point_correlation(orig_img, i, var=grain_phase)
        two_point_covariance_grain_phase_orig[direc] = two_point_direc

    direc_covariances_grain_phase_orig = {}
    for direc in ["x", "y", "z"]:
        direc_covariances_grain_phase_orig[direc] = np.mean(np.mean(two_point_covariance_grain_phase_orig[direc], axis=0), axis=0)
    #print('Shape of x dir is: ', direc_covariances_grain_phase_orig["x"].shape)

    orig_cov_gph = pd.DataFrame(direc_covariances_grain_phase_orig)
    orig_cov_gph.to_csv(os.path.join(args.output, "orig_gph.csv"), sep=",", index=False)

    # Check first few cov results
    #covariances_orig_df_backload = pd.read_csv(os.path.join(args.output, "orig_pph.csv"))
    #covariances_orig_df_backload.head()

    del two_point_covariance_grain_phase_orig
    del two_point_direc

    # Compute slope of covariance at origin to get specific surface area, and chord length for each phase
    print("Saving covariance data for original sample...")
    # Compute radial average
    original_average_pph = radial_average(orig_cov_pph.values.T)
    original_average_gph = radial_average(orig_cov_gph.values.T)

    # Compute slope at origin of radially-averaged covariance, fit straight line at origin to get SSA
    N = 5
    slope_pph, slope_pph_cov = curve_fit(straight_line_at_origin(original_average_pph[0]), range(0, N),
                                         original_average_pph[0:N])
    slope_gph, slope_gph_cov = curve_fit(straight_line_at_origin(original_average_gph[0]), range(0, N),
                                         original_average_gph[0:N])
    #print("Slope for pore phase is: ", slope_pph)
    #print("Slope for grain phase is: ", slope_gph)

    specific_surface_orig = -4 * slope_pph
    #print("Original SSA is: ", specific_surface_orig)

    # Compute chord length
    chord_length_pph = -original_average_pph[0] / slope_pph
    chord_length_gph = -original_average_gph[0] / slope_gph
    #print("Chord length of pore phase is: ", chord_length_pph)
    #print("Chord length of grain phase: ", chord_length_gph)

    orig_data = {
        "slope_gph": float(slope_gph), "slope_pph": float(slope_pph),
        "specific_surface": float(specific_surface_orig),
        "chord_length_pph": float(chord_length_pph), "chord_length_gph": float(chord_length_gph)}

    # Store orig covariance values
    covariance_values = {}
    covariance_values["orig"] = orig_data


    # Repeat process for generated samples (pore phase only)
    for seed in trange(args.seed_min, (args.seed_min + args.seed_max)):
        im_in = tifffile.imread(os.path.join(args.synthetic, "*" + str(seed).zfill(3) + args.ending))
        image = im_in.astype(np.int8)

        # determine phase values
        pore_phase = image.min()
        grain_phase = image.max()

        for phase, phase_label in zip([pore_phase, grain_phase], ["pph", "gph"]):
            # phase computation
            two_point_covariance = {}
            for i, direc in enumerate(["x", "y", "z"]):
                two_point_direc = two_point_correlation(image, i, var=phase)
                two_point_covariance[direc] = two_point_direc

            # phase averaging
            direc_covariances = {}
            for direc in ["x", "y", "z"]:
                direc_covariances[direc] = np.mean(np.mean(two_point_covariance[direc], axis=0), axis=0)

            # covariance storage
            covariance_df = pd.DataFrame(direc_covariances)
            covariance_df.to_csv(os.path.join(args.output, "S_" + str(seed).zfill(3) + "_" + phase_label + ".csv"), sep=",", index=False)

    del im_in
    del image
    del two_point_covariance
    del direc_covariances
    del covariance_df

    # Compute slope of covariance at origin, specific surface area, and chord length for each phase
    print("Saving covariance data for synthetic samples...")
    for i in range(args.seed_min, (args.seed_min + args.seed_max)):
        cov_pph = pd.read_csv(os.path.join(args.output, "S_" + str(i).zfill(3) + "_pph.csv"))
        cov_gph = pd.read_csv(os.path.join(args.output, "S_" + str(i).zfill(3) + "_gph.csv"))

        average_pph = radial_average(cov_pph.values.T)
        average_gph = radial_average(cov_gph.values.T)

        slope_pph, slope_pph_cov = curve_fit(straight_line_at_origin(average_pph[0]), range(0, N), average_pph[0:N])
        slope_gph, slope_gph_cov = curve_fit(straight_line_at_origin(average_gph[0]), range(0, N), average_gph[0:N])

        specific_surface = -4 * slope_pph

        chord_length_pph = -average_pph[0] / slope_pph
        chord_length_gph = -average_gph[0] / slope_gph

        data = {
            "slope_gph": float(slope_gph), "slope_pph": float(slope_pph),
            "specific_surface": float(specific_surface),
            "chord_length_pph": float(chord_length_pph), "chord_length_gph": float(chord_length_gph)}
        covariance_values["S_" + str(i).zfill(3)] = data

    print("Synthetic SSA is: ", np.mean(specific_surface))

    # Store synthetic covariance values
    with open(os.path.join(args.output, "covariance_data.json"), "w") as f:
        json.dump(covariance_values, f)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)