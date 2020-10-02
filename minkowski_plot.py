# Plot data from ImageJ calculations
# Based on notebook from: https://github.com/LukasMosser/PorousMediaGan/blob/master/code/notebooks/minkowski/Minkowski%20Functional%20Analysis%20Berea.ipynb
from pylab import setp
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import glob
from matplotlib.ticker import ScalarFormatter

parser = argparse.ArgumentParser()

parser.add_argument('--original', help='Path to folder with original csv files} ')
parser.add_argument('--synthetic', help='Path to folder with synthetic csv files} ')

parser.add_argument("--perm", type=bool, default=False, help='Include perm or not in graphs')
parser.add_argument("--perm_path", help="path to csv file with permeability results")

parser.add_argument("--pixel_size", type=int, default=128, help="size of input rock image")
parser.add_argument('--dim', type=int, default=3, help='2 or 3 for 2D or 3D dataset')

def main(args):
    #Load data

    # Read individual .csv files from ImageJ script output and compile into a single file
    original_files = glob.glob(os.path.join(args.original, "*.csv"))
    img_original = pd.concat((pd.read_csv(f) for f in original_files))
    #img_original.to_csv(os.path.join(args.original, 'original-compiled.csv'), index=False)

    synthetic_files = glob.glob(os.path.join(args.synthetic, "*.csv"))
    img_synthetic = pd.concat((pd.read_csv(f) for f in synthetic_files))
    #img_synthetic.to_csv(os.path.join(args.synthetic, 'synthetic-compiled.csv'), index=False)

    img_original["Type"] = 0
    img_synthetic["Type"] = 1
    img_minkowski = pd.concat([img_original, img_synthetic])
    img_minkowski["Type"] = img_minkowski["Type"].astype("category")

    if args.perm:
        perms = pd.read_csv(args.perm_path)

    if args.dim == 2:
        parameters = ["Area", "Perimeter", "EulerNumber"]
        labels = [r"Porosity \ [-]", r"Perimeter \ [-]", r"Euler \ Characteristic  \ [\frac{1}{pixel^2}]"]
    else:
        if args.perm:
            parameters = ["Volume", "SurfaceArea", "EulerNumber", "Perm"]
            labels = [r"Porosity \ [-]", r"SurfaceArea \ [\frac{1}{voxel}]",
                                  r"Euler \ Characteristic  \ [\frac{1}{voxel^3}]", r"Permeability \ [m^2]"]
            units = [r'$1\times 10^{-2}$', r'$1\times 10^{-5}$', r'$1\times 10^{-13}$']
        else:
            parameters = ["Volume", "SurfaceArea", "EulerNumber"]
            labels = [r"Porosity \ [-]", r"SurfaceArea \ [\frac{1}{voxel}]",
                      r"Euler \ Characteristic  \ [\frac{1}{voxel^3}]"]
            units = [r'$1\times 10^{-2}$', r'$1\times 10^{-5}$']

    # Calculate porosity, surface area, and Euler number
    img_original[parameters[0]] /= args.pixel_size ** args.dim
    img_original[parameters[0]] = (1. - img_original[parameters[0]])
    img_original[parameters[1]] /= args.pixel_size ** args.dim
    img_original[parameters[2]] /= args.pixel_size ** args.dim

    img_synthetic[parameters[0]] /= args.pixel_size ** args.dim
    img_synthetic[parameters[0]] = (1. - img_synthetic[parameters[0]])
    img_synthetic[parameters[1]] /= args.pixel_size ** args.dim
    img_synthetic[parameters[2]] /= args.pixel_size ** args.dim

    # Set figure formatting
    def setBoxColors(bp):
        width = 3
        setp(bp['boxes'][0], color='black', linewidth=width)
        setp(bp['boxes'][1], color='black', linewidth=width)

        for i in range(4):
            setp(bp['caps'][i], color='black', linewidth=width * 2)
            setp(bp['whiskers'][i], color='black', linewidth=width, linestyle='dashed')

        setp(bp['fliers'][0], color='black', linewidth=width)
        setp(bp['fliers'][1], color='black', linewidth=width)
        setp(bp['medians'][0], color='black', linewidth=width, linestyle='dotted')
        setp(bp['medians'][1], color='black', linewidth=width, linestyle='dotted')


    fig, ax = plt.subplots(1, len(labels), figsize=(48, 12))

    for i, prop in enumerate(parameters):
        if prop == "Perm":
            perms_orig = perms["Perms_orig"]
            perms_synth = perms["Perms_16"]
            data = [perms_orig, perms_synth]
            bp = ax[i].boxplot(data)
            setBoxColors(bp)
        else:
            data = [img_original[prop].values, img_synthetic[prop].values]
            bp = ax[i].boxplot(data)
            setBoxColors(bp)

    for j, prop in enumerate(labels):
        ax[j].set_title(r"$" + prop + r"$", fontsize=42, y=1.06)

    fig.canvas.draw()

    for i in range(len(labels)):
        labels = [item.get_text() for item in ax[i].get_xticklabels()]
        labels[0] = r'$Original$'
        labels[1] = r'$Synthetic$'
        ax[i].set_xticklabels(labels, fontsize=36)

        labels_y = [item.get_text() for item in ax[i].get_yticklabels()]
        ax[i].set_yticklabels(labels_y, fontsize=26)
        ax[i].grid()
        # ax[i].xaxis.set_major_formatter(ScalarFormatter())
        ax[i].yaxis.set_major_formatter(ScalarFormatter())

        if i > 0:
            ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax[i].get_yaxis().get_offset_text().set(va='bottom', ha='left')
            ax[i].yaxis.get_offset_text().set_fontsize(26)

    for i, s in enumerate(units):
        t = ax[i + 1].text(0.01, 1.016, s, transform=ax[i + 1].transAxes, fontsize=30)
        t.set_bbox(dict(color='white', alpha=1.0, edgecolor=None))

    # Save in results folder (change)
    fig.savefig(os.path.join("figures/minkowski_functionals.png"), bbox_extra_artists=None, bbox_inches='tight', dpi=72)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
