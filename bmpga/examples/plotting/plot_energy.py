# coding=utf-8
"""
Simple utility for visualising the progress of optimisations
"""
import argparse
import numpy as np

import matplotlib.pyplot as plt

from bmpga.storage import Database

def get_args():
    """Get CLI args for plotting"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--ifiles",
                        default=None,
                        help="File(s) to parse for energies (optional)\n"
                             "(default = None)")

    parser.add_argument("--ofile",
                        default=None,
                        type=str,
                        help="File in which to save graph (optional) (default = None)")

    parser.add_argument("--show", action="store_true", default=False)

    parser.add_argument("--ulim", type=float, help="Upper bound on y-axis")
    parser.add_argument("--llim", type=float, help="Lower bound on y-axis")

    return parser.parse_args()


class EnergyVsStep:

    def __init__(self, infiles=None, outfile=None, show=True, ulim=0.0, llim=0.0):

        if infiles is not None:
            self.infiles = infiles
            self.data_labels = infiles
            self.__call__ = self.plot_files

        else:
            self.infiles = ["all_energies.txt", "best_energies.txt"]
            self.data_labels = ["All energies", "Best energies"]
            self.__call__ = self.plot_all_best_files

        self.outfile = outfile

        self.ylim = ulim
        self.ymin = llim

        self.show = show
        self.fig, self.ax = None, None
        self.colour_idx = 0
        self.__call__()

    def plot_files(self):
        """Iterate through the list of input files and plot each on the same ax"""

        self.fig, self.ax = plt.subplots()

        plt.title("Energy vs step")

        plt.xlabel("Step number")
        plt.ylabel("Energy")

        for file, label in zip(self.infiles, self.data_labels):
            self.plot_file(file, label)

        plt.ylim(self.ymin, self.ylim, )
        plt.legend()

        if self.outfile is not None:
            plt.savefig(self.outfile, dpi=1000, bbox="tight")

        if self.show:
            plt.show()

    def plot_all_best_files(self):

        self.fig, self.ax = plt.subplots()

        plt.title("Energy vs step")

        plt.xlabel("Step number")
        plt.ylabel("Energy")

        self.plot_all_ens_file(self.infiles[0], self.data_labels[0])
        self.plot_best_ens_file(self.infiles[1], self.data_labels[1])

        plt.ylim(self.ymin, self.ylim, )
        plt.legend()

        if self.outfile is not None:
            plt.savefig(self.outfile, dpi=1000, bbox="tight")

        if self.show:
            plt.show()

    @staticmethod
    def parse_file(fn):

        x_data = []
        y_data = []

        with open(fn) as f:
            for line in f.readlines()[1:]:
                if line == "\n":
                    continue
                line.strip()
                line = line.strip("\n")
                line = line.split(",")
                x_data.append(int(line[0]))
                y_data.append(float(line[1]))
        return x_data, y_data

    def plot_all_ens_file(self, file, label=None):
        """Plot on new or provided ax"""

        x_data, y_data = self.parse_file(file)

        colour = self.get_next_colour()

        if min(y_data) < self.ymin:
            self.ymin = min(y_data)

        self.ax.scatter(x_data, y_data,
                        c=colour, label=label)
        # self.ax.plot(x_data, y_data,
        #              c=colour)

    def plot_best_ens_file(self, file, label=None):
        """Plot on new or provided ax"""

        x_data, y_data = self.parse_file(file)

        colour = self.get_next_colour()

        if min(y_data) < self.ymin:
            self.ymin = min(y_data)

        self.ax.scatter(x_data, y_data,
                        c=colour, label=label)
        self.ax.plot(x_data, y_data,
                     c=colour)

    def plot_file(self, file, label=None):
        """Plot on new or provided ax"""

        x_data, y_data = self.parse_file(file)

        colour = self.get_next_colour()

        if min(y_data) < self.ymin:
            self.ymin = min(y_data)

        self.ax.scatter(x_data, y_data,
                        c=colour, label=label)
        self.ax.plot(x_data, y_data,
                     c=colour)


    def get_next_colour(self):
        """Returns the next colour in the sequence"""
        colours = ["k", "r", "c", "b"]
        self.colour_idx += 1
        return colours[self.colour_idx]


if __name__ == "__main__":
    CLI_args = get_args()

    plotter = EnergyVsStep(infiles=CLI_args.ifiles,
                           outfile=CLI_args.ofile,
                           show=CLI_args.show,
                           ulim=CLI_args.ulim,
                           llim = CLI_args.llim)

    plotter.plot_files()

