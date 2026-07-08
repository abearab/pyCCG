import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from synergy.combination import Bliss
from synergy.combination.loewe import Loewe


class SynergyData:
    """Data class for druh synergy analysis
    Assuming the experiment is a dose-titration experiment with 2 drugs
    happening in the middle 60 wells of a 96-well plate
    """
    def __init__(self, df, wide_treatment, narrow_treatment):
        self.df = df
        self.wide_treatment = wide_treatment
        self.narrow_treatment = narrow_treatment
    
    def _ave_replicates(self, value_col):
        df = self.df.copy()
        df = df.set_index(['cell_type',self.narrow_treatment,self.wide_treatment]).pivot(
            columns='replicate', values=value_col
        ).mean(axis=1).reset_index()
        df.columns = df.columns[:-1].tolist() + [value_col]
        
        return df

    def extract_single_treatment(self, treatment_col, treatment_dose=0):
        if treatment_col == self.wide_treatment:
            other_treatment_col = self.narrow_treatment
        elif treatment_col == self.narrow_treatment:
            other_treatment_col = self.wide_treatment
        else:
            raise ValueError(f"expected {self.wide_treatment} or {self.narrow_treatment}")

        df = self.df.copy()
        df = df.query(f'{other_treatment_col} == {treatment_dose}').drop(columns=[other_treatment_col])

        df = df.rename(columns={treatment_col: 'Compound Conc'})
        df.insert(0, 'treatment', treatment_col)

        df.query('`Compound Conc` > 0', inplace=True)

        return df
    
    def calculate_synergy(self, method='bliss', inplace=True, **kwargs):
        #TODO add checks here
        # if method not in ['bliss', 'loewe']:

        df = self.df.copy()

        # https://github.com/djwooten/synergy/issues/40
        # TODO: make sure how to normalize model fit for synergy values
        
        df[method] = np.nan

        for _,row in df.query(
            f'`{self.wide_treatment}` == 0 & `{self.narrow_treatment}` == 0').iterrows():
            single_plate = df.loc[
                (df.cell_type == row['cell_type']) & 
                (df.replicate == row['replicate']),:]

            if method == 'bliss':
                model = Bliss()
            elif method == 'loewe':
                model = Loewe(mode="delta_hsa")
            
            res = model.fit(
                d1=single_plate[self.wide_treatment].to_numpy(), 
                d2=single_plate[self.narrow_treatment].to_numpy(), 
                E=single_plate['viability'].to_numpy(),
                **kwargs
            )
            
            df.loc[single_plate.index, method] = res
        
        if inplace:
            self.df = df
        else:
            return df        
    
    def heatmap_df(self, value_col, query=None, dose_unit="nM"):
        """
        Return a pivoted DataFrame suitable for plotting a heatmap.

        Parameters
        ----------
        value_col : str
            Column to plot. If 'bliss' or 'loewe' and the column is not
            present, synergy values are calculated automatically.
        query : str, optional
            Pandas query string to subset the data before pivoting.

        Returns
        -------
        pandas.DataFrame
            Rows correspond to narrow_treatment doses and columns correspond
            to wide_treatment doses. Replicates are averaged.
        """
        df = self.df.copy()

        # Calculate synergy if needed
        if value_col in {"bliss", "loewe"} and value_col not in df.columns:
            df = self.calculate_synergy(method=value_col, inplace=False)

        # Average replicates
        if "replicate" in df.columns:
            df = (
                df.set_index(
                    ["cell_type", self.narrow_treatment, self.wide_treatment]
                )
                .pivot(columns="replicate", values=value_col)
                .mean(axis=1)
                .reset_index(name=value_col)
            )

        if query is not None:
            df = df.query(query)

        heatmap_df = (
            df.pivot_table(
                index=self.narrow_treatment,
                columns=self.wide_treatment,
                values=value_col,
                aggfunc="mean",
            )
            .sort_index()
            .sort_index(axis=1)
        )

        # Add units to the index and columns (round to 2 decimals)
        if dose_unit == "nM":
            heatmap_df.index = heatmap_df.index.round(2).astype(str) + "(nM)"
            heatmap_df.columns = heatmap_df.columns.round(2).astype(str) + "(nM)"
        elif dose_unit == "uM":
            heatmap_df.index = (heatmap_df.index / 1000).round(2).astype(str) + "(uM)"
            heatmap_df.columns = (heatmap_df.columns / 1000).round(2).astype(str) + "(uM)"
        else:
            raise ValueError(f"Unsupported unit: {dose_unit}, please use 'nM' or 'uM'.")

        return heatmap_df

    def plot_heatmap(
        self,
        ax=None,
        value_col="viability",
        xlabel="auto",
        ylabel="auto",
        cmap="PRGn",
        colorbar=True,
        remove_ticks=False,
        show_values=False,
        value_fontsize=8,
        value_fmt=".2f",
        value_color="black",
        # center_on_zero=False,
        vmin=None,
        vmax=None,
        drop_wide_dose=None,
        drop_narrow_dose=None,
        dose_scale=1000.0,
        dose_unit="uM",
        tick_fontsize=8,
        x_rotation=30,
        y_rotation=0,
        x_ha="center",
        y_va="center",
        title=None,
        **imshow_kwargs,
    ):
        """
        Plot a heatmap produced by `heatmap_df()`.

        Parameters
        ----------
        TBD

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        heatmap_df = self.heatmap_df(
            value_col=value_col, dose_unit=dose_unit
        )
        df = heatmap_df.copy()

        # Drop selected concentrations if requested
        if drop_wide_dose is not None:
            df = df.loc[:, ~df.columns.isin(drop_wide_dose)]
        if drop_narrow_dose is not None:
            df = df.loc[~df.index.isin(drop_narrow_dose), :]

        if df.empty:
            raise ValueError("heatmap_df is empty after applying dose filters")

        z = df.to_numpy(dtype=float)

        im = ax.imshow(
            z,
            origin="lower",
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **imshow_kwargs,
        )

        xlabels = heatmap_df.columns.to_numpy()
        ylabels = heatmap_df.index.to_numpy()

        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels)

        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)

        if xlabel == "auto":
            xlabel = self.wide_treatment
        if ylabel == "auto":
            ylabel = self.narrow_treatment

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        if remove_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis="both", which="both", length=0)
        else:
            ax.set_xticks(np.arange(len(xlabels)))
            ax.set_xticklabels(
                xlabels,
                fontsize=tick_fontsize,
                rotation=x_rotation,
                ha=x_ha,
            )

            ax.set_yticks(np.arange(len(ylabels)))
            ax.set_yticklabels(
                ylabels,
                fontsize=tick_fontsize,
                rotation=y_rotation,
                va=y_va,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        if show_values:
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    value = z[i, j]
                    if np.isnan(value):
                        continue

                    ax.text(
                        j,
                        i,
                        format(value, value_fmt),
                        ha="center",
                        va="center",
                        fontsize=value_fontsize,
                        color=value_color,
                    )
        
        if colorbar:
            cbar = ax.figure.colorbar(
                im,
                ax=ax,
                shrink=0.3,
                aspect=25,
                pad=0.02,
            )

            cbar.set_label(value_col, rotation=270, labelpad=15)
            cbar.ax.tick_params(labelsize=tick_fontsize)

            ticks = [vmin, (vmin + vmax) / 2, vmax]

            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t:.2g}" for t in ticks])        
        
        return ax


def read_synergy_data(filename):
    data = pd.read_csv(filename,sep='\t', header=0, index_col=None, skiprows=1)

    #TODO: come up with a better way to get the treatment names
    wide_treatment = pd.read_csv(filename,sep='\t', nrows=1, header=None).T.dropna().values[0][0]
    narrow_treatment = data.columns[2]

    df = data.melt(
        id_vars=['cell_type','replicate',narrow_treatment], 
        value_name='ctg', 
        var_name=wide_treatment
    )
    df[wide_treatment] = df[wide_treatment].astype(float) * 1000 # convert to nM
    df[narrow_treatment] = df[narrow_treatment].astype(float) * 1000 # convert to nM

    # round to 3 decimals
    for col in ['ctg', narrow_treatment, wide_treatment]:
        df[col] = df[col].round(decimals=3)

    # calculate relative CTG values (normalized to baseline, i.e. no treatment)
    df['baseline'] = np.nan

    for _,row in df.query(
        f'`{wide_treatment}` == 0 & `{narrow_treatment}` == 0').iterrows():
        df.loc[
            (df.cell_type == row['cell_type']) & 
            (df.replicate == row['replicate']), 
            'baseline'] = row['ctg']

    # viability
    df['viability'] = (df['ctg'] / df['baseline'])
    del df['baseline']

    return SynergyData(df, wide_treatment, narrow_treatment)
