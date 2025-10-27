import os
import math
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, save
import matplotlib.pyplot as plt

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVDataLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File not found: {self.path}")
        df = pd.read_csv(self.path)
        df.columns = [c.strip().lower() for c in df.columns]  # Normalize names
        logger.info("Loaded %s (shape=%s)", self.path, df.shape)
        return df

class IdealFunctionMatcher:
    def __init__(self, train_df, ideal_df, test_df):
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.test_df = test_df
        self.best_map = {}
        self.max_deviation = {}

    def find_best(self):
        train_cols = [c for c in self.train_df.columns if c != 'x']
        ideal_cols = [c for c in self.ideal_df.columns if c != 'x']

        for tcol in train_cols:
            best_function = None
            min_mse = float('inf')

            for icol in ideal_cols:
                merged = self.train_df[['x', tcol]].merge(
                    self.ideal_df[['x', icol]], on='x', how='inner'
                )

                # ✅ SAFETY CHECK to prevent KeyError
                if merged.empty or tcol not in merged or icol not in merged:
                    continue

                mse = np.mean((merged[tcol] - merged[icol]) ** 2)
                if mse < min_mse:
                    min_mse = mse
                    best_function = icol

            if best_function:
                self.best_map[tcol] = best_function
                merged_final = self.train_df[['x', tcol]].merge(
                    self.ideal_df[['x', best_function]], on='x', how='inner'
                )
                self.max_deviation[tcol] = float(
                    np.max(np.abs(merged_final[tcol] - merged_final[best_function]))
                )
                logger.info(f"{tcol} -> {best_function} (MSE={min_mse:.6f}, max_dev={self.max_deviation[tcol]:.6f})")

    def map_test_points(self):
        results = []
        ideal_by_x = self.ideal_df.set_index('x')

        for _, row in self.test_df.iterrows():
            x_val, y_val = row['x'], row['y']
            if x_val not in ideal_by_x.index:
                results.append({'x': x_val, 'y': y_val, 'delta_y': None, 'ideal_function': None})
                continue

            assigned = False
            for tcol, icol in self.best_map.items():
                ideal_y = ideal_by_x.loc[x_val, icol]
                delta = abs(y_val - ideal_y)
                allowed = math.sqrt(2) * self.max_deviation[tcol]

                if delta <= allowed:
                    results.append({'x': x_val, 'y': y_val, 'delta_y': delta, 'ideal_function': icol})
                    assigned = True
                    break

            if not assigned:
                results.append({'x': x_val, 'y': y_val, 'delta_y': None, 'ideal_function': None})

        return pd.DataFrame(results)

    def save_to_sqlite(self, df, db_name="results.db"):
        engine = create_engine(f"sqlite:///{db_name}", echo=False)
        self.train_df.to_sql("training", engine, if_exists="replace", index=False)
        self.ideal_df.to_sql("ideal_functions", engine, if_exists="replace", index=False)
        df.to_sql("test_results", engine, if_exists="replace", index=False)
        logger.info("Saved to database: results.db")

    def plot_bokeh(self):
        os.makedirs("plots_bokeh", exist_ok=True)
        for tcol, icol in self.best_map.items():
            p = figure(title=f"{tcol} vs {icol}", width=800, height=400)
            p.scatter(self.train_df['x'], self.train_df[tcol], size=5, legend_label=f"Train {tcol}")
            p.line(self.ideal_df['x'], self.ideal_df[icol], legend_label=f"Ideal {icol}", line_width=2)
            output_file(f"plots_bokeh/{tcol}_vs_{icol}.html")
            save(p)

    def plot_png(self):
        os.makedirs("plots_png", exist_ok=True)
        for tcol, icol in self.best_map.items():
            plt.figure(figsize=(8, 4))
            plt.scatter(self.train_df['x'], self.train_df[tcol], s=10, label=f"Train {tcol}")
            plt.plot(self.ideal_df['x'], self.ideal_df[icol], label=f"Ideal {icol}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"{tcol} vs {icol}")
            plt.legend()
            plt.grid()
            plt.savefig(f"plots_png/{tcol}_vs_{icol}.png", dpi=150)
            plt.close()

def main():
    # Load data
    train = CSVDataLoader("train.csv").load()
    ideal = CSVDataLoader("ideal.csv").load()
    test = CSVDataLoader("test.csv").load()

    matcher = IdealFunctionMatcher(train, ideal, test)
    matcher.find_best()

    results = matcher.map_test_points()
    results.to_csv("test_results.csv", index=False)
    matcher.save_to_sqlite(results)
    matcher.plot_bokeh()
    matcher.plot_png()

    print("\n✅ DONE — Best Matches:")
    for tcol, icol in matcher.best_map.items():
        print(f"{tcol} → {icol}  |  max deviation = {matcher.max_deviation[tcol]:.6f}")

if __name__ == "__main__":
    main()
