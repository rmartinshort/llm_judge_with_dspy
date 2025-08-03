import seaborn as sns
import matplotlib.pyplot as plt

def plot_judge_results(df, judge_count_col="satisfied"):
    ax = sns.countplot(
        df,
        y=judge_count_col,
        palette="viridis",
        hue=judge_count_col
    )

    total = len(df)

    for p in ax.patches:
        count = p.get_width()
        percent = 100 * count / total
        if percent > 0:
            x = count + 0.2  # Offset the text position a bit from the end of the bar
            y = p.get_y() + p.get_height() / 2
            ax.text(x, y, f"{percent:.1f}%", va="center")

    plt.show()