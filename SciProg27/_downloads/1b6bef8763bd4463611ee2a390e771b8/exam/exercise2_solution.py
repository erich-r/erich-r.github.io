from datetime import datetime

import matplotlib.pyplot as plt

from exercise1 import parse_games, player_points


def player_color(game, player="Carlsen,M"):
    return "white" if game["white"]["name"] == player else "black"

def plot_elo_progress(games, player="Carlsen,M"):
    dates, elos = [], []
    for game in games:
        date = datetime.strptime(game["date"], "%Y.%m.%d").date()
        dates.append(date)
        elos.append(game[player_color(game, player)]["elo"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, elos)
    ax.axhline(max(elos), linestyle="dashed", color="red", label="peak")
    ax.set_title(f"Elo rating of {player}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Elo")
    ax.legend()
    fig.savefig("elo_progress.png")

def plot_results_pies(games, player="Carlsen,M"):
    counts = {"White": [0, 0, 0], "Black": [0, 0, 0]}  # wins, draws, losses
    result_to_index = {"won": 0, "tie": 1, "lost": 2}

    for game in games:
        color = player_color(game, player).capitalize()
        index = result_to_index[player_points(game, player)]
        counts[color][index] += 1
    counts["White + Black"] = [w + b for w, b in
                               zip(counts["White"], counts["Black"])]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, values) in zip(axes, counts.items()):
        ax.pie(values, labels=["Wins", "Draws", "Losses"], autopct="%1.1f%%",
               colors=["tab:green", "tab:gray", "tab:red"])
        ax.set_title(title)
    fig.suptitle(f"Results of {player}")
    fig.savefig("results_pies.png")


def plot_top_openings(games, n=10, player="Carlsen,M"):
    stats = {}
    result_to_index = {"won": 0, "tie": 1, "lost": 2}
    for game in games:
        index = result_to_index[player_points(game, player)]
        if game["eco"] not in stats:
            stats[game["eco"]] = [0, 0, 0]
        stats[game["eco"]][index] += 1
        
    top = sorted(stats, key=lambda eco: (-sum(stats[eco]), eco))[:n]

    wins = [stats[eco][0] for eco in top]
    draws = [stats[eco][1] for eco in top]
    losses = [stats[eco][2] for eco in top]
    bottoms = [w + d for w, d in zip(wins, draws)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top, wins, label="Wins", color="tab:green")
    ax.bar(top, draws, bottom=wins, label="Draws", color="tab:gray")
    ax.bar(top, losses, bottom=bottoms, label="Losses", color="tab:red")
    ax.set_title(f"Results of {player} in the {n} most played openings")
    ax.set_xlabel("ECO code")
    ax.set_ylabel("Games")
    ax.legend()
    fig.savefig("top_openings.png")

if __name__ == "__main__":
    plot_elo_progress(parse_games("../exam/dataset/Carlsen.txt"))
    plot_results_pies(parse_games("../exam/dataset/Carlsen.txt"))
    plot_top_openings(parse_games("../exam/dataset/Carlsen.txt"))