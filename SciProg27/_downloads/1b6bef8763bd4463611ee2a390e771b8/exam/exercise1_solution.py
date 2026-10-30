def parse_games(path):
    games = []
    game = None
    move_text = []

    def format_moves_and_add_game(game, move_text):
        tokens = " ".join(move_text).split()
        tokens = tokens[:-1]  # the last token is the result
        moves = []
        for token in tokens:
            if "." in token:  # 12.Nf3 starts a new move pair
                moves.append([token.split(".", 1)[1], None])
            else:
                moves[-1][1] = token
        game["moves"] = [tuple(m) for m in moves]
        games.append(game)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "===NEW GAME===":
                if game is not None:
                    format_moves_and_add_game(game, move_text)
                game = {"white": {}, "black": {}}
                move_text = []
            elif line.startswith("["):
                key, value = line[1:-1].split(" ", 1)
                value = value.strip('"')
                if key in ("White", "Black"):
                    game[key.lower()]["name"] = value
                elif key in ("WhiteElo", "BlackElo"):
                    game[key[:5].lower()]["elo"] = int(value)
                else:
                    game[key.lower()] = value
            elif line:
                move_text.append(line)
    if game is not None:
        format_moves_and_add_game(game, move_text)
    return games


def player_points(game, player="Carlsen,M"):
    if game["result"] == "1/2-1/2":
        return "tie"
    winner = "white" if game["result"] == "1-0" else "black"
    return "won" if game[winner]["name"] == player else "lost"

def results_by_color(games, player="Carlsen,M"):
    res = {color: {"wins": 0, "draws": 0, "losses": 0}
           for color in ("white", "black")}
    for game in games:
        for color in ("white", "black"):
            if game[color]["name"] == player:
                outcome = player_points(game, player)
                key = {"won": "wins", "tie": "draws", "lost": "losses"}[outcome]
                res[color][key] += 1
    return res

def top_opponents(games, n=5, player="Carlsen,M"):
    counts = {}
    for game in games:
        for color in ("white", "black"):
            name = game[color]["name"]
            if name != player:
                if name in counts:
                    counts[name] +=1
                else:
                    counts[name] = 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:n]


def longest_game(games):
    longest = games[0]
    for game in games[1:]:
        if len(game["moves"]) > len(longest["moves"]):
            longest = game
    
    print("I have found the longest game:")
    print(f"{longest['white']['name']} vs {longest['black']['name']}, "
          f"{longest['event']} ({longest['date']}): "
          f"{len(longest['moves'])} moves")
    return longest

