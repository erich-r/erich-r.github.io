import importlib
import os
import unittest

ex = importlib.import_module("exercise1")

DATASET = os.path.join("dataset", "Carlsen.txt")
TEST_SET = os.path.join("dataset", "test_set.txt")

_games = None


def dataset_games():
    global _games
    if _games is None:
        _games = ex.parse_games(DATASET)
    return _games

def test_games():
   
    _games = ex.parse_games(TEST_SET)
    return _games



class TestParseGames(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.games = ex.parse_games(TEST_SET)

    def test_number_of_games(self):
        self.assertIsInstance(self.games, list)
        self.assertEqual(len(self.games), 3)

    def test_header_fields(self):
        game = self.games[0]
        self.assertEqual(game["event"], "Blitz Cup")
        self.assertEqual(game["site"], "Oslo NOR")
        self.assertEqual(game["date"], "2025.01.02")
        self.assertEqual(game["round"], "1")
        self.assertEqual(game["result"], "1-0")
        self.assertEqual(game["eco"], "C50")

    def test_players_are_nested_with_integer_elo(self):
        self.assertEqual(self.games[1]["white"], {"name": "Doe,J", "elo": 2700})
        self.assertEqual(self.games[1]["black"],
                         {"name": "Carlsen,M", "elo": 2831})

    def test_moves_are_pairs(self):
        self.assertEqual(self.games[0]["moves"],
                         [("e4", "e5"), ("Nf3", "Nc6"), ("Bc4", "Nf6"),
                          ("Ng5", "d5")])

    def test_game_ending_on_white_move(self):
        self.assertEqual(self.games[1]["moves"],
                         [("g4", "e5"), ("f3", "Qh4#")])
        self.assertEqual(self.games[2]["moves"][-1], ("h8=Q", None))

    def test_moves_over_several_lines(self):
        moves = self.games[2]["moves"]
        self.assertEqual(len(moves), 7)
        self.assertEqual(moves[3], ("O-O", "Nf6"))


class TestParseDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.games = dataset_games()

    def test_number_of_games(self):
        self.assertEqual(len(self.games), 1635)

    def test_first_game(self):
        game = self.games[0]
        self.assertEqual(game["event"], "Titled Tue 13th Aug Late")
        self.assertEqual(game["date"], "2024.08.13")
        self.assertEqual(game["white"], {"name": "Duda,J", "elo": 2733})
        self.assertEqual(game["black"], {"name": "Carlsen,M", "elo": 2832})
        self.assertEqual(game["result"], "0-1")
        self.assertEqual(game["eco"], "C10")
        self.assertEqual(len(game["moves"]), 52)
        self.assertEqual(game["moves"][0], ("e4", "e6"))
        self.assertEqual(game["moves"][-1], ("Rg8", "Ke6"))

    def test_last_game_ends_on_white_move(self):
        game = self.games[-1]
        self.assertEqual(game["date"], "2026.07.02")
        self.assertEqual(game["moves"][-1], ("Qh5+", None))

    def test_all_moves(self):
        self.assertEqual(sum(len(g["moves"]) for g in self.games), 76774)
        ending_on_white = [g for g in self.games if g["moves"][-1][1] is None]
        self.assertEqual(len(ending_on_white), 840)


class TestResultsByColor(unittest.TestCase):
    def test_carlsen(self):
        res = ex.results_by_color(dataset_games())
        self.assertEqual(res, {
            "white": {"wins": 593, "draws": 129, "losses": 106},
            "black": {"wins": 530, "draws": 156, "losses": 121},
        })

    def test_other_player(self):
        res = ex.results_by_color(dataset_games(), player="Firouzja,Alireza")
        self.assertEqual(res, {
            "white": {"wins": 10, "draws": 10, "losses": 22},
            "black": {"wins": 9, "draws": 14, "losses": 25},
        })


class TestTopOpponents(unittest.TestCase):
    def test_default_n_is_five(self):
        self.assertEqual(ex.top_opponents(dataset_games()), [
            ("Firouzja,Alireza", 90),
            ("Nakamura,Hi", 61),
            ("Lazavik,Denis", 56),
            ("Niemann,Hans Moke", 46),
            ("Caruana,F", 43),
        ])

    def test_top_ten(self):
        res = ex.top_opponents(dataset_games(), n=10)
        self.assertEqual(len(res), 10)
        self.assertEqual(res[5:], [
            ("Martinez Alcantara,Jose Eduardo", 40),
            ("Duda,J", 39),
            ("Erigaisi,Arjun", 37),
            ("Nepomniachtchi,I", 26),
            ("Sarana,A", 23),
        ])
    def test_top_opponents_alphabetically_sorted(self):
        games = test_games()
        top_2 = ex.top_opponents(games,2)
        self.assertEqual(top_2,[('Boh,A', 1), ('Doe,J', 1)])
      


class TestLongestGame(unittest.TestCase):
    def test_returns_longest(self):
        game = ex.longest_game(dataset_games())
        self.assertEqual(game["white"]["name"], "Klukin,Kirill")
        self.assertEqual(game["event"], "Titled Tue 13th Jan 2026")
        self.assertEqual(game["date"], "2026.01.13")
        self.assertEqual(len(game["moves"]), 148)



if __name__ == "__main__":
    unittest.main()
