import time

from ggplib.player import get
from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game

from ggplearn import msgdefs
from ggplearn.player.puctplayer import PUCTPlayer
from ggplearn.player.policyplayer import PolicyPlayer

import py.test

ITERATIONS = 1

current_gen = "testgen_normal_1"
default_puct_config = msgdefs.PUCTPlayerConf(generation=current_gen,
                                             playouts_per_iteration=42,
                                             playouts_per_iteration_noop=1)

default_policy_config = msgdefs.PolicyPlayerConf(generation=current_gen)


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggplearn.util.keras import constrain_resources
    constrain_resources()


def test_reversi_tournament():
    py.test.skip("no reversi right now")
    gm = GameMaster(get_gdl_for_game("reversi"))

    # add two players
    pymcs = get.get_player("pymcs")
    pymcs.max_run_time = 0.25

    random = get.get_player("pyrandom")
    nn0 = PolicyPlayer("no-scores1")

    gm.add_player(nn0, "black")
    gm.add_player(random, "red")

    acc_black_score = 0
    acc_red_score = 0
    for _ in range(ITERATIONS):
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_black_score += gm.scores["black"]
        acc_red_score += gm.scores["red"]

    print "black_score", gm.players_map["black"].name, acc_black_score
    print "red_score", gm.players_map["red"].name, acc_red_score


def test_policy_tournament():
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    # add two players
    pymcs = get.get_player("pymcs")
    pymcs.max_run_time = 0.1

    black = PolicyPlayer(conf=default_policy_config)

    gm.add_player(pymcs, "white")
    gm.add_player(black, "black")

    acc_black_score = 0
    acc_red_score = 0
    for _ in range(ITERATIONS):
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_black_score += gm.scores["black"]
        acc_red_score += gm.scores["white"]

    print "white_score", gm.players_map["white"].name, acc_red_score
    print "black_score", gm.players_map["black"].name, acc_black_score


def test_puct_tournament():
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    # add two players
    pymcs = get.get_player("pymcs")
    pymcs.max_run_time = 0.1

    black = PUCTPlayer(conf=default_puct_config)

    gm.add_player(pymcs, "white")
    gm.add_player(black, "black")

    acc_black_score = 0
    acc_red_score = 0
    for _ in range(ITERATIONS):
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_black_score += gm.scores["black"]
        acc_red_score += gm.scores["white"]

    print "white_score", gm.players_map["white"].name, acc_red_score
    print "black_score", gm.players_map["black"].name, acc_black_score


def test_tournament_2():
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    # add two players
    white = PUCTPlayer(conf=default_puct_config)
    black = PolicyPlayer(conf=default_policy_config)

    gm.add_player(black, "white")
    gm.add_player(white, "black")

    acc_black_score = 0
    acc_red_score = 0
    for _ in range(ITERATIONS):
        gm.reset()
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_black_score += gm.scores["black"]
        acc_red_score += gm.scores["white"]

    print "white_score", gm.players_map["white"].name, acc_red_score
    print "black_score", gm.players_map["black"].name, acc_black_score


def test_fast_plays():
    ''' very fast rollouts, basically this config of puct player is a policy player '''
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    import attr
    conf = msgdefs.PUCTPlayerConf(**attr.asdict(default_puct_config))
    conf.verbose = False

    # just checking that we haven't modified default
    assert not conf.verbose and default_puct_config.verbose

    conf.playouts_per_iteration = 1
    conf.playouts_per_iteration_noop = 0
    conf.dirichlet_noise_alpha = -1
    conf.expand_root = -1
    print conf

    # add two players
    white = PUCTPlayer(conf=conf)
    black = PUCTPlayer(conf=conf)

    gm.add_player(white, "white")
    gm.add_player(black, "black")

    acc_black_score = 0
    acc_red_score = 0
    s = time.time()
    for _ in range(ITERATIONS):
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_black_score += gm.scores["black"]
        acc_red_score += gm.scores["white"]

        print gm.depth

    print "time taken", time.time() - s
    print "white_score", gm.players_map["white"].name, acc_red_score
    print "black_score", gm.players_map["black"].name, acc_black_score


def test_not_taking_win():
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    # add two players
    gm.add_player(PUCTPlayer(default_puct_config), "white")
    gm.add_player(PUCTPlayer(default_puct_config), "black")

    str_state = """ (true (control black))
    (true (cellHolds 8 8 black)) (true (cellHolds 8 7 black)) (true (cellHolds 8 2 white))
    (true (cellHolds 8 1 white)) (true (cellHolds 7 8 black)) (true (cellHolds 6 7 white))
    (true (cellHolds 6 2 white)) (true (cellHolds 6 1 white)) (true (cellHolds 5 4 black))
    (true (cellHolds 5 3 black)) (true (cellHolds 5 2 white)) (true (cellHolds 5 1 white))
    (true (cellHolds 4 8 black)) (true (cellHolds 4 7 black)) (true (cellHolds 4 1 white))
    (true (cellHolds 3 8 black)) (true (cellHolds 3 6 black)) (true (cellHolds 3 2 white))
    (true (cellHolds 2 8 black)) (true (cellHolds 2 7 black)) (true (cellHolds 2 2 white))
    (true (cellHolds 2 1 white)) (true (cellHolds 1 8 black)) (true (cellHolds 1 7 black))
    (true (cellHolds 1 2 white)) (true (cellHolds 1 1 white)) """

    gm.start(meta_time=30, move_time=5,
             initial_basestate=gm.convert_to_base_state(str_state))

    last_move = gm.play_single_move(last_move=None)
    assert last_move[1] == "(move 7 8 6 7)"


def test_choose_policy_random():
    # ITERATIONS = 100
    gm = GameMaster(get_gdl_for_game("breakthrough"))

    conf = msgdefs.PolicyPlayerConf(name="white", generation="testgen_normal_1", verbose=False)
    conf.choose_exponential_scale = 0.15
    white = PolicyPlayer(conf)

    conf = msgdefs.PolicyPlayerConf(name="black_", generation="testgen_normal_1", verbose=False)
    conf.choose_exponential_scale = 0.15
    black = PolicyPlayer(conf)

    gm.add_player(white, "white")
    gm.add_player(black, "black")

    gm.reset()
    gm.start(meta_time=30, move_time=15)
    gm.play_to_end()

    acc_white_score = 0
    acc_black_score = 0
    game_depths = []
    for _ in range(ITERATIONS):
        gm.reset()
        gm.start(meta_time=30, move_time=15)
        gm.play_to_end()

        acc_white_score += gm.scores["white"]
        acc_black_score += gm.scores["black"]

        game_depths.append(gm.depth)

    print "white_score", gm.players_map["white"].name, acc_white_score
    print "black_score", gm.players_map["black"].name, acc_black_score
    print game_depths

def test_noops_are_the_only_move():
    gm = GameMaster(get_gdl_for_game("reversi"))

    current_gen = "v3_gen_small_1"
    conf = msgdefs.PUCTPlayerConf(generation=current_gen,
                                  verbose=True,
                                  playouts_per_iteration=42,
                                  playouts_per_iteration_noop=1)

    # add two players
    gm.add_player(PUCTPlayer(conf), "black")
    gm.add_player(PUCTPlayer(conf), "red")

    str_state = """ (true (control red)) (true (cell 8 8 red)) (true (cell 8 7 red))
    (true (cell 8 6 red)) (true (cell 8 5 red)) (true (cell 8 4 red)) (true (cell 8 3 red))
    (true (cell 8 2 red)) (true (cell 8 1 black)) (true (cell 7 8 red)) (true (cell 7 7 red))
    (true (cell 7 6 red)) (true (cell 7 5 red)) (true (cell 7 4 red)) (true (cell 7 3 red))
    (true (cell 7 2 red)) (true (cell 7 1 black)) (true (cell 6 8 red)) (true (cell 6 7 red))
    (true (cell 6 6 red)) (true (cell 6 5 red)) (true (cell 6 4 red))
    (true (cell 6 3 black)) (true (cell 6 2 red)) (true (cell 6 1 red))
    (true (cell 5 8 red)) (true (cell 5 7 red)) (true (cell 5 6 red)) (true (cell 5 5 red))
    (true (cell 5 4 red)) (true (cell 5 3 black)) (true (cell 5 2 red)) (true (cell 5 1 black))
    (true (cell 4 8 black)) (true (cell 4 7 red)) (true (cell 4 6 black)) (true (cell 4 5 red))
    (true (cell 4 4 black)) (true (cell 4 3 black)) (true (cell 4 2 red)) (true (cell 4 1 black))
    (true (cell 3 8 black)) (true (cell 3 7 black)) (true (cell 3 6 black)) (true (cell 3 5 black))
    (true (cell 3 4 red)) (true (cell 3 3 black)) (true (cell 3 2 black)) (true (cell 3 1 black))
    (true (cell 2 7 red)) (true (cell 2 6 black)) (true (cell 2 5 black)) (true (cell 2 4 black))
    (true (cell 2 3 black)) (true (cell 2 2 black)) (true (cell 1 8 black)) (true (cell 1 7 red))
    (true (cell 1 5 red)) """

    gm.start(meta_time=30, move_time=5,
             initial_basestate=gm.convert_to_base_state(str_state))
    gm.play_to_end()
