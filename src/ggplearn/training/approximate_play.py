"""
back to square 1, more or less
  * first play is random-ish with policy player - gm_select
  * second uses puct player, just on selected state - gm_policy
  * third starting from the same state as policy was trained
    on (not the resultant state), policy player for score - gm_score
"""

import time

import numpy as np

from ggplib.util import log

from ggplib.player.gamemaster import GameMaster
from ggplib.db.helper import get_gdl_for_game

from ggplearn.util import attrutil

from ggplearn import msgdefs
from ggplearn.player.puctplayer import PUCTPlayer
from ggplearn.player.policyplayer import PolicyPlayer


class Runner(object):
    def __init__(self, conf):
        assert isinstance(conf, msgdefs.ConfigureApproxTrainer)

        attrutil.pprint(conf)

        self.conf = conf

        # create two game masters, one for the score playout, and one for the policy evaluation
        self.gm_select = GameMaster(get_gdl_for_game(self.conf.game), fast_reset=True)
        self.gm_policy = GameMaster(get_gdl_for_game(self.conf.game), fast_reset=True)
        self.gm_score = GameMaster(get_gdl_for_game(self.conf.game), fast_reset=True)

        # cache roles and count
        self.roles = self.gm_select.sm.get_roles()
        self.role_count = len(self.roles)

        # cache a local statemachine basestate (doesn't matter which gm it comes from)
        self.basestate = self.gm_select.sm.new_base_state()

        # add players to gamemasters
        for role in self.roles:
            self.gm_select.add_player(PolicyPlayer(self.conf.player_select_conf), role)

        for role in self.roles:
            self.gm_policy.add_player(PUCTPlayer(self.conf.player_policy_conf), role)

        for role in self.roles:
            self.gm_score.add_player(PolicyPlayer(self.conf.player_score_conf), role)

        # we want unique samples per generation, so store a unique_set here
        self.unique_states = set()
        self.reset_debug()

    def reset_debug(self):
        # debug times
        self.acc_time_for_play_one_game = 0
        self.acc_time_for_do_policy = 0
        self.acc_time_for_do_score = 0

    def add_to_unique_states(self, state):
        self.unique_states.add(state)

    def get_bases(self):
        self.gm_select.sm.get_current_state(self.basestate)
        return tuple(self.basestate.to_list())

    def play_one_game_for_selection(self):
        self.gm_select.reset()

        self.gm_select.start(meta_time=20, move_time=10)

        states = [(0, self.get_bases())]

        last_move = None
        depth = 1
        while not self.gm_select.finished():
            last_move = self.gm_select.play_single_move(last_move=last_move)
            states.append((depth, self.get_bases()))
            depth += 1

        # cleanup
        self.gm_select.play_to_end(last_move)
        return states

    def do_policy(self, state):
        for i, v in enumerate(state):
            self.basestate.set(i, v)

        self.gm_policy.reset()
        self.gm_policy.start(meta_time=30, move_time=10, initial_basestate=self.basestate)

        # self.last_move used in playout_state
        self.last_move = self.gm_policy.play_single_move(None)

        dists = []
        for ri in range(self.role_count):
            player = self.gm_policy.get_player(ri)
            dists.append(player.get_probabilities(ri, self.conf.temperature))

        return dists

    def do_score(self, state):
        for i, v in enumerate(state):
            self.basestate.set(i, v)

        self.gm_score.reset()
        self.gm_score.start(meta_time=30, move_time=10, initial_basestate=self.basestate)
        self.gm_score.play_to_end()

        # return a list of scores as we expect them in the neural network
        return [self.gm_score.get_score(role) / 100.0 for role in self.roles]

    def generate_sample(self):
        # debug
        log.debug("Entering generate_sample()")
        log.debug("unique_states: %s" % len(self.unique_states))

        start_time = time.time()
        states = self.play_one_game_for_selection()
        self.acc_time_for_play_one_game += time.time() - start_time

        game_length = len(states)
        log.debug("Done play_one_game_for_selection(), game_length %d" % game_length)

        shuffle_states = states[:]

        # pop the final state, as we don't want terminal states.  But keep in states intact
        shuffle_states.pop()
        np.random.shuffle(shuffle_states)

        duplicate_count = 0

        while shuffle_states:
            depth, state = shuffle_states.pop()

            if state in self.unique_states:
                duplicate_count += 1
                continue

            start_time = time.time()
            policy_dists = self.do_policy(state)
            log.debug("Done do_policy()")
            self.acc_time_for_do_policy += time.time() - start_time

            # start from state and not from what policy returns (which would add bias)
            start_time = time.time()
            final_score = self.do_score(state)
            log.debug("Done do_score()")
            self.acc_time_for_do_score += time.time() - start_time

            prev_state = states[depth - 1] if depth >= 1 else None
            sample = msgdefs.Sample(prev_state,
                                    state, policy_dists, final_score,
                                    depth, game_length)

            return sample, duplicate_count

        log.warning("Ran out of states, lots of duplicates.  Please do something about this, "
                    "shouldn't be playing with lots of duplicates.  Hack for now is to rerun.")
        return self.generate_sample()
