''' takes forever to create data - so not a py.test '''

import os
import random

from ggplearn.util import attrutil

from ggplearn import msgdefs

MAX_STATES_FOR_ROLLOUT = 500


class Rollout(object):
    def __init__(self, game_info):
        self.game_info = game_info
        self.sm = game_info.get_sm()

        self.states = [self.sm.new_base_state() for _ in range(MAX_STATES_FOR_ROLLOUT)]
        self.role_count = len(self.sm.get_roles())

        # get and cache fast move
        self.static_joint_move = self.sm.get_joint_move()
        self.lookahead_joint_move = self.sm.get_joint_move()

        self.depth = None
        self.legals = None
        self.scores = None

        # this is really approximate, works for some games
        assert len(self.game_info.model.roles) == 2
        role0, role1 = self.game_info.model.roles
        self.piece_counts = []
        for b in self.game_info.model.bases:
            if 'control' in b:
                self.piece_counts.append(None)
            elif role0 in b and role1 not in b:
                self.piece_counts.append(role0)
            elif role1 in b and role0 not in b:
                self.piece_counts.append(role1)
            else:
                self.piece_counts.append(None)

    def count_states(self, basestate, ri):
        role = self.game_info.model.roles[ri]
        total = 0
        for i in range(basestate.len()):
            if basestate.get(i) == 0:
                continue
            if self.piece_counts[i] == role:
                total += 1
        return total

    def reset(self):
        self.sm.reset()

        # (legals0, legals1)
        self.legals = []
        self.sm.get_current_state(self.states[0])

    def make_data(self, unique_states):
        state = None
        for _ in range(self.depth):
            d = random.randrange(self.depth)
            a_state = tuple(self.states[d].to_list())

            final_score = [s / 100.0 for s in self.scores]
            all_main_legals = self.legals[d]

            if a_state not in unique_states:
                state = a_state
                unique_states.add(state)
                break

        if state is None:
            return None

        self.sm.update_bases(self.states[d])

        policy_dists = []
        for ri in range(self.role_count):
            main_legal = all_main_legals[ri]
            ls = self.sm.get_legal_state(ri)
            total = float(ls.get_count() * 4)
            legals = ls.to_list()
            policy_dist = [(l, (1 / total)) for l in ls.to_list()]
            policy_dist[legals.index(main_legal)] = (main_legal, (ls.get_count() * 3 + 1) / total)

            policy_dists.append(policy_dist)

        # now we can create a sample :)
        return msgdefs.Sample(None, state, policy_dists, final_score, d, self.depth)

    def get_current_state(self):
        return self.states[self.depth]

    def choose_move(self, role_index):
        ls = self.sm.get_legal_state(role_index)
        if ls.get_count() == 1:
            return ls.get_legal(0)

        other_role_index = 0 if role_index else 1

        # set other move
        ls_other = self.sm.get_legal_state(other_role_index)
        # XXX dont do this
        assert ls_other.get_count() == 1

        self.lookahead_joint_move.set(other_role_index, ls_other.get_legal(0))

        # steal new state for now...
        next_state = self.states[self.depth + 1]

        best_moves = []
        best_count = 10000

        # want to reduce this
        for ii in range(ls.get_count()):
            legal = ls.get_legal(ii)
            self.lookahead_joint_move.set(role_index, legal)
            self.sm.next_state(self.lookahead_joint_move, next_state)

            # move forward and see if we won the game?
            self.sm.update_bases(next_state)
            if self.sm.is_terminal():
                if self.sm.get_goal_value(role_index) == 100:
                    # return this move (but fix the state of statemachine first)
                    self.sm.update_bases(self.get_current_state())
                    return legal

            count = self.count_states(next_state, other_role_index)
            if count < best_count:
                best_moves = [legal]
                best_count = count
            elif count == best_count:
                best_moves.append(legal)

            # revert statemachine
            self.sm.update_bases(self.get_current_state())

        return random.choice(best_moves)

    def go(self):
        self.reset()

        self.depth = 0
        self.legals = []
        while True:
            if self.sm.is_terminal():
                break

            # play move
            legal_move = []
            for ri in range(self.role_count):
                choice = self.choose_move(ri)
                legal_move.append(choice)
                self.static_joint_move.set(ri, choice)

            self.legals.append(legal_move)

            # borrow the next state (side affect of assigning it)
            next_state = self.states[self.depth + 1]
            self.sm.next_state(self.static_joint_move, next_state)
            self.sm.update_bases(next_state)

            self.depth += 1

        self.scores = []
        for ii, _ in enumerate(self.sm.get_roles()):
            self.scores.append(self.sm.get_goal_value(ii))


def nn_train_random_generated():
    ' not a unit test - like can take over a few hours ! '
    CREATE_FILE = False
    ACTUALLY_TRAIN = True
    SAMPLE_COUNT = 25000

    train_conf = msgdefs.TrainNNRequest()
    train_conf.game = "reversi"

    train_conf.network_size = "tiny"
    train_conf.generation_prefix = "v4_"
    train_conf.store_path = os.path.join(os.environ["GGPLEARN_PATH"], "data", train_conf.game, "v4")

    # uses previous network
    train_conf.use_previous = False
    train_conf.next_step = 1

    train_conf.validation_split = 0.8
    train_conf.batch_size = 32
    train_conf.epochs = 30
    train_conf.max_sample_count = SAMPLE_COUNT
    attrutil.pprint(train_conf)

    if CREATE_FILE:
        from ggplib.db import lookup
        game_info = lookup.by_name(train_conf.game)
        r = Rollout(game_info)

        # perform a million rollouts
        unique_states = set()

        try:
            samples = []
            for i in range(SAMPLE_COUNT):
                r.go()
                sample = None
                for _ in range(10):
                    sample = r.make_data(unique_states)
                    if sample is not None:
                        break

                if sample is None:
                    print "DUPE NATION", i
                    continue

                samples.append(sample)

                if i % 5000 == 0:
                    print i
        except KeyboardInterrupt:
            pass

        gen = msgdefs.Generation()
        gen.game = train_conf.game
        gen.with_score_generation = 0
        gen.with_policy_generation = 0
        gen.num_samples = len(samples)
        gen.samples = samples

        filename = os.path.join(train_conf.store_path, "gendata_%s_0.json" % gen.game)
        with open(filename, 'w') as open_file:
            open_file.write(attrutil.attr_to_json(gen))

    if ACTUALLY_TRAIN:
        # import here so can run with pypy wihtout hitting import keras issues
        from ggplearn.training.nn_train import parse_and_train
        parse_and_train(train_conf)


def v3_retrain():
    conf = msgdefs.TrainNNRequest()
    conf.game = "breakthrough"

    conf.network_size = "normal"
    conf.generation_prefix = "v3x_"
    conf.store_path = "/home/rxe/working/ggplearn/data/breakthrough/v3"

    # uses previous network
    conf.use_previous = False
    conf.next_step = 25

    conf.validation_split = 0.8
    conf.batch_size = 64
    conf.epochs = 32
    conf.max_sample_count = 250000

    # import here so can run with pypy wihtout hitting import keras issues
    from ggplearn.training.nn_train import parse_and_train
    parse_and_train(conf)


if __name__ == "__main__":
    from ggplearn.util.main import main_wrap

    # main_wrap(v3_retrain)
    main_wrap(nn_train_random_generated)
