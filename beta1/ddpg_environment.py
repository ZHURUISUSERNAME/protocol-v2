import random as r

import numpy as np
import pandas as pd

from ddpg_options import get_default_object

debug = not True


class Environment:
    '''
        state = [solar, load, energy_level, outdoor temperature, indoor temperature, thermalDisturance, price, time_step]
    '''
    env_options = None

    def __init__(self, env_options):
        self.env_options = env_options
        self.eta = self.env_options.eta  # (battery efficiency)
        self.Gamma = self.env_options.gamma
        self.start = self.env_options.start  # pick season
        self.day_chunk = self.env_options.day_chunk
        self.training_time = self.env_options.total_years
        self.df_solar = pd.read_csv(U"./Data/solar_double.csv")
        self.df_solar = self.df_solar[self.start: self.start + self.day_chunk].reset_index()
        self.df_load = pd.read_csv(U"./Data/base_load_modified.csv")
        self.df_load = self.df_load[self.start: self.start + self.day_chunk].reset_index()
        self.df_outTemp = pd.read_csv(U"./Data/temp_modified.csv")
        self.df_outTemp = self.df_outTemp[self.start: self.start + self.day_chunk].reset_index()
        self.df_price = pd.read_csv(U"./Data/price_modified.csv")
        self.df_price = self.df_price[self.start: self.start + self.day_chunk].reset_index()
        self.current_state = None
        self.day_number = 0
        self.time_step = 0
        self.DepriciationParam = self.env_options.DepriciationParam
        self.TempViolation = []

    def NormalizedPreprocess(self, state):
        state[0] = (state[0] - 0) / (5.31)
        state[1] = (state[1] - 0) / (11.2233)
        state[2] = (state[2] - self.env_options.E_min) / (self.env_options.E_max - self.env_options.E_min)
        state[3] = (state[3] - 61.27) / (107.58 - 61.27)
        state[4] = (state[4] - self.env_options.T_min) / (self.env_options.T_max - self.env_options.T_min)
        state[5] = (state[5] - 0.0808) / (0.325 - 0.0808)
        state[6] = (state[6]) / 23
        return state.reshape(1, 7)

    def reset(self, reset_day=False):
        # initial_state = self.get_initial_state(0, self.env_options.E_init
        # if self.current_state is None else self.current_state[2])
        initial_state = self.get_initial_state(0, self.env_options.E_init, self.env_options.IndoorTemp_init)
        self.current_state = initial_state
        self.time_step = 0
        if reset_day:
            self.day_number = 0
        return initial_state

    def ChooseRandomParameter(self, low, high):
        self.start = int(r.uniform(low, high))
        # print('day number:',self.start)
        self.df_solar = pd.read_csv(U"./Data/solar_double.csv")
        self.df_solar = self.df_solar[self.start: self.start + self.day_chunk].reset_index()
        self.df_load = pd.read_csv(U"./Data/base_load_modified.csv")
        self.df_load = self.df_load[self.start: self.start + self.day_chunk].reset_index()
        self.df_outTemp = pd.read_csv(U"./Data/temp_modified.csv")
        self.df_outTemp = self.df_outTemp[self.start: self.start + self.day_chunk].reset_index()
        self.df_price = pd.read_csv(U"./Data/price_modified.csv")
        self.df_price = self.df_price[self.start: self.start + self.day_chunk].reset_index()

    def get_initial_state(self, day_number, e_init, IndoorTemp_init):
        '''
            Set's the initialState (0th hour) for day_number.
            day_number
        '''
        solar = float(self.df_solar[self.get_key(0)][day_number])
        load = float(self.df_load[self.get_key(0)][day_number])
        energy_level = e_init
        outdoor_temperature = float(self.df_outTemp[self.get_key(0)][day_number])
        indoor_temperature = IndoorTemp_init
        price = float(self.df_price[self.get_key(0)][day_number])

        return [solar, load, energy_level, outdoor_temperature, indoor_temperature, price, 0]

    def step(self, action):
        next_state, reward_original, c1_, c2_, c3_ = self.get_next_state(self.day_number, self.time_step,
                                                                         self.current_state, action)
        self.time_step += 1
        if self.time_step > 23:
            self.day_number = self.day_number + 1
            self.time_step = 0
        if self.day_number >= self.day_chunk:
            self.day_number = 0
        self.current_state = next_state
        return next_state, reward_original, c1_, c2_, c3_

    def get_next_state(self, day_number, time_step, state_k, action_k):

        current_solar = state_k[0]
        current_load = state_k[1]
        current_energy = state_k[2]
        current_outdoorTemp = state_k[3]
        current_indoorTemp = state_k[4]
        current_netload = current_load - current_solar

        if action_k[0] >= 0:
            p_charge, p_discharge = np.clip(action_k[0], 0, min((self.env_options.E_max - current_energy) / self.eta,
                                                                self.env_options.P_cap)), 0.0
            if p_charge == 0:
                action_k[0] = 0
        else:
            p_charge, p_discharge = 0.0, np.clip(action_k[0], max(-self.env_options.P_cap,
                                                                  self.eta * (self.env_options.E_min - current_energy)),
                                                 0)
            if p_discharge == 0:
                action_k[0] = 0

        e_next = current_energy + self.eta * p_charge + p_discharge / self.eta
        if abs(e_next - self.env_options.E_min) < 1e-8:
            e_next = self.env_options.E_min
        if abs(e_next - self.env_options.E_max) < 1e-8:
            e_next = self.env_options.E_max
        if e_next > self.env_options.E_max or e_next < self.env_options.E_min:
            print('battery overflow!!!');

        if current_indoorTemp <= self.env_options.T_min:
            action_k[1] = 0
            # print('Lower limit touch!!!')
        elif current_indoorTemp > self.env_options.T_max:
            action_k[1] = np.clip(action_k[1], 0.1, self.env_options.hvac_p_cap)
        T_next = self.env_options.Ewuxilong * current_indoorTemp + (1 - self.env_options.Ewuxilong) * (
                    current_outdoorTemp - self.env_options.eta_hvac * action_k[1] / self.env_options.A)
        p_grid = current_netload + p_charge + p_discharge + action_k[1]
        is_valid = True
        # reward_original,c1_,c2_,c3_ = self.get_non_myopic_reward_function(p_grid, time_step,T_next,action_k)
        reward_original, c1_, c2_, c3_ = self.get_reward(p_grid, time_step, T_next, action_k)

        next_state = [self.get_solar(day_number, time_step + 1), self.get_load(day_number, time_step + 1), e_next,
                      self.get_outdoorTemp(day_number, time_step + 1), T_next,
                      self.get_price(day_number, time_step + 1), time_step + 1]

        return next_state, reward_original, c1_, c2_, c3_

    def get_reward(self, p_grid, time_step, T_next, action_k):
        if p_grid > 0:
            c1 = p_grid * self.get_price(self.day_number, time_step)
        else:
            c1 = 0.9 * p_grid * self.get_price(self.day_number, time_step)
        c2 = abs(action_k[0]) * self.env_options.DepriciationParam
        c3 = (max(0, self.env_options.T_min - T_next) + max(0, T_next - self.env_options.T_max))
        self.TempViolation.append(max(0, self.env_options.T_min - T_next) + max(0, T_next - self.env_options.T_max))
        return -((c1 + c2) * self.env_options.CostReImportance + c3), c1, c2, c3

    def get_non_myopic_reward_function(self, p_grid, time_step, T_next, action_k):
        current_price = self.get_price(self.day_number, time_step)
        if p_grid > 0:
            reward = current_price
            Real_Energy_Cost = reward * p_grid
            for price in [self.get_price(self.day_number, time) for time in range(time_step + 1, 24)]:
                reward += (price - current_price)
        else:
            reward = 0.9 * current_price
            Real_Energy_Cost = reward * p_grid
            for price in [self.get_price(self.day_number, time) for time in range(time_step + 1, 24)]:
                reward += 0.9 * (current_price - price)

        self.TempViolation.append(max(0, self.env_options.T_min - T_next) + max(0, T_next - self.env_options.T_max))
        c1 = (reward * p_grid + abs(action_k[0]) * self.env_options.DepriciationParam)
        c2 = abs(action_k[0]) * self.env_options.DepriciationParam
        c3 = (max(0, self.env_options.T_min - T_next) + max(0, T_next - self.env_options.T_max))
        return -(c1 * self.env_options.CostReImportance + c3), Real_Energy_Cost, c2, c3

    def get_price(self, day_number, time_step):
        if time_step > 23:
            day_number = day_number + 1
            time_step %= 24
        day_number = day_number % self.day_chunk
        time_step = self.get_key(time_step)
        return self.df_price[time_step][day_number]

    @staticmethod
    def get_key(time_step):
        time_step = str(time_step)
        return time_step + ':00'

    def get_solar(self, day_number, time_step):
        if time_step > 23:
            day_number = day_number + 1
            time_step %= 24
        day_number = day_number % self.day_chunk
        time_step = self.get_key(time_step)
        return self.df_solar[time_step][day_number]

    def get_load(self, day_number, time_step):
        if time_step > 23:
            day_number = day_number + 1
            time_step %= 24
        day_number = day_number % self.day_chunk
        time_step = self.get_key(time_step)
        return self.df_load[time_step][day_number]

    def get_outdoorTemp(self, day_number, time_step):
        if time_step > 23:
            day_number = day_number + 1
            time_step %= 24
        day_number = day_number % self.day_chunk
        time_step = self.get_key(time_step)
        return self.df_outTemp[time_step][day_number]


if __name__ == '__main__':
    '''
        code for testing the environment class
    '''
    environment = Environment()
    environment.reset()
