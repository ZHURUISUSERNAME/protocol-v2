import numpy as np
import itertools


class EnvironmentOptions:

    def __init__(self, ComfortPenalty,eta, gamma, start, day_chunk, total_years, price_data, look_ahead, E_cap, P_cap, hvac_p_cap,E_init, T_min, T_max, TemperPenalty,Ewuxilong,eta_hvac,A,IndoorTemp_init,
                 epsilon, action_dim,solar_data, load_data, temperature_data,learning_rate,DepriciationParameter,CostReImportance,batch_size,state_size):
        '''
            options class from where environment get's all it's parameters.
            can be custom created or a default veriosn can also be loaded.
            added to provide flexibility as well as make the process of creating
            the environment uniform, like other openai-gym environments
		'''
        self.eta = eta  # battery efficiency
        self.gamma = gamma  # discount factor, importance given to future Q value predicted by the model
        self.start = start  # starting index in training files
        self.day_chunk = day_chunk  # no of days(elements) to consider in the data files
        self.total_years = total_years  # how many times do we want to repeat over the data
        self.E_cap = E_cap  # battery energy capacity (confirm)
        self.P_cap = P_cap  # battery power capacity (confirm)
        self.hvac_p_cap=hvac_p_cap
        self.E_init = E_init  # initial energy of the battery
        self.epsilon = epsilon  # what is this ?
        self.look_ahead = look_ahead  # how many steps in time when we want to look ahead
        self.solar_data = solar_data  # solar data path file
        self.load_data = load_data  # load data path file
        self.temperature_data=temperature_data # temperature data path file
        self.price_data=price_data
        self.T_min=T_min
        self.T_max=T_max
        self.TemperPenalty=TemperPenalty
        self.ComfortPenalty=ComfortPenalty
        self.CostReImportance=CostReImportance
        self.Ewuxilong=Ewuxilong
        self.eta_hvac=eta_hvac
        self.A=A
        self.IndoorTemp_init=IndoorTemp_init
        self.E_max = E_cap  # maximum energy cap of the battery
        self.E_min = (1 - 0.9) * self.E_max  # minimum energy cap of the battery
        self.E_init = 0.2 * self.E_max  # initial starting capacity of the battery
        self.learning_rate = learning_rate  # importance to new and old Q values
        self.DepriciationParam=DepriciationParameter
        self.batch_size=batch_size
        self.state_size=state_size
        self.action_dim=action_dim

    def set_default_object(self,day_chunk1):
        self.day_chunk=day_chunk1
        
        


def get_default_object():
    '''
        get a default object from here with default variable values
        some examples of price schemes

        #price = [.040,.040,.040,.040,.040,.040,.080,.080,.080,.080,.040,.040,.080,.080,.080,.040,.040,.120,.120,.040,.040,.040,.040,.040]
        #price = [.040,.040,.080,.080,.120,.240,.120,.040,.040,.040,.040,.080,.120,.080,.120,.040,.040,.120,.120,.040,.040,.040,.040,.040]
        #price = [.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.080, .080,.120,.120,.040,.040,.040]
	'''
    gamma = 0.3
    eta = 0.95
    day_chunk =1
    total_years = 2000
    e_cap = 6.0
    p_battery_cap = 3
    hvac_p_cap = 2
    DepriciationParameter=0.001
    e_init = 0.3 * e_cap
    epsilon = 1.0
    T_min=19*1.8+32
    T_max=24*1.8+32
    TemperPenalty=0.1
    ComfortPenalty=0
    CostReImportance=1
    Ewuxilong=0.7
    eta_hvac=2.5
    A=0.14
    IndoorTemp_init=23*1.8+32
    #action_charging_discharging = np.arange(- p_battery_cap, p_battery_cap+0.0001, 0.001).tolist()
    #action_hvac = np.arange(0, hvac_p_cap+0.01, 0.5).tolist()
    #Action=[x for x in itertools.product(action_charging_discharging,action_hvac)]
    action_dim=2
    look_ahead = 1
    solar_data = './Data/solar_double'
    load_data = './Data/base_load_modified'
    temperature_data = './Data/tempe_modified'
    price_data='./Data/price_modified'
    
    start = 1
    #lowerPrice=0.3583
    #HighPrice=0.5583
    #price_scheme= [lowerPrice, lowerPrice,lowerPrice, lowerPrice,lowerPrice, lowerPrice,lowerPrice, lowerPrice,HighPrice,HighPrice,HighPrice,HighPrice,HighPrice,HighPrice,HighPrice,HighPrice,
     #                HighPrice,HighPrice,HighPrice,HighPrice,HighPrice,lowerPrice,lowerPrice,lowerPrice]

    learning_rate = 0.1
    batch_size=96
    state_size=7
    return EnvironmentOptions(ComfortPenalty,eta, gamma, start, day_chunk, total_years, price_data, look_ahead, e_cap, p_battery_cap,hvac_p_cap, e_init, T_min, T_max, TemperPenalty,Ewuxilong,eta_hvac,A,IndoorTemp_init,
                              epsilon, action_dim,solar_data, load_data, temperature_data, learning_rate,DepriciationParameter,CostReImportance,batch_size,state_size)

          
    
