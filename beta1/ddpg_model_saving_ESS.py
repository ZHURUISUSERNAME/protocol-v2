import tensorflow as tf
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomUniform
import keras.backend as K
from keras.layers.merge import concatenate, Add
from keras.layers import Dense, Flatten, Input, Lambda, Activation
import numpy as np
#from car_env import CarEnv
import matplotlib.pyplot as plt
from keras.models import model_from_json
from ddpg_environment import Environment
from ddpg_options import get_default_object

from numpy.random import seed
import datetime
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
import xlwt
workbook = xlwt.Workbook(encoding='utf-8')
booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)    

MAX_EPISODES = 3000
MAX_EP_STEPS = 24  
GAMMA = 0.995
MEMORY_CAPACITY = 120*200
BATCH_SIZE = 120
Target_Tau=0.001
Number_of_Hidden_neurons_1=300
Number_of_Hidden_neurons_2=600
Critic_Learning_Rate=0.001
Actor_Learning_Rate=0.0001



#env = CarEnv(discrete_action=DISCRETE_ACTION)
env_options = get_default_object()
env = Environment(env_options)
STATE_DIM = env.env_options.state_size
ACTION_DIM = env.env_options.action_dim 



    
class Actor(object):
    def __init__(self, sess, env, env_options, state_dim, action_dim):
        self.sess = sess
        self.env = env
        self.env_options = env_options
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()
        self.actor_critic_grad = tf.placeholder(tf.float32,[None, self.a_dim]) # where we will feed de/dC (from critic)
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(Actor_Learning_Rate).apply_gradients(grads)

    def create_actor_model(self):
        print("Now we build the model")
        state_input= Input(shape=[self.s_dim])
        h1 = Dense(Number_of_Hidden_neurons_1, activation='relu')(state_input)
        h2 = Dense(Number_of_Hidden_neurons_2, activation='relu')(h1)
        ESS_operation = Dense(1, activation='tanh')(h2)
        HVAC_action = Dense(1, activation='sigmoid')(h2)
        output= concatenate([ESS_operation, HVAC_action])
        model = Model(input=state_input, output=output)
        return state_input, model
    


    def learn(self, s, critic_grads, critic_state_input, critic_action_input):
        predicted_actions = self.actor_model.predict(s)
        grads = self.sess.run(critic_grads, feed_dict={
            critic_state_input:  s,
            critic_action_input: predicted_actions
        })[0]
        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: s,
            self.actor_critic_grad: grads
        })
        self.target_actor_model.set_weights(np.multiply(Target_Tau,self.actor_model.get_weights())+np.multiply(1-Target_Tau,self.target_actor_model.get_weights()))

    def choose_action(self, s):
        actions = self.actor_model.predict(s)[0]
        return actions

class Critic(object):
    def __init__(self, sess, env, env_options,state_dim, action_dim, gamma, target_actor_model):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.gamma = gamma
        self.target_actor_model = target_actor_model
        self.t_replace_counter = 0
        self.critic_state_input, self.critic_action_input, \
            self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()
        self.critic_grads = tf.gradients(self.critic_model.output,
            self.critic_action_input) # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    def create_critic_model(self):
        print("Now we build the model")
        state_input= Input(shape=[self.s_dim])
        state_h1 = Dense(Number_of_Hidden_neurons_1, activation='relu')(state_input)
        state_h2 = Dense(Number_of_Hidden_neurons_2, activation='linear')(state_h1)
        action_input = Input(shape=[self.a_dim])
        action_h1= Dense(Number_of_Hidden_neurons_2, activation='linear')(action_input)
        merged= Add()([state_h2, action_h1])
        merged_h1 = Dense(Number_of_Hidden_neurons_2, activation='relu')(merged)
        V = Dense(1,activation='linear')(merged_h1)
        model  = Model(input=[state_input,action_input], output=V)
        adam  = Adam(lr=Critic_Learning_Rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model
    

    def learn(self, s, a, r, s_):
        target_actions = self.target_actor_model.predict(s_)
        future_rewards = self.target_critic_model.predict([s_, target_actions])
        r += self.gamma * future_rewards
        self.critic_model.train_on_batch([s, a], r)
        self.target_critic_model.set_weights(np.multiply(Target_Tau,self.critic_model.get_weights())+np.multiply(1-Target_Tau,self.target_critic_model.get_weights()))

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        s_modified=[item for sublist in s for item in sublist]
        s_modified_1 = [item for sublist in s_ for item in sublist]
        transition = np.hstack((s_modified, a, [r], s_modified_1))
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]



def OU_function(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)


M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

def train(xxx,i_index):
    sess = tf.Session()
    K.set_session(sess)
    actor = Actor(sess, env, env_options,STATE_DIM, ACTION_DIM,)
    critic = Critic(sess, env, env_options,STATE_DIM, ACTION_DIM, GAMMA, actor.target_actor_model)
    Total_reward=[]
    var =1
    LastMean=1000
    epsilon=1
    env.day_chunk=1
    M.pointer=0
    for ep in range(MAX_EPISODES):
        env.ChooseRandomParameter(0,60)
        env.env_options.DepriciationParam=xxx
        state = env.reset()
        ep_step = 0
        done=False
        for t in range(MAX_EP_STEPS):
            a = actor.choose_action(env.NormalizedPreprocess(np.array(state)))
            if np.random.random()<epsilon:
                a[0]=np.random.uniform(-1,1)
                a[1]=np.random.uniform(0,1)
            #a=np.clip(np.random.normal(a, var),[-1,0],[1,1])
            #a[0]=a[0]+max(epsilon, 0)*OU_function(a[0], 0.0 , 1.0, 0.20)
            #a[1]=a[1]+max(epsilon, 0)*OU_function(a[1], 0.5 , 1.0, 0.20)
            
            #a=np.clip(a,[-1,0],[1,1])
            a_normal=[a[0]*env.env_options.P_cap, a[1]*env.env_options.hvac_p_cap]
            state_,reward_original,c1_,c2_,c3_ = env.step(a_normal)
            M.store_transition(env.NormalizedPreprocess(np.array(state)), a, reward_original, env.NormalizedPreprocess(np.array(state_)))
            Total_reward.append(reward_original)
            '''
            if c1_<-100:
                print('a_normal:',a_normal,'indoorTemp:',state[4],'reward_original',reward_original,'(c1,c2,c3_)',c1_,c2_,c3_)
            '''
            if M.pointer > MEMORY_CAPACITY:
                #var = max([1-0.003*ep, VAR_MIN])
                #var = max([var*0.99995, VAR_MIN])
                epsilon=max(1-0.0005*(ep-(MEMORY_CAPACITY/MAX_EP_STEPS)),0.1)
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]
                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s, critic.critic_grads, critic.critic_state_input, critic.critic_action_input)
                
            state = state_
            ep_step += 1
            if t == MAX_EP_STEPS - 1:
                done=1
                if ep%10==0:
                    print('Ep:', ep,'| Steps: %i' % int(ep_step),'| Explore: %.2f' % epsilon,'| TotalRewards: %.5f' %np.array(Total_reward).mean(),'| IndoorTemperature: %.5f' %state[4],'|error: %.5f'%abs(np.array(Total_reward).mean()-LastMean))
                break

    model_json = actor.actor_model.to_json()
    with open('./actor_model'+'%d.json'%i_index, "w") as json_file:
        json_file.write(model_json)
    actor.actor_model.save_weights('./actor_model'+'%d.h5'%i_index)
    model_json = critic.critic_model.to_json()
    with open('./critic_model'+'%d.json'%i_index, "w") as json_file:
        json_file.write(model_json)
    critic.critic_model.save_weights('./critic_model'+'%d.h5'%i_index)
    np.savetxt('.\Total_reward_Convergence'+'%d.txt'%i_index, np.array(Total_reward))
    '''
    plt.plot(Total_reward)
    plt.show()
    '''


def test(xxx,i_index):
    
    Total_reward_Energy_1=[]
    Total_reward_Energy_2=[]
    Total_reward_Comfort=[]
    IndoorTemp_list_all=[]
    IndoorTemp_baseline_list_all=[]
    StateList=[]
    OutdoorTemp_list_all=[]
    ESS_list_all=[]
    Netload=[]
    Baseline_Temp_Violation=[]
    Price_list=[]
    Baseline_HVAC_input=[]
    Proposed_HVAC_input=[]
    HVAC_input=0
    InitalTemperature=0

    sess = tf.Session()
    K.set_session(sess)
    actor = Actor(sess, env, env_options,STATE_DIM, ACTION_DIM,)
    critic = Critic(sess, env, env_options,STATE_DIM, ACTION_DIM, GAMMA, actor.target_actor_model)

    json_file = open('./actor_model'+'%d.json'%i_index, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('./actor_model'+'%d.h5'%i_index)
    actor.actor_model=loaded_model
    actor.target_actor_model=loaded_model

    json_file = open('./critic_model'+'%d.json'%i_index, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('./critic_model'+'%d.h5'%i_index)
    critic.critic_model=loaded_model
    critic.target_critic_model=loaded_model

    
    print("Loaded model from disk")
    MAX_EPISODES=1
    env.day_chunk=31
    MAX_Test_STEPS=24*env.day_chunk
    
    for ep in range(MAX_EPISODES):
        env.ChooseRandomParameter(61,61)
        env.env_options.DepriciationParam=xxx
        s = env.reset()
        ep_step = 0
        for t in range(MAX_Test_STEPS):
            a = actor.choose_action(env.NormalizedPreprocess(np.array(s)))
            a_normal=[a[0]*env.env_options.P_cap, a[1]*env.env_options.hvac_p_cap]
            state_, reward_original,c1_,c2_,c3_ = env.step(a_normal)
            IndoorTemp_list_all.append((s[4]-32)/1.8)
            OutdoorTemp_list_all.append((s[3]-32)/1.8)
            Proposed_HVAC_input.append(a_normal[1])
            ESS_list_all.append(s[2])
            Price_list.append(s[5])
            Total_reward_Energy_1.append(c1_)
            Total_reward_Energy_2.append(c2_)
            Total_reward_Comfort.append(c3_)


            #baseline---On/Off policy
            if ep_step==0:
                if s[4]>env.env_options.T_max:
                    HVAC_input=env.env_options.hvac_p_cap
                elif s[4]<env.env_options.T_min:
                    HVAC_input=0
                Baseline_HVAC_input.append(HVAC_input)
                InitalTemperature=env.env_options.Ewuxilong*s[4]+(1-env.env_options.Ewuxilong)*(s[3]-env.env_options.eta_hvac*HVAC_input/env.env_options.A)
                IndoorTemp_baseline_list_all.append((s[4]-32)/1.8)
                Netload.append(s[1]-s[0]+HVAC_input)
                Baseline_Temp_Violation.append((max(0,env.env_options.T_min-InitalTemperature)+max(0,InitalTemperature-env.env_options.T_max)))
            else:
                if InitalTemperature>env.env_options.T_max:
                    HVAC_input=env.env_options.hvac_p_cap
                elif InitalTemperature<env.env_options.T_min:
                    HVAC_input=0
                Baseline_HVAC_input.append(HVAC_input)
                InitalTemperature=env.env_options.Ewuxilong*InitalTemperature+(1-env.env_options.Ewuxilong)*(s[3]-env.env_options.eta_hvac*HVAC_input/env.env_options.A)
                IndoorTemp_baseline_list_all.append((InitalTemperature-32)/1.8)
                Netload.append(s[1]-s[0]+HVAC_input)
                Baseline_Temp_Violation.append((max(0,env.env_options.T_min-InitalTemperature)+max(0,InitalTemperature-env.env_options.T_max)))

            s = state_
            ep_step += 1
            if t == MAX_Test_STEPS - 1:
                '''
                print('Ep:', ep,
                      '| Steps: %i' % int(ep_step),
                      '| Explore: %.2f' % var,
                      )
                '''
                break
            
    base_energy_cost = sum([a * b for a, b in zip(Price_list, Netload)])
    
    #np.savetxt('.\data_Netload.txt', np.array(Netload))
    np.savetxt('.\EnergyCost_'+'%d.txt'%i_index,np.array(Total_reward_Energy_1))
    np.savetxt('.\BatteryCost_'+'%d.txt'%i_index,np.array(Total_reward_Energy_2))
    np.savetxt('.\DisComfortCost_'+'%d.txt'%i_index,np.array(Total_reward_Comfort))
    print('******DepriciationParam:',xxx,'*********Index*********:',i_index,'*******CostReImportance*********:',env.env_options.CostReImportance)
    print('Pro: Total energy cost:',np.array(Total_reward_Energy_1+Total_reward_Energy_2).sum(),'(c1,c2)',np.array(Total_reward_Energy_1).sum(),np.array(Total_reward_Energy_2).sum())
    print('Pro: Total discomfort cost:',np.array(Total_reward_Comfort).sum())
    print('Baseline: Total Energy cost:',base_energy_cost)
    print('Baseline: Total discomfort cost:',np.array(Baseline_Temp_Violation).sum())
    booksheet.write(i_index + 1, 1, np.array(Total_reward_Energy_1 + Total_reward_Energy_2).sum())
    booksheet.write(i_index + 1, 2, np.array(Total_reward_Energy_1).sum())
    booksheet.write(i_index + 1, 3, np.array(Total_reward_Energy_2).sum())
    booksheet.write(i_index + 1, 4, float(np.array(Total_reward_Comfort).sum()))
    booksheet.write(i_index + 1, 5, base_energy_cost)
    booksheet.write(i_index + 1, 6, np.array(Baseline_Temp_Violation).sum())
    workbook.save(nowTime+'.xls')
    #np.savetxt('E:\data_Comfort.txt', np.array(Total_reward_Comfort))
    #plt.plot(Total_reward)

    '''
    print('The lowest indoor temperature:',np.array(OutdoorTemp_list_all).min())
    plt.plot(OutdoorTemp_list_all,label='Outdoor temperature')
    plt.plot(IndoorTemp_list_all,label='Indoor temperature')
    plt.plot(IndoorTemp_baseline_list_all,label='Indoor temperature (On-off)')
    plt.plot([24]*744,label='Upper limit')
    plt.plot([19]*744,label='Lower limit')
    plt.xlabel('Time slots')
    plt.ylabel('Temperature (oC)')
    plt.legend(loc=1)
    plt.show()
    '''
    
    
    np.savetxt('.\OutdoorTemp_list_all.txt',np.array(OutdoorTemp_list_all))
    np.savetxt('.\IndoorTemp_list_all.txt',np.array(IndoorTemp_list_all))
    np.savetxt('.\IndoorTemp_baseline_list_all.txt',np.array(IndoorTemp_baseline_list_all))
    

 

    '''
    plt.plot(ESS_list_all,label='Stored energy in ESS')
    plt.plot([6]*720,label='Upper limit')
    plt.plot([0.6]*720,label='Lower limit')
    plt.plot(np.multiply(Price_list,20),label='Scaled electricity price')
    plt.xlabel('Time slots')
    plt.ylabel('Energy (kWh)')
    plt.legend(loc=1)
    plt.show()
    '''
    
    '''
    x = np.arange(0., 720, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, Price_list,'r')
    ax1.legend(loc=1)
    ax1.set_ylabel('Price ($/kWh)')
    ax2 = ax1.twinx() # this is the important function
    ax2.plot(x, ESS_list_all, 'go-')
    ax2.legend(loc=2)
    ax2.set_ylabel('ESS energy level (kWh)')
    ax2.set_xlabel('Time slots')
    plt.show()
    '''

    '''
    x = np.arange(0., 720, 1)
    fig,left_axis=plt.subplots()
    right_axis = left_axis.twinx()
    p1, = left_axis.plot(x, Price_list, 'b.-')
    p2, = right_axis.plot(x, ESS_list_all, 'r.-')

    left_axis.set_xlim(200,280)
    left_axis.set_xticks(np.arange(200, 280, 10))
    left_axis.set_ylim(0.1,0.36)
    left_axis.set_yticks(np.arange(0.1,0.36,0.05))
    right_axis.set_ylim(0,6.2)
    right_axis.set_yticks(np.arange(0,6.2,1))

    left_axis.set_xlabel('Time slots')
    left_axis.set_ylabel('Price ($/kWh)')
    right_axis.set_ylabel('ESS energy level (kWh)')
    left_axis.yaxis.label.set_color(p1.get_color())
    right_axis.yaxis.label.set_color(p2.get_color())
    
    plt.show()
    '''


    '''
    x = np.arange(0., 744, 1)
    fig,left_axis=plt.subplots()
    right_axis = left_axis.twinx()
    p1, = left_axis.plot(x, Price_list, 'b.-')
    p2, = right_axis.plot(x, Proposed_HVAC_input, 'r.-')

    left_axis.set_xlim(200,280)
    left_axis.set_xticks(np.arange(200, 280, 10))
    left_axis.set_ylim(0.1,0.36)
    left_axis.set_yticks(np.arange(0.1,0.36,0.05))
    right_axis.set_ylim(0,2.1)
    right_axis.set_yticks(np.arange(0,2.1,0.5))

    left_axis.set_xlabel('Time slots')
    left_axis.set_ylabel('Price ($/kWh)')
    right_axis.set_ylabel('HVAC power input (kW)')
    left_axis.yaxis.label.set_color(p1.get_color())
    right_axis.yaxis.label.set_color(p2.get_color())
    
    plt.show()
    '''



if __name__ == '__main__':
    #AAA=[0.2]
    #BBB=[3]
    #AAA=[0.001]
    AAA=[0.001]
    BBB=2
    for iter in range(len(AAA)):
        for loop in range(BBB):
            train(AAA[iter],iter*BBB+loop+1)
            test(AAA[iter],iter*BBB+loop+1)
	
