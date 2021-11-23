import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import copy


# Loading data here
p = open('data.pickle', 'rb')
data = np.array(pickle.load(p))
p.close()

''' Keeping measurements in the imaging ragion only '''
p2 = open('data.pickle', 'rb')
data_temp = np.array(pickle.load(p2))
p2.close()

count = 0
for i in range(data_temp.shape[0]):
    for name in data_temp[i]['radar']:
        ''' Checking if the measurement is in given imaging region '''
        meas = data_temp[i]['radar'][name]
        if meas[0] < 0 or meas[0] > 15 or meas[1] > 30 or meas[1] < 0:
            ''' Try except useful when ran multiple times '''
            try:
                data[i]['radar'].pop(name)
                count +=1
            except KeyError:
                continue
print('\n' + 'Removed measurements outside imaging region : ' + str(count) )

''' Uncomment to visualize sensor data for first 50 seconds '''
# landmarks = np.zeros((5000, 6 ,2))
# for i in range(5000):
#     idx = 0
#     for a in data[i]['radar']: 
#         landmarks[i, idx, :] = data[i]['radar'][a]
#         idx += 1

# for i in range(6):    
#     plt.scatter(landmarks[:,i,0]-7.5, landmarks[:,i,1])
# plt.scatter(0, 0, color='white')
# plt.xlim([-8, 8])
    
''' Creating Particle class '''
class particle():
    
    '''Initializing particle with the given state'''
    def __init__(self, state):
        self.state            = state # Particle state : [x, y, heading] global coordinates
        self.landmarks_state  = {} # Dictionary : Landmark name and global position
        self.landmarks_var    = {} # Dictionary : Landmark name and variance
        
    ''' Predict new state from odometer data '''
    def move(self, dist, angle):   # This is of predict step
        self.state[2] += angle  # radian
        self.state[0] += dist*np.cos(self.state[2]) # X meters
        self.state[1] += dist*np.sin(self.state[2]) # Y meters
                
    def landmarkCount(self):
        return len(self.landmarks_state)
   
    def polar(self, measurement): # converts cartesian to polar
        ''' Sensor coordinate system's origin is shifted to the position of the sensor '''
        distance = np.sqrt( (measurement[0] - 7.5)**2 + (measurement[1])**2 ) 
        bearing = np.arctan2(-(measurement[0] - 7.5), (measurement[1])) * (180/np.pi)
        return np.array([distance, bearing])
    
    def sensor_to_global(self, state, pt, L):      
        c = np.cos(-0.5*np.pi + state[2])
        s = np.sin(-0.5*np.pi + state[2])
        x, y = pt
        m    = np.array([[0], [(x-7.5) * c - (y+L) * s + state[0]], [(x-7.5) * s + (y+L) * c + state[1]]])
        return np.array([m[1], m[2]]) 
    
    def get_h(self, state, landmark, L):
        r_state  = [L*state[0]*np.cos(state[2]), L*state[1]*np.cos(state[2])] # Global sensor position
        distance = np.sqrt( (landmark[0]-r_state[0])**2 + (landmark[1]-r_state[1])**2) # Measurement range (m)
        bearing  = np.arctan2( (landmark[1]-r_state[1]), (landmark[0]-r_state[0])) - state[2] # Measurement bearing (rad)
        return  [distance[0], bearing[0]*180/np.pi] # [Meters, degrees]
    
    def get_jacobian_H(self, state, landmark, L):
        del_x = landmark[0] - state[0] + L * np.cos(state[2]) 
        del_y = landmark[1] - state[1] + L * np.sin(state[2]) 
        r     = (del_x)**2 + (del_y)**2

        H = np.zeros((2,2), dtype=float)
        H[0,0] = del_x/np.sqrt(r)
        H[0,1] = del_y/np.sqrt(r)
        H[1,0] = -del_y/r
        H[1,1] = del_x/r
        
        return H
    
    def update_particle(self, name, measurement, m_cov, L):
        ''' I/p argument : Landmark name, it's measurement and measurement covariance
        Check if landmark exists in particle's map or else create new landmark '''
        try:             
            old_state = self.landmarks_state[name]  # In Global coordinate system
            old_var   = self.landmarks_var[name]
            
            ''' Calculating expected measurement h '''
            h = self.get_h(self.state, old_state, L)   # Expected measurement in polar
            H = self.get_jacobian_H(self.state, old_state, L) #2x2
            
            Q = np.dot(H, np.dot(old_var,H.transpose())) + m_cov #2x2
            K = np.dot(old_var, np.dot(H.transpose(), np.linalg.inv(Q))) #2x2
            
            # meas_p is conversion from cartesian sensor measurement to polar
            meas_p    = self.polar(measurement)            
            del_z     = np.array([meas_p - h])
            
            new_state = old_state + np.dot(K, del_z.T)
            new_var   = np.dot((np.eye(2) - np.dot(K,H)), old_var)
            
            self.landmarks_state[name] = new_state
            self.landmarks_var[name]   = new_var
            
            Ql = np.sqrt(np.linalg.det(Q))
            weight = 1/(2*np.pi*Ql) * \
                     np.exp ( (-1/2) * (np.dot(del_z, np.dot(np.linalg.inv(Q), del_z.T) ) ) )
            
        except KeyError:
            
            ''' Initialize new landmark state '''
            landmark_state = self.sensor_to_global(self.state, measurement, L) # Sensor coordinates to global coordinates
            
            # Compute landmark covariance
            H        = self.get_jacobian_H(self.state, landmark_state, L)
            H_inv    = np.linalg.inv(H)
        
            variance = np.dot(H_inv, np.dot(m_cov, H_inv.T))
            
            self.landmarks_state[name] = landmark_state
            self.landmarks_var[name]   = variance
            weight = 1.0
        
        return weight
        

'''Creating SLAM class for convenience and calling routines '''
class slam():
    def __init__(self, meas_cov, control_cov, particles, L):
        self.particles = particles
        self.m_cov = meas_cov
        self.c_cov = control_cov
        self.L = L
    
    def localized_robot(self):
        self.mean = np.zeros(3)
        for p in self.particles:
            self.mean += p.state
        return self.mean/len(self.particles)
            
    def predict(self, dist, angle):
        for p in self.particles:
            p.move(dist, angle)
    
    # def predict(self, vel, ang, dt):
    #     for p in self.particles:
    #         dist = random.gauss(vel, self.c_cov[0,0]) * dt
    #         angle  = random.gauss(ang, self.c_cov[1,1]) * dt
    #         p.move(dist, angle)

    def update_weights(self, measurements):
        weights = []
        for p in self.particles:
            w = 1.0
            for name in measurements:
                ''' I/p argument : Landmark name ,it's measurement & measurement covariance '''
                w *= p.update_particle(name, measurements[name], self.m_cov, self.L) 
            weights.append(w)
        return weights
    
    def resample(self, weights):
        new_particles = []
        max_weight    = max(weights)
        index         = random.randint(0, len(self.particles)-1)
        beta          = 0.0

        for i in range(len(self.particles)):
            beta += random.random() * 2.0 * max_weight
            while beta > weights[index]:
                beta   -= weights[index]
                index   = (index+1)%len(self.particles)
            new_particles.append(copy.deepcopy(self.particles[index]))
            
        return new_particles
    
    def get_best_particle(self, mean_state):
        min_index = 0   #Index of particle with minimum distance from mean
        min_dist  = 200 # Setting minimum distance big initially 
        for i in range(len(self.particles)):
            x_dist = (mean_state[0] - self.particles[i].state[0])
            y_dist = (mean_state[1] - self.particles[i].state[1])
            dist   = np.sqrt(x_dist**2 + y_dist**2)
            if dist < min_dist:
                min_index = i
                min_dist  = dist
                
        return self.particles[min_index], min_dist
    


# ------------------------------------- MAIN --------------------------------------------#


''' Covariance matrices '''
measurement_covariance = np.array([[0.0099589,-0.0002977],[-0.0002977, 5.2257288]])
control_covariance     = np.array([[0.08620446,-0.00025468],[-0.00025468,0.00251587]])
prev_t   = 0.0 # previous timestamp
L        = 0.3 # Sensor offset in meters

num_particles = 25 # Initial number of particles
all_particles = []
lands         = []
states        = np.zeros((data.shape[0],3), dtype=float)
landmarks     = np.zeros((data.shape[0],2), dtype=float)

''' Creating particles and adding to the list '''
for i in range(num_particles):
    ''' Initializing particles with random heading direction (1 standard deviation) '''
    all_particles.append(particle([0,0,random.gauss(0.0, np.sqrt(control_covariance[1,1]))]))    
    
    ''' Initializing particles with non-random state'''
    # all_particles.append(particle([0.0,0.0,0.0]))

''' Creating new SLAM object '''
s = slam(measurement_covariance, control_covariance, all_particles, L)

''' Iterating for all timesteps '''
N = data.shape[0]
# N= 10000

for i in range(N):
    # print(i)
    dt       = data[i]['timestamp'] - prev_t  # Time interval
    prev_t   = data[i]['timestamp']
    distance = data[i]['odometry'] * dt     # Distance travelled during the interval
    angle    = data[i]['gyroscope'] * dt    # Change in heading angle during the interval
    
    # Predict particle's new state
    s.predict(distance, angle)
    # s.predict(data[i]['odometry'], data[i]['gyroscope'], dt)
    
    # Update weight of all particles
    weights = s.update_weights(data[i]['radar']) # i/p argument : current measurements
    
    # Resample particles according to new weights
    s.particles = s.resample(weights)
    
    # Extract robot's current location
    states[i,:] = s.localized_robot()
    
    # Get the best particle according to current localized robot
    best, min_dist = s.get_best_particle(states[i,:])
    
    # Landmarks and their states    
    lands.append(best.landmarks_state)
    

''' Plot Robot's path '''
plt.plot(states[:,0], states[:,1]) 
# plt.scatter(states[:,0], states[:,1]) 
plt.scatter(0,0, marker='*', linewidths=5.0, color='black')  
plt.text(-2, 1, 'START')
plt.text(states[-1,0], states[-1,1]-2, 'END')
plt.scatter(states[-1,0], states[-1,1], marker='*', linewidths=5.0, color='black')  

'''Plot landmarks from best particle'''
temp = lands[N-1]
for a in temp:
    plt.scatter(temp[a][0], temp[a][1])
    plt.text(temp[a][0]-1, temp[a][1]+1, a)
    
# Creating pickle files
pos = {}
for i in range(data.shape[0]):
    pos[data[i]['timestamp']] = [states[i,0],states[i,1]]
a = open('trajectory.pickle','wb')
pickle.dump(pos, a)
a.close()

lands_pickle = {}
for a in temp:
    lands_pickle[a] = temp[a]
a = open('targets.pickle','wb')
pickle.dump(lands_pickle, a)
a.close()
    
print('\n DONE!!')



















