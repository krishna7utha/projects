#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static default_random_engine r;


// Particle filter initialization.
// Set number of particles and initialize them to first position based on GPS estimate.
void ParticleFilter::init(double x, double y, double theta, double std[]) {

    num_particles = 100;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {
        Particle a;
        a.x		= dist_x(r);
        a.y		= dist_y(r);
        a.theta	= dist_theta(r);
        a.id	= i;
        a.weight= 1.0;
        particles.push_back(a);
    }
    is_initialized = true;
}


// Move each particle according to bicycle motion model (taking noise into account)
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    for (int i = 0; i < num_particles; ++i) {
        double theta_p, x_p, y_p;
        if (abs(yaw_rate) > 1e-5) {
            x_p	= particles[i].x + velocity / yaw_rate * (sin(theta_p) - sin(particles[i].theta));
            y_p	= particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(theta_p));
            theta_p = particles[i].theta + yaw_rate * delta_t;
        } else {
            x_p   = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            y_p   = particles[i].y + velocity * delta_t * sin(particles[i].theta);
            theta_p = particles[i].theta;
        }

        normal_distribution<double> dist_x(x_p, std_pos[0]);
        normal_distribution<double> dist_y(y_p, std_pos[1]);
        normal_distribution<double> dist_theta(theta_p, std_pos[2]);
        particles[i].x	   = dist_x(r);
        particles[i].y	   = dist_y(r);
        particles[i].theta = dist_theta(r);
    }
}


// Finds which observations correspond to which landmarks by using a nearest-neighbor data association
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

    for (auto& obs : observations) {
        double distMin = numeric_limits<double>::max();

        for (auto& pred : predicted) {
            double dst = dist(obs.x, obs.y, pred.x, pred.y);
            if (dst<distMin) {obs.id = pred.id; distMin = dst;} 
        }
    }
  
}


// Update the weight of each particle taking into account current measurements.
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                     const std::vector<LandmarkObs> &observations,
                     const Map &map_landmarks) {

    double std_x = std_landmark[0];
    double std_y = std_landmark[1];

    for (int i = 0; i < num_particles; ++i) {
        vector<LandmarkObs> landmarks_pred;
        for (auto& map_landmark : map_landmarks.landmark_list) {
            double dst = dist(particles[i].x, particles[i].y, map_landmark.x_f, map_landmark.y_f);
            if (dst<sensor_range) {
                LandmarkObs l_pred;
                l_pred.x = map_landmark.x_f;
                l_pred.y = map_landmark.y_f;
                l_pred.id = map_landmark.id_i;
                landmarks_pred.push_back(l_pred);
            }
        }

        vector<LandmarkObs> obs_ref;
        for (int j = 0; j < observations.size(); ++j) {
            LandmarkObs rt_obs;
            rt_obs.x = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
            rt_obs.y = particles[i].y + sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y;
            obs_ref.push_back(rt_obs);
        }

        dataAssociation(landmarks_pred, obs_ref);
        double particle_weight = 1.0;
        double mu_x, mu_y;
      
        for (auto& obs : obs_ref) {
            for (auto& land: landmarks_pred)
                if (obs.id == land.id) {
                    mu_x = land.x;
                    mu_y = land.y;
                    break;}
            particle_weight *= exp(-(pow(obs.x - mu_x, 2) / (2 * std_x * std_x) + pow(obs.y - mu_y, 2) / (2 * std_y * std_y))) / 2 * M_PI * std_x * std_y;
        }

        particles[i].weight = particle_weight;

    }

    double norm = 0.0;
    for (auto& particle : particles){
        norm += particle.weight;
    }

    for (auto& particle : particles){
        particle.weight /= (norm + numeric_limits<double>::epsilon());
    }
}

// Resample particles with replacement with probability proportional to their weight.
void ParticleFilter::resample() {

    vector<double> particle_weights;
    for (auto& particle : particles){
        particle_weights.push_back(particle.weight);
    }
  
    discrete_distribution<int> distribution(particle_weights.begin(), particle_weights.end());
    vector<Particle> resample;
  
    for (int i = 0; i < num_particles; ++i) {
        resample.push_back(particles[distribution(r)]);
    }

    particles = resample;
    for (auto& particle : particles){
        particle.weight = 1.0;
    }
  
}


void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}