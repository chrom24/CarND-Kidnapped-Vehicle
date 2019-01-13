/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

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

//Random engine for various functions
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
	num_particles = 100;

	//Define normal distributions for sensor noise
	normal_distribution<double> x_noise_init(0, std[0]);
	normal_distribution<double> y_noise_init(0, std[1]);
	normal_distribution<double> theta_noise_init(0, std[2]);
                                                   
	for(int i=0; i<num_particles; i++){
    	//Init particles
		Particle p;
		p.id=i;
		p.x=x;
		p.y=y;
		p.theta=theta;
		p.weight = 1.0;
      
		//Add noise to particles
		p.x+=x_noise_init(gen);
		p.y+=y_noise_init(gen);
		p.theta+=theta_noise_init(gen);
      
		particles.push_back(p);
    }
  is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//Define normal distributions for sensor noise
	normal_distribution<double> x_noise(0, std_pos[0]);
	normal_distribution<double> y_noise(0, std_pos[1]);
	normal_distribution<double> theta_noise(0, std_pos[2]);
                                                       
	for (int i=0; i<num_particles;i++){
		//Add measurements
		if(abs(yaw_rate)!=0){
			particles[i].x=particles[i].x+velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
			particles[i].y=particles[i].y+velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
			particles[i].theta=particles[i].theta+yaw_rate*delta_t;
        }
		else{
			particles[i].x=particles[i].x+velocity * delta_t * cos(particles[i].theta);
			particles[i].y=particles[i].y+velocity * delta_t * sin(particles[i].theta);
        }
   
		//Add noise
		particles[i].x+=x_noise(gen);
		particles[i].y+=y_noise(gen);
		particles[i].theta+=theta_noise(gen);
      
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++){
    
	LandmarkObs o = observations[i];

	double min_dist = numeric_limits<double>::max();
	int map_id = -1;
    
	for (int j = 0; j < predicted.size(); j++){
		LandmarkObs p = predicted[j];
      
		// get distance between current/predicted landmarks
		double cur_dist = dist(o.x, o.y, p.x, p.y);

		// find nearest landmark
		if (cur_dist < min_dist){
			min_dist = cur_dist;
			map_id = p.id;
			}
		}
	observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
	for (int i = 0; i < num_particles; i++) {

		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		vector<LandmarkObs> predictions;

		for (int j = 0; j < map_landmarks.landmark_list.size(); j++){

			// Find landmarks in sensor range of particles
			float lm_x = map_landmarks.landmark_list[j].x_f;
			float lm_y = map_landmarks.landmark_list[j].y_f;
			int lm_id = map_landmarks.landmark_list[j].id_i;
			
			double sensor_range_2 = sensor_range * sensor_range;
			double dX = p_x - lm_x;
			double dY = p_y - lm_y;
			if (dX*dX + dY*dY <= sensor_range_2){
				predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
				}
			}


		// Transform observation coordinates
		vector<LandmarkObs> transformed_os;
		for (int j = 0; j < observations.size(); j++){
			double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
			double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
			transformed_os.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
			}

		// Observation association to landmark
		dataAssociation(predictions, transformed_os);

		// reinit weight
		particles[i].weight = 1.0;
		
		// Calculate weights
		for (unsigned int j = 0; j < transformed_os.size(); j++) {
      
		double o_x, o_y, pr_x, pr_y;
		o_x = transformed_os[j].x;
		o_y = transformed_os[j].y;

		int associated_prediction = transformed_os[j].id;

		for (int k = 0; k < predictions.size(); k++){
			if (predictions[k].id == associated_prediction){
				pr_x = predictions[k].x;
				pr_y = predictions[k].y;
				}
			}

		//Calculate weight with multivariate Gaussian
		double s_x = std_landmark[0];
		double s_y = std_landmark[1];
		double obs_w = (1/(2*M_PI*s_x*s_y)) * exp(-( pow(pr_x-o_x,2)/(2*pow(s_x, 2)) + (pow(pr_y-o_y,2)/(2*pow(s_y, 2)))));

		particles[i].weight *= obs_w;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Get weights and max weight.
	vector<double> weights;
	double max_weight = numeric_limits<double>::min();
	for(int i = 0; i < num_particles; i++){
		weights.push_back(particles[i].weight);
		if ( particles[i].weight > max_weight ){
			max_weight = particles[i].weight;
			}
		}

	// Creating distributions.
	uniform_real_distribution<double> dist_double(0.0, max_weight);
	uniform_int_distribution<int> dist_int(0, num_particles - 1);

	// Generating index.
	int index = dist_int(gen);

	double beta = 0.0;

	// the wheel
	vector<Particle> resampled_particles;
	for(int i = 0; i < num_particles; i++){
		beta += dist_double(gen) * 2.0;
		while( beta > weights[index]){
			beta -= weights[index];
			index = (index + 1) % num_particles;
			}
		resampled_particles.push_back(particles[index]);
		}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
