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

static default_random_engine generator;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	// Determined by trial and error
	num_particles = 120;

	// Normal distribution, as suggested by the lesson
	normal_distribution <double> x_init(0, std[0]);
	normal_distribution <double> y_init(0, std[1]);
	normal_distribution <double> theta_init(0, std[2]);

	// Initialize! A straightforward loop...
	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = x;
		particle.y = y;
		particle.theta = theta;
		particle.weight = 1.0;

		// noise
		particle.x += x_init(generator);
		particle.y += y_init(generator);
		particle.theta += theta_init(generator);

		particles.push_back(particle);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// We need distributions for sensor noise
	normal_distribution<double> x_noise(0, std_pos[0]);
	normal_distribution<double> y_noise(0, std_pos[1]);
	normal_distribution<double> theta_noise(0, std_pos[2]);
	
	// You can really look at the Unscented Kalman Filter prediction step and grab a couple of ideas!
	// The particle + yaw_rate code block is essentially the same, just factoring in theta when yaw_rate 
	// is >= 0.001

	for (int i = 0; i < particles.size(); i++) {
		if (fabs(yaw_rate) < 0.001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		particles[i].x += x_noise(generator);
		particles[i].y += y_noise(generator);
		particles[i].theta += theta_noise(generator);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	// Definitely useful as the notes above indicate. My workflow here is to 
	// run through each observation, subsequently running through each prediction
	// and distance in order to obtain the closes observed landmark.

	for (unsigned int i = 0; i < observations.size(); i++) {

		// Upon each iteration, I want to set a default "non-value" for the predicted map ID, 
		// which will be rewritten upon each prediction update below. It will then be reinitialized
		// on each observation iteration.
		int predicted_map_id = -10;

		// My initial approach was to set a very high number. Turns out it's 
		// better to initialize with the maximum value the data type can handle. 
		// Easier to handle & more elegant.
		float minimum_distance = numeric_limits<float>::max();
		
		for (unsigned int j = 0; j < predicted.size(); j++) {
		  
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (distance < minimum_distance) {
				minimum_distance = distance;
				predicted_map_id = predicted[j].id;
			}
		}
	
		// set the observation's id to the nearest predicted landmark's id
		observations[i].id = predicted_map_id;
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

	// The workflow here is to run through each particle, then map through each landmark obtaining its coordinates,
	// only considering those in the provided sensor_range function parameter. Said observation should be added
	// to the prediction vector initialized for this particle.
	// After this, we need to transform coordinates to appropriate map coordinates, so we can then iterate
	// and reassign weights for that particle.

	for (int i = 0; i < particles.size(); i++) {

		vector<LandmarkObs> predictions;
		vector<LandmarkObs> transformed_observations;

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {			
			if (fabs(map_landmarks.landmark_list[j].x_f - particles[i].x) <= sensor_range && fabs(map_landmarks.landmark_list[j].y_f - particles[i].y) <= sensor_range) {
				predictions.push_back(LandmarkObs{ map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f });
			}
		}

		for (unsigned int j = 0; j < observations.size(); j++) {
			double transformed_x = cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y + particles[i].x;
			double transformed_y = sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y + particles[i].y;
			transformed_observations.push_back(LandmarkObs{ observations[j].id, transformed_x, transformed_y });
		}

		dataAssociation(predictions, transformed_observations);

		// At this point, we should have predictions in the range of the particle, 
		// transformed and stored in the transformation_observations vector.
		// Time to recalculate weight.
		
		for (unsigned int j = 0; j < transformed_observations.size(); j++) {
			// Loop through all predictions to find the x and y of the observed.
			double prediction_x, prediction_y;

			for (unsigned int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == transformed_observations[j].id) {
					prediction_x = predictions[k].x;
					prediction_y = predictions[k].y;
				}
			}

			// New weight from multivariate Gaussian - this looks better on paper.
			particles[i].weight = (1/(2 * M_PI * std_landmark[0] * std_landmark[1])) * exp(-(pow(prediction_x - transformed_observations[j].x, 2)/(2 * pow(std_landmark[0], 2)) + (pow(prediction_y - transformed_observations[j].y, 2)/(2 * pow(std_landmark[1], 2)))));
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Vectors for particle and weight storage + max weight initialization
	vector<Particle> new_particles;
	vector<double> weights;
	double maximum_weight = 0;
	double beta = 0.0;

	// Pull all weights
	for (int i = 0; i < particles.size(); i++) {
		weights.push_back(particles[i].weight);
	}

	// Time to get the maximium weight
	for (std::vector<Particle>::const_iterator particle = particles.begin(); particle != particles.end(); ++particle)
	{
		if (particle->weight > maximum_weight) {
			maximum_weight = particle->weight;
		}
	}

	// Distribution initialization
	uniform_int_distribution<int> int_distribution(0, particles.size() - 1);
	uniform_real_distribution<double> real_distribution(0.0, 2 * maximum_weight);
	auto index = int_distribution(generator);

	// spin the resample wheel!
	for (int i = 0; i < particles.size(); i++) {
		beta += real_distribution(generator);

		while (weights[index] < beta) {
			beta -= weights[index];
			index = (index + 1) % particles.size();
		}

		new_particles.push_back(particles[index]);
	}

	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
