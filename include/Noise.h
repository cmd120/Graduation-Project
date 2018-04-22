#ifndef NOISE_H
#define NOISE_H
#include <random>

enum types{NORMAL,UNIFORM};

class Noise
{
private:
	types distribution_type;
	std::random_device rd{};  //Will be used to obtain a seed for the random number engine
	std::mt19937 sed{rd()}; //Standard mersenne_twister_engine seeded with rd()
	std::normal_distribution<double> normalDist;
	//这个好像没有用到过
	std::uniform_int_distribution<int> uniformDist;
public:
	Noise(double mean, double variance) :normalDist(mean, variance) 
	{
		this->distribution_type = NORMAL;
	}
	Noise(int begin, int end) : uniformDist(begin, end) 
	{
		this->distribution_type = UNIFORM;
	}
	double gen() 
	{
		return this->distribution_type == NORMAL ? this->normalDist(this->sed) : this->uniformDist(this->sed);
	}
};

#endif