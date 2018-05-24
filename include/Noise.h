#ifndef NOISE_H
#define NOISE_H
#include <random>

typedef enum {
  NORMAL,   ///< normal distribution
  UNIFORM,  ///< uniform distribution
} types;
/// 
/// Class generates noise following specified distribution.
///
class Noise {
 private:
  types distribution_type;  ///< the distribution type of noise
  std::random_device
      rd{};  ///< Will be used to obtain a seed for the random number engine
  std::mt19937 sed{
      rd()};  ///< Standard mersenne_twister_engine seeded with rd()
  std::normal_distribution<double> normalDist;
  std::uniform_int_distribution<int> uniformDist;

 public:
  ///
  /// Normal distribution constructor.
  ///
  Noise(double mean, double variance) : normalDist(mean, variance) {
    this->distribution_type = NORMAL;
  }
  ///
  /// Uniform distribution constructor.
  ///
  Noise(int begin, int end) : uniformDist(begin, end) {
    this->distribution_type = UNIFORM;
  }
  ///
  /// Get the type of distribution.
  ///
  types get_type() { return this->distribution_type; }
  ///
  /// Generate a random noise which follows object's distribution type.
  ///
  double gen() {
    return this->distribution_type == NORMAL ? this->normalDist(this->sed)
                                             : this->uniformDist(this->sed);
  }
};

#endif