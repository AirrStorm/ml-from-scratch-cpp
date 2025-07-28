#include <cmath>
#include <iostream>
#include <utility>
#include <vector>
using namespace std;

vector<double> predictions(double weight, double bias,
                           vector<double> features) {
  vector<double> predicted_labels;
  int length = features.size();
  for (size_t i = 0; i < length; i++) {
    double value = weight * features[i] + bias;
    predicted_labels.push_back(value);
  }
  return predicted_labels;
}

double calc_MSE(vector<double> labels, vector<double> predicted_labels) {
  double MSE = 0;
  int n_samples = labels.size();
  for (size_t i = 0; i < n_samples; i++) {
    double squared_error = pow((labels[i] - predicted_labels[i]), 2);
    MSE += squared_error;
  }
  MSE = (1.0 / n_samples) * MSE;

  return MSE;
}

double gradient_weight(vector<double> features, vector<double> labels,
                       vector<double> predicted_labels) {
  double grad_w = 0;
  int n_samples = features.size();
  for (size_t i = 0; i < n_samples; i++) {
    double error = predicted_labels[i] - labels[i];
    grad_w += error * features[i];
  }
  grad_w = (2.0 / n_samples) * grad_w;

  return grad_w;
}

double gradient_bias(vector<double> labels, vector<double> predicted_labels) {
  double grad_b = 0;
  int n_samples = labels.size();
  for (size_t i = 0; i < n_samples; i++) {
    double error = predicted_labels[i] - labels[i];
    grad_b += error;
  }
  grad_b = (2.0 / n_samples) * grad_b;

  return grad_b;
}

double update_weight(double weight, double learning_rate,
                     double gradient_weight) {
  weight = weight - learning_rate * gradient_weight;
  return weight;
}

double update_bias(double bias, double learning_rate, double gradient_bias) {
  bias = bias - learning_rate * gradient_bias;
  return bias;
}

pair<double, double> Train(vector<double> features, vector<double> labels,
                           double weight, double bias, double learning_rate,
                           int epochs) {
  for (size_t i = 0; i < epochs; i++) {
    vector<double> predicted_labels = predictions(weight, bias, features);
    double grad_w = gradient_weight(features, labels, predicted_labels);
    double grad_b = gradient_bias(labels, predicted_labels);
    weight = update_weight(weight, learning_rate, grad_w);
    bias = update_bias(bias, learning_rate, grad_b);
    if (i % 100 == 0) {
      cout << "Epoch " << i << ", w = " << weight << ", b = " << bias << endl;
    }
  }
  return {weight, bias};
}

int main() {

  vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

  vector<double> y = {52.0, 55.5, 61.0, 64.0, 68.0,
                      74.0, 78.0, 83.0, 88.0, 94.0};
  double w = 0.0;
  double b = 0.0;
  double lr = 0.001;
  int epochs = 10000;

  pair<double, double> result = Train(x, y, w, b, lr, epochs);
  w = result.first;
  b = result.second;

  vector<double> predicted_labels = predictions(w, b, x);
  double MSE = calc_MSE(y, predicted_labels);
  cout << "\nFinal Weight: " << w << ", Final Bias: " << b << endl;
  cout << "MSE: " << MSE << endl;
  return 0;
}
