#include <cmath>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>
using namespace std;

vector<double> predictions(vector<double> weight, double bias,
                           vector<vector<double>> features) {
  vector<double> predicted_labels;
  int length = features.size();
  for (size_t i = 0; i < length; i++) {
    double value = 0;
    for (size_t j = 0; j < weight.size(); j++) {
      value += weight[j] * features[i][j];
    }
    value += bias;
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

vector<double> gradient_weight(vector<vector<double>> features,
                               vector<double> labels,
                               vector<double> predicted_labels) {
  vector<double> total_grad_w;
  int n_samples = features.size();
  for (size_t i = 0; i < features[0].size(); i++) {
    double grad_w = 0;
    for (size_t j = 0; j < n_samples; j++) {
      double error = predicted_labels[j] - labels[j];
      grad_w += features[j][i] * error;
    }
    grad_w = (2.0 / n_samples) * grad_w;
    total_grad_w.push_back(grad_w);
  }

  return total_grad_w;
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

vector<double> update_weight(vector<double> old_weight, double learning_rate,
                             vector<double> gradient_weight) {
  vector<double> new_weight;
  double weight = 0;
  for (size_t i = 0; i < old_weight.size(); i++) {
    weight = old_weight[i] - learning_rate * gradient_weight[i];
    new_weight.push_back(weight);
  }

  return new_weight;
}

double update_bias(double bias, double learning_rate, double gradient_bias) {
  bias = bias - learning_rate * gradient_bias;
  return bias;
}

pair<vector<double>, double> Train(vector<vector<double>> features,
                                   vector<double> labels, vector<double> weight,
                                   double bias, double learning_rate,
                                   int epochs) {
  for (size_t i = 0; i < epochs; i++) {
    vector<double> predicted_labels = predictions(weight, bias, features);
    vector<double> grad_w = gradient_weight(features, labels, predicted_labels);
    double grad_b = gradient_bias(labels, predicted_labels);
    weight = update_weight(weight, learning_rate, grad_w);
    bias = update_bias(bias, learning_rate, grad_b);

    if (i % 100 == 0) {
      cout << "Epoch " << i << ", w = [";
      for (size_t j = 0; j < weight.size(); j++) {
        cout << weight[j];
        if (j < weight.size() - 1)
          cout << ", ";
      }
      cout << "], b = " << bias << endl;
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

  cout << "\nFinal Weight: " << w << ", Final Bias: " << b << endl;

  return 0;
}
