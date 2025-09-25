#pragma once

#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>


void displayMiniBatch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch);

void displayVectorXd(const Eigen::VectorXd& vec, size_t max_elements = 0);

void displayMatrixXd(const Eigen::MatrixXd& mat, size_t max_elements = 0);

#endif




