#include"utils.hpp"

void displayMiniBatch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch) {
    std::cout << "Mini-batch contents:\n";
    for (size_t i = 0; i < mini_batch.size(); ++i) {
        std::cout << "Sample " << i + 1 << ":\n";
        std::cout << "  Input: [ ";
        for (int j = 0; j < mini_batch[i].first.size(); ++j) {
            std::cout << mini_batch[i].first(j);
            if (j < mini_batch[i].first.size() - 1) std::cout << ", ";
        }
        std::cout << " ]\n";

        std::cout << "  Output: [ ";
        for (int j = 0; j < mini_batch[i].second.size(); ++j) {
            std::cout << mini_batch[i].second(j);
            if (j < mini_batch[i].second.size() - 1) std::cout << ", ";
        }
        std::cout << " ]\n\n";
    }
}

void displayVectorXd(const Eigen::VectorXd& vec, size_t max_elements) {
    std::cout << "VectorXd (size: " << vec.size() << "):\n[ ";

    size_t limit = (max_elements == 0 || max_elements > vec.size()) ? vec.size() : max_elements;
    for (int i = 0; i < limit; ++i) {
        std::cout << vec(i);
        if (i < limit - 1) std::cout << ", ";
    }

    if (limit < vec.size()) std::cout << ", ..."; // Indicate truncation
    std::cout << " ]\n";
}

void displayMatrixXd(const Eigen::MatrixXd& mat, size_t max_elements) {
    std::cout << "MatrixXd (rows: " << mat.rows() << ", cols: " << mat.cols() << "):\n";

    size_t limit = (max_elements == 0 || max_elements > mat.size()) ? mat.size() : max_elements;
    size_t count = 0;

    for (int i = 0; i < mat.rows(); ++i) {
        std::cout << "[ ";
        for (int j = 0; j < mat.cols(); ++j) {
            if (count < limit) {
                std::cout << mat(i, j);
                if (j < mat.cols() - 1) std::cout << ", ";
                ++count;
            }
            else {
                if (j == 0) std::cout << "...";
                break;
            }
        }
        std::cout << " ]\n";
        if (count >= limit && i < mat.rows() - 1) {
            std::cout << "[ ... ]\n";
            break;
        }
    }
    if (count < mat.size()) std::cout << "...\n";
}