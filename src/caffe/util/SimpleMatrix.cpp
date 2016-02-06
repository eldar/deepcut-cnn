#include <caffe/util/SimpleMatrix.hpp>
#include <glog/logging.h>

#include <iostream>
#include <fstream>

using std::string;

std::vector<SimpleMatrix*> readMatricesFromFile(std::string filename)
{
    std::ifstream infile(filename.c_str());
    CHECK(infile.good()) << "Failed to open pose file "
        << filename << std::endl;

    string hashtag;
    string matrix_name;

    std::vector<SimpleMatrix*> res;

    infile >> hashtag >> matrix_name;
    do {
        int rows, cols;
        infile >> rows >> cols;
        SimpleMatrix *a = new SimpleMatrix(rows, cols);
        for(int j = 0; j < rows; ++j)
            for(int i = 0; i < cols; ++i)
            {
                float val;
                infile >> val;
                a->val(j, i) = val;
            }
        //LOG(WARNING) << a->rows() << " " << a->cols();
        res.push_back(a);
    } while(infile >> hashtag >> matrix_name);

    return res;
}

