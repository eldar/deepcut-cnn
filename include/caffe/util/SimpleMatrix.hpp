#ifndef SIMPLEMATRIX_HPP
#define SIMPLEMATRIX_HPP

#include <vector>
#include <string>

class SimpleMatrix
{
public:
    SimpleMatrix(int rows, int cols) :
        rows_(rows),
        cols_(cols)
    {
        data_.resize(rows_ * cols_);
    }

    int rows() const
    {
        return rows_;
    }

    int cols() const
    {
        return cols_;
    }

    const float val(int j, int i) const
    {
        return data_[j*cols_+i];
    }

    float &val(int j, int i)
    {
        return data_[j*cols_+i];
    }

private:
    int rows_, cols_;
    std::vector<float> data_;
};

std::vector<SimpleMatrix*> readMatricesFromFile(std::string filename);

#endif // SIMPLEMATRIX_HPP
