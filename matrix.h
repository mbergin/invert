#pragma once
#include "eigen/Core"

// A subset of row r from a matrix, selecting columns by the mapping column_map.
template<int N>
Eigen::RowVectorXd row_map(const Eigen::MatrixXd& m, int r, const std::array<int, N> column_map)
{
	auto row = Eigen::RowVectorXd(column_map.size());
	for (size_t c = 0; c < column_map.size(); ++c)
	{
		row[c] = m(r, column_map[c]);
	}
	return row;
}

// A subset of column c from a matrix, selecting rows by the mapping row_map.
template<int N>
Eigen::VectorXd col_map(const Eigen::MatrixXd& m, int c, const std::array<int, N> row_map)
{
	auto col = Eigen::VectorXd(row_map.size());
	for (size_t r = 0; r < row_map.size(); ++r)
	{
		col[r] = m(row_map[r], c);
	}
	return col;
}

// Calculates (A+uv)^-1 given inv=A^-1
// 
// See Sherman, Jack; Morrison, Winifred J. (1949). "Adjustment of an Inverse Matrix Corresponding 
// to Changes in the Elements of a Given Column or a Given Row of the Original Matrix (abstract)". 
// Annals of Mathematical Statistics. 20: 621
inline Eigen::MatrixXd sherman_morrison_update_inverse(const Eigen::MatrixXd& inv, const Eigen::VectorXd& u, const Eigen::RowVectorXd& v)
{
	return inv - (inv * u) * (v * inv) / (1 + v * inv * u);
}